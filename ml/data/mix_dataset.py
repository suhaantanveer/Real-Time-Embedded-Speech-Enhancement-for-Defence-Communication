"""Deterministic SIH26052 Mic-1-primary dataset builder with held-out noise/RIR sources."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

SR = 16000
FRAME_MS = 20.0
TARGET_LEVEL_MIN_DBFS = -45.0
TARGET_LEVEL_MAX_DBFS = -15.0
NOISE_ONLY_LEVEL_MIN_DBFS = -45.0
NOISE_ONLY_LEVEL_MAX_DBFS = -15.0
MIC2_OFFSET_MIN_DB = -3.0
MIC2_OFFSET_MAX_DB = 10.0
MIN_VALID_NOISE_RMS = 1e-6
MIN_VALID_NOISE_PEAK = 1e-5
MAX_NOISE_SAMPLE_ATTEMPTS = 40
SNR_TOL_DB = 0.25

# Preserve the accepted variable leakage distribution: weak leakage is common,
# strong leakage exists often enough to matter, but not in every example.
def sample_leakage_db(rng: np.random.Generator) -> float:
    u = float(rng.random())
    if u < 0.15:
        return float(rng.uniform(-25.0, -18.0))
    if u < 0.50:
        return float(rng.uniform(-18.0, -12.0))
    if u < 0.80:
        return float(rng.uniform(-12.0, -6.0))
    return float(rng.uniform(-6.0, 0.0))


def wav_paths(root: Path) -> list[Path]:
    exts = {".wav", ".flac", ".ogg"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def read_mono(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = np.asarray(x, dtype=np.float32)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def active_rms(x: np.ndarray, frame_ms: float = FRAME_MS, threshold_db: float = -35.0) -> float:
    frame_len = int(round(frame_ms * 1e-3 * SR))
    usable = len(x) - (len(x) % frame_len)
    if usable < frame_len:
        return rms(x)
    frames = x[:usable].reshape(-1, frame_len)
    powers = np.mean(frames.astype(np.float64) ** 2, axis=1)
    peak_power = float(np.max(powers) + 1e-12)
    threshold = peak_power * (10.0 ** (threshold_db / 10.0))
    active = powers > threshold
    if not np.any(active):
        return rms(x)
    return float(np.sqrt(np.mean(powers[active]) + 1e-12))


def active_mask(x: np.ndarray, frame_ms: float = FRAME_MS, threshold_db: float = -35.0) -> np.ndarray:
    frame_len = int(round(frame_ms * 1e-3 * SR))
    usable = len(x) - (len(x) % frame_len)
    if usable < frame_len:
        return np.ones(max(1, len(x) // frame_len), dtype=bool)
    frames = x[:usable].reshape(-1, frame_len)
    powers = np.mean(frames.astype(np.float64) ** 2, axis=1)
    threshold = float(np.max(powers) + 1e-12) * (10.0 ** (threshold_db / 10.0))
    return (powers > threshold)


def match_length(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(x) < n:
        reps = int(np.ceil(n / len(x)))
        x = np.tile(x, reps)
    if len(x) == n:
        return x.astype(np.float32)
    start = int(rng.integers(0, len(x) - n + 1))
    return x[start:start + n].astype(np.float32)


def match_length_speech_active(x: np.ndarray, n: int, rng: np.random.Generator,
                               attempts: int = 20, min_active_fraction: float = 0.20) -> np.ndarray:
    """Randomly crop speech while rejecting crops dominated by silence."""
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n:
        reps = int(np.ceil(n / len(x)))
        return np.tile(x, reps)[:n].astype(np.float32)

    frame_len = int(round(FRAME_MS * 1e-3 * SR))
    best = x[:n].astype(np.float32)
    best_score = -np.inf
    valid: list[np.ndarray] = []
    for _ in range(attempts):
        start = int(rng.integers(0, len(x) - n + 1))
        crop = x[start:start+n]
        usable = len(crop) - (len(crop) % frame_len)
        if usable < frame_len:
            continue
        frames = crop[:usable].reshape(-1, frame_len)
        powers = np.mean(frames.astype(np.float64) ** 2, axis=1)
        threshold = float(np.max(powers) + 1e-12) * (10.0 ** (-35.0 / 10.0))
        active = powers > threshold
        active_fraction = float(np.mean(active))
        if np.any(active):
            active_level = float(np.sqrt(np.mean(powers[active]) + 1e-12))
        else:
            active_level = 0.0
        score = active_fraction + 0.05 * math.log10(active_level + 1e-12)
        if active_fraction >= min_active_fraction:
            valid.append(crop.astype(np.float32))
        if score > best_score:
            best_score = score
            best = crop.astype(np.float32)
    if valid:
        return valid[int(rng.integers(0, len(valid)))]
    return best


def random_delay(x: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0:
        return x
    if samples >= len(x):
        return np.zeros_like(x)
    return np.concatenate([np.zeros(samples, dtype=np.float32), x[:-samples]])


def apply_rir(x: np.ndarray, rir: np.ndarray | None) -> np.ndarray:
    if rir is None:
        return x.astype(np.float32)
    r = np.asarray(rir, dtype=np.float32)
    if r.size == 0 or np.max(np.abs(r)) <= 1e-8:
        return x.astype(np.float32)
    r = r / (np.sqrt(np.sum(r * r)) + 1e-12)
    return fftconvolve(x, r, mode="full")[:len(x)].astype(np.float32)


def early_rir(rir: np.ndarray | None, max_ms: float = 50.0, search_ms: float = 300.0) -> np.ndarray | None:
    """Trim an RIR from its dominant arrival rather than blindly from file index 0."""
    if rir is None:
        return None
    r = np.asarray(rir, dtype=np.float32)
    if r.size == 0 or np.max(np.abs(r)) <= 1e-8:
        return None
    search_n = min(len(r), max(1, int(round(search_ms * 1e-3 * SR))))
    peak = int(np.argmax(np.abs(r[:search_n])))
    n = min(len(r) - peak, max(1, int(round(max_ms * 1e-3 * SR))))
    out = r[peak:peak+n].copy()
    return out.astype(np.float32)


def apply_leakage(clean: np.ndarray, gain_db: float, rng: np.random.Generator) -> np.ndarray:
    gain = 10.0 ** (gain_db / 20.0)
    taps = int(rng.integers(4, 17))
    fir = rng.normal(0.0, 1.0, taps).astype(np.float32)
    fir *= np.exp(-np.arange(taps, dtype=np.float32) / max(1.0, taps / 2.0))
    fir /= np.sum(np.abs(fir)) + 1e-12
    return (gain * fftconvolve(clean, fir, mode="same")).astype(np.float32)


def scale_noise_to_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_level = active_rms(speech)
    noise_level = rms(noise)
    desired_noise = speech_level / (10.0 ** (snr_db / 20.0))
    return noise * (desired_noise / (noise_level + 1e-12))


def scale_to_rms(x: np.ndarray, target_rms: float) -> np.ndarray:
    return x * (target_rms / (rms(x) + 1e-12))


def valid_noise(x: np.ndarray) -> bool:
    """Reject a crop that is effectively silent after its acoustic path."""
    return (rms(x) >= MIN_VALID_NOISE_RMS and
            float(np.max(np.abs(x))) >= MIN_VALID_NOISE_PEAK)


def sample_valid_noise(
    class_pool: list[Path],
    rir_pool: list[Path],
    n: int,
    rng: np.random.Generator,
    *,
    attempts: int = MAX_NOISE_SAMPLE_ATTEMPTS,
) -> tuple[Path, np.ndarray, np.ndarray | None, np.ndarray, int]:
    """Pick a noise clip/crop and acoustic path with usable energy.

    Impulsive clips can contain long silent regions. A uniformly selected crop can
    therefore be effectively silent even though the source file is valid. We
    reject those crops before SNR scaling so the requested SNR is physically
    achievable instead of relying on a tiny epsilon in the denominator.
    """
    last_path: Path | None = None
    last_filtered = np.zeros(n, dtype=np.float32)
    for attempt in range(1, attempts + 1):
        nz = class_pool[int(rng.integers(0, len(class_pool)))]
        noise_sig = match_length(read_mono(nz), n, rng)
        rir = read_mono(Path(rng.choice(rir_pool))) if rir_pool else None
        filtered = apply_rir(noise_sig, rir)
        last_path = nz
        last_filtered = filtered
        if valid_noise(filtered):
            return nz, noise_sig, rir, filtered, attempt

    raise RuntimeError(
        f"Could not find a non-silent noise crop after {attempts} attempts; "
        f"last source={last_path}, rms={rms(last_filtered):.3e}, "
        f"peak={float(np.max(np.abs(last_filtered))):.3e}"
    )


def make_valid_mic2_noise(
    noise_sig: np.ndarray,
    rir_pool: list[Path],
    rng: np.random.Generator,
    target_rms: float,
    *,
    attempts: int = MAX_NOISE_SAMPLE_ATTEMPTS,
) -> tuple[np.ndarray, int]:
    """Pass the same underlying noise through a different usable Mic-2 path."""
    if target_rms <= 0.0:
        return np.zeros_like(noise_sig), 0
    for attempt in range(1, attempts + 1):
        rir = read_mono(Path(rng.choice(rir_pool))) if rir_pool else None
        filtered = apply_rir(noise_sig, rir)
        if valid_noise(filtered):
            return scale_to_rms(filtered, target_rms), attempt
    # With no RIR directory, fall back to the direct noise signal if it is valid.
    if not rir_pool and valid_noise(noise_sig):
        return scale_to_rms(noise_sig, target_rms), 1
    raise RuntimeError(
        f"Could not find a usable Mic-2 acoustic path after {attempts} attempts"
    )


def noise_class(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        return rel.parts[0] if len(rel.parts) >= 2 else "unknown"
    except ValueError:
        return "unknown"


def stable_hash(path: Path) -> int:
    return int(hashlib.sha256(str(path).replace(chr(92), "/").encode("utf-8")).hexdigest()[:16], 16)


def split_paths_stratified(paths: list[Path], group_fn) -> dict[str, list[Path]]:
    """Deterministically partition each group into disjoint 80/10/10 pools."""
    groups: dict[str, list[Path]] = {}
    for p in paths:
        groups.setdefault(group_fn(p), []).append(p)
    out = {"train": [], "val": [], "test": []}
    for group, items in groups.items():
        items = sorted(items, key=stable_hash)
        n = len(items)
        if n < 3:
            out["train"].extend(items)
            continue
        n_val = max(1, int(round(n * 0.10)))
        n_test = max(1, int(round(n * 0.10)))
        if n_val + n_test >= n:
            n_val = 1
            n_test = 1
        out["test"].extend(items[:n_test])
        out["val"].extend(items[n_test:n_test+n_val])
        out["train"].extend(items[n_test+n_val:])
    return {k: sorted(v, key=stable_hash) for k, v in out.items()}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_speech_records(speech_root: Path, metadata_path: Path | None) -> dict[str, list[dict]]:
    if metadata_path is None:
        metadata_path = speech_root / "metadata" / "speech.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Speech metadata not found: {metadata_path}")

    meta_root = metadata_path.parent.parent
    rows = read_csv(metadata_path)
    by_path: dict[str, dict] = {}
    for row in rows:
        if row.get("quality_status") not in {"KEEP", "REVIEW"}:
            continue
        rel = row.get("output_path", "").strip()
        if not rel:
            continue
        abs_path = (meta_root / rel).resolve()
        if not abs_path.exists():
            continue
        rec = dict(row)
        rec["_path"] = str(abs_path)
        by_path[str(abs_path)] = rec

    pools = {"train": [], "val": [], "test": []}
    split_files = {
        "train": metadata_path.parent / "speaker_disjoint_public_train.csv",
        "val": metadata_path.parent / "speaker_disjoint_public_val.csv",
        "test": metadata_path.parent / "speaker_disjoint_public_test.csv",
    }
    for split, manifest in split_files.items():
        if manifest.exists():
            for row in read_csv(manifest):
                rel = row.get("output_path", "").strip()
                if not rel:
                    continue
                abs_path = (meta_root / rel).resolve()
                rec = by_path.get(str(abs_path))
                if rec is not None and rec.get("source") == "librispeech":
                    rec = dict(rec)
                    rec["split"] = split
                    pools[split].append(rec)

    for rec in by_path.values():
        if rec.get("source") == "common_voice_hindi" and rec.get("speaker_id") == "UNKNOWN_HI":
            rec = dict(rec)
            rec["split"] = "train_only_unknown_speaker"
            pools["train"].append(rec)

    if not pools["train"] or not pools["val"] or not pools["test"]:
        raise RuntimeError("Speech pools are incomplete; expected LibriSpeech public split manifests.")
    return pools


def snr_bucket(snr_db: float) -> str:
    if snr_db <= 0:
        return "<=0"
    if snr_db < 10:
        return "0_10"
    if snr_db < 20:
        return "10_20"
    return ">=20"


def build(args: argparse.Namespace) -> Path:
    speech_root = Path(args.speech_dir)
    noise_root = Path(args.noise_dir)
    rir_root = Path(args.rir_dir) if args.rir_dir else None
    out = Path(args.out_dir)
    audio_out = out / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    speech_pools = load_speech_records(speech_root, Path(args.speech_metadata) if args.speech_metadata else None)
    all_noise = wav_paths(noise_root)
    if not all_noise:
        raise RuntimeError("Need at least one noise file")

    noise_split = split_paths_stratified(
        all_noise,
        lambda p: noise_class(p, noise_root),
    )

    rir_split = {"train": [], "val": [], "test": []}
    if rir_root:
        all_rir = wav_paths(rir_root)
        rir_split = split_paths_stratified(
            all_rir,
            lambda p: p.parts[len(rir_root.parts)] if len(p.parts) > len(rir_root.parts) else "unknown",
        )

    # Sample the four intended noise families evenly. Keep uncertain/review clips
    # at a small fixed probability instead of letting them disappear or dominate.
    classes = ["stationary", "nonstationary", "impulsive", "ambient"]
    per_split_class: dict[str, dict[str, list[Path]]] = {s: {} for s in ("train", "val", "test")}
    for split in per_split_class:
        for p in noise_split[split]:
            cls = noise_class(p, noise_root)
            per_split_class[split].setdefault(cls, []).append(p)

    entries = []
    global_idx = 0
    split_seed_offset = {"train": 11, "val": 22, "test": 33}

    for split, n_samples in {
        "train": args.train_samples,
        "val": args.val_samples,
        "test": args.test_samples,
    }.items():
        pool = speech_pools[split]
        split_rng = np.random.default_rng(args.seed + split_seed_offset[split])
        available_base = [c for c in classes if per_split_class[split].get(c)]
        if not available_base:
            raise RuntimeError(f"No canonical noise classes available in {split}")

        for _ in range(n_samples):
            sp_rec = pool[int(split_rng.integers(0, len(pool)))]
            chosen_cls = available_base[int(split_rng.integers(0, len(available_base)))]
            class_pool = per_split_class[split][chosen_cls]
            # The manually tagged `review` pool is not trusted for training until
            # it has been audited for speech contamination, so it is excluded here.

            rir_pool = rir_split[split]
            if not rir_pool and rir_root:
                raise RuntimeError(f"No RIRs available in {split}")

            duration = int(round(args.duration * SR))
            clean_raw = match_length_speech_active(read_mono(Path(sp_rec["_path"])), duration, split_rng)
            # Reject effectively silent noise crops before any SNR scaling.
            # This specifically protects impulsive sources whose events occupy only
            # a small part of the original recording.
            (nz, noise_sig, rir_noise1, filtered_noise1, noise_attempts) = sample_valid_noise(
                class_pool, rir_pool, duration, split_rng
            )
            rir_speech = early_rir(read_mono(Path(split_rng.choice(rir_pool))) if rir_pool else None)

            bucket = float(split_rng.random())
            if bucket < 0.55:
                condition = "noisy"
                target_snr = float(split_rng.uniform(-5.0, 20.0))
            elif bucket < 0.75:
                condition = "near_clean"
                target_snr = float(split_rng.uniform(20.0, 40.0))
            elif bucket < 0.90:
                condition = "clean"
                target_snr = 60.0
            else:
                condition = "noise_only"
                target_snr = -60.0

            speech_m1 = apply_rir(clean_raw, rir_speech)
            target = speech_m1.copy()

            if condition == "noise_only":
                target = np.zeros_like(clean_raw)
                speech_m1 = np.zeros_like(clean_raw)
                noise_level_db = float(split_rng.uniform(NOISE_ONLY_LEVEL_MIN_DBFS, NOISE_ONLY_LEVEL_MAX_DBFS))
                mic1_noise = scale_to_rms(filtered_noise1, 10.0 ** (noise_level_db / 20.0))
                mic1 = mic1_noise.copy()
                speech_target_level_db = None
            else:
                mic1_noise = (np.zeros_like(filtered_noise1) if condition == "clean"
                              else scale_noise_to_snr(speech_m1, filtered_noise1, target_snr))
                speech_target_level_db = float(split_rng.uniform(TARGET_LEVEL_MIN_DBFS, TARGET_LEVEL_MAX_DBFS))
                level_gain = 10.0 ** (speech_target_level_db / 20.0) / (active_rms(speech_m1) + 1e-12)
                speech_m1 *= level_gain
                target = speech_m1.copy()
                mic1_noise *= level_gain

                # Hard SNR correction after all level scaling. This makes the
                # requested SNR an invariant of the generated example, rather
                # than merely a target that can fail on pathological source crops.
                desired_noise_rms = active_rms(target) / (10.0 ** (target_snr / 20.0))
                mic1_noise = scale_to_rms(mic1_noise, desired_noise_rms)
                mic1 = speech_m1 + mic1_noise

            # Mic 2 noise is tied to the actual Mic-1 noise component, plus a physically
            # plausible path-level offset. Clean-primary cases therefore do not acquire
            # an unrelated loud environmental noise source on Mic 2.
            m1_noise_rms = rms(mic1_noise)
            if m1_noise_rms > 1e-10:
                offset_db = float(split_rng.uniform(MIC2_OFFSET_MIN_DB, MIC2_OFFSET_MAX_DB))
                target_m2_rms = m1_noise_rms * 10.0 ** (offset_db / 20.0)
                noise_m2, mic2_noise_attempts = make_valid_mic2_noise(
                    noise_sig, rir_pool, split_rng, target_m2_rms
                )
            else:
                offset_db = 0.0
                mic2_noise_attempts = 0
                noise_m2 = np.zeros_like(noise_sig)
            noise_m2 = random_delay(noise_m2, int(split_rng.integers(0, int(0.002 * SR) + 1)))

            leakage_db = sample_leakage_db(split_rng)
            leak = apply_leakage(speech_m1, leakage_db, split_rng) if condition != "noise_only" else np.zeros_like(clean_raw)
            mic2 = noise_m2 + leak

            # Preserve relative levels; only apply a transparent global scale if the
            # synthesized waveform would exceed the WAV safety ceiling. Record it.
            max_abs = max(float(np.max(np.abs(mic1))), float(np.max(np.abs(mic2))), float(np.max(np.abs(target))))
            mix_scale_db = 0.0
            if max_abs > 0.98:
                scale = 0.98 / max_abs
                mix_scale_db = float(20.0 * math.log10(scale))
                mic1 *= scale
                mic2 *= scale
                target *= scale
                mic1_noise *= scale

            actual_snr = None
            snr_error_db = None
            # SNR is physically meaningful for noisy and near-clean mixtures.
            # It is intentionally not measured for the clean condition because its
            # noise component is exactly zero (infinite SNR).
            if condition in {"noisy", "near_clean"}:
                noise_residual = mic1 - target
                actual_snr = float(10.0 * math.log10(
                    (active_rms(target) ** 2 + 1e-12) / (rms(noise_residual) ** 2 + 1e-12)
                ))
                snr_error_db = float(actual_snr - target_snr)
                if abs(snr_error_db) > SNR_TOL_DB:
                    raise RuntimeError(
                        f"SNR invariant violated for {nz}: target={target_snr:.2f} dB, "
                        f"actual={actual_snr:.2f} dB, error={snr_error_db:.2f} dB"
                    )

            stem = f"{split}_{global_idx:07d}"
            global_idx += 1
            p_clean = audio_out / f"{stem}_clean.wav"
            p_primary = audio_out / f"{stem}_primary.wav"
            p_ref = audio_out / f"{stem}_ref.wav"
            sf.write(p_clean, target.astype(np.float32), SR, subtype="PCM_16")
            sf.write(p_primary, mic1.astype(np.float32), SR, subtype="PCM_16")
            sf.write(p_ref, mic2.astype(np.float32), SR, subtype="PCM_16")
            entries.append({
                "id": stem,
                "split": split,
                "speech_source": sp_rec["_path"],
                "speech_source_dataset": sp_rec.get("source", ""),
                "speaker_id": sp_rec.get("speaker_id", ""),
                "language": sp_rec.get("language", ""),
                "noise_class": chosen_cls,
                "noise_source": str(nz),
                "snr_db": target_snr,
                "actual_primary_snr_db": actual_snr,
                "snr_error_db": snr_error_db,
                "noise_attempts": noise_attempts,
                "condition": condition,
                "target_level_dbfs": speech_target_level_db,
                "mic2_noise_offset_db": offset_db,
                "mic2_noise_attempts": mic2_noise_attempts,
                "leakage_db": leakage_db,
                "mix_scale_db": mix_scale_db,
                "paths": {"clean": str(p_clean), "primary_mic": str(p_primary), "ref_mic": str(p_ref)},
            })

    manifest = out / "manifest.json"
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} examples to {manifest}")
    for split in ("train", "val", "test"):
        subset=[e for e in entries if e["split"]==split]
        print(f"  {split}: {len(subset)}")
        print(f"    noise: {sorted({e['noise_class'] for e in subset})}")
    print(f"  noise source overlap: {len(set(noise_split['train']) & set(noise_split['val']) | set(noise_split['train']) & set(noise_split['test']) | set(noise_split['val']) & set(noise_split['test']))}")
    if rir_root:
        overlap = ((set(rir_split['train']) & set(rir_split['val'])) |
                   (set(rir_split['train']) & set(rir_split['test'])) |
                   (set(rir_split['val']) & set(rir_split['test'])))
        print(f"  RIR source overlap: {len(overlap)}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--speech-dir", required=True)
    p.add_argument("--speech-metadata", default=None)
    p.add_argument("--noise-dir", required=True)
    p.add_argument("--rir-dir", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--train-samples", type=int, default=None)
    p.add_argument("--val-samples", type=int, default=None)
    p.add_argument("--test-samples", type=int, default=None)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    if args.samples is not None and any(v is not None for v in (args.train_samples, args.val_samples, args.test_samples)):
        p.error("Use either --samples or explicit split counts, not both.")
    if args.samples is not None:
        args.train_samples = max(1, int(round(args.samples * 0.80)))
        args.val_samples = max(1, int(round(args.samples * 0.10)))
        args.test_samples = max(1, args.samples - args.train_samples - args.val_samples)
    else:
        args.train_samples = args.train_samples if args.train_samples is not None else 1000
        args.val_samples = args.val_samples if args.val_samples is not None else 200
        args.test_samples = args.test_samples if args.test_samples is not None else 200
    build(args)


if __name__ == "__main__":
    main()
