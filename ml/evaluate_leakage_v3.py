"""
Evaluate speech-leakage robustness of the trained 89-feature Mic-1-primary model.

This test keeps Mic 1 fixed and compares the model's speech-frame mean gain
when Mic 2 is unchanged versus when an additional controlled speech-leakage
component is added at a requested level relative to the clean speech.

Important:
- The stored reference mic already contains whatever natural/random leakage
  was present in the generated dataset.
- Therefore this is an "additional leakage" stress test, not a mathematically
  pure no-leakage-vs-leakage comparison.
- The -6 dB case is the main acceptance stress case.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from config import SAMPLE_RATE
from front_end import MicFrontEnd
from features import FeatureExtractor
from losses import speech_active_frame_mask
from model import TinySpeechEnhancer


def mono_16k(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected 16 kHz WAV: {path}, got {sr} Hz")
    return np.asarray(x, dtype=np.float32)


def center_crop(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        start = (len(x) - n) // 2
        return x[start:start + n].astype(np.float32)
    return np.pad(x, (0, n - len(x))).astype(np.float32)


def add_controlled_leakage(
    reference: np.ndarray,
    clean: np.ndarray,
    leakage_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add a short, near-synchronous speech leakage path to Mic 2."""
    s = np.asarray(clean, dtype=np.float32)
    ref = np.asarray(reference, dtype=np.float32)

    # Short random FIR: acoustical/mechanical coupling rather than a raw copy.
    fir = np.array([0.70, 0.22, -0.10, 0.055, 0.025, -0.012], dtype=np.float32)

    leak = signal.lfilter(fir, [1.0], s)

    # Speech-band limiting.
    sos = signal.butter(
        2, [100.0, 4500.0], btype="bandpass",
        fs=SAMPLE_RATE, output="sos"
    )
    leak = signal.sosfilt(sos, leak).astype(np.float32)

    # Small sub-millisecond delay.
    d = int(rng.integers(0, int(0.0005 * SAMPLE_RATE) + 1))
    if d:
        delayed = np.zeros_like(leak)
        delayed[d:] = leak[:-d]
        leak = delayed

    speech_rms = float(np.sqrt(np.mean(s * s) + 1e-12))
    leak_rms = float(np.sqrt(np.mean(leak * leak) + 1e-12))
    target_rms = speech_rms * (10.0 ** (float(leakage_db) / 20.0))

    if leak_rms > 1e-8:
        leak *= target_rms / leak_rms

    return (ref + leak).astype(np.float32)


def features_and_gain(
    model: TinySpeechEnhancer,
    extractor: FeatureExtractor,
    primary: np.ndarray,
    reference: np.ndarray,
    clean: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    fe = MicFrontEnd(SAMPLE_RATE)
    primary_fe, ref_fe, _ = fe.process_audio(primary, reference, apply_guard=True)

    clean_fe = MicFrontEnd(SAMPLE_RATE)
    clean_target, _, _ = clean_fe.process_audio(
        clean, np.zeros_like(clean), apply_guard=False
    )

    primary_t = torch.from_numpy(primary_fe).unsqueeze(0)
    ref_t = torch.from_numpy(ref_fe).unsqueeze(0)
    clean_t = torch.from_numpy(clean_target).unsqueeze(0)

    with torch.no_grad():
        features, _, _ = extractor.extract_dual(
            primary_t,
            ref_t,
            torch.ones((1, 1), dtype=torch.float32),
        )
        gains, _ = model(features)

    active = speech_active_frame_mask(clean_t)
    frame_gain = gains.mean(dim=-1).squeeze(0)

    return frame_gain.cpu(), active.squeeze(0).cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", default="./ml/leakage_eval_v3")
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=26052)
    p.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=[-12.0, -6.0, 0.0],
        help="Additional Mic-2 speech leakage levels in dB relative to clean speech.",
    )
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.model, map_location=device)
    model = TinySpeechEnhancer().eval()
    model.load_state_dict(ckpt["model_state_dict"])
    extractor = FeatureExtractor()

    with open(args.manifest, "r", encoding="utf-8") as f:
        entries = json.load(f)
    test = [e for e in entries if e.get("split") == "test"]

    n = int(round(args.duration * SAMPLE_RATE))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    rows = []
    for i, entry in enumerate(test):
        clean = center_crop(mono_16k(entry["paths"]["clean"]), n)
        primary = center_crop(mono_16k(entry["paths"]["primary_mic"]), n)
        reference = center_crop(mono_16k(entry["paths"]["ref_mic"]), n)

        base_gain, active = features_and_gain(
            model, extractor, primary, reference, clean
        )

        active_np = active.numpy().astype(bool)
        if not np.any(active_np):
            continue

        base_mean = float(base_gain.numpy()[active_np].mean())

        row = {
            "id": entry["id"],
            "condition": entry.get("condition", ""),
            "noise_class": entry.get("noise_class", ""),
            "actual_primary_snr_db": entry.get("actual_primary_snr_db"),
            "base_speech_mean_gain": base_mean,
        }

        for level in args.levels:
            ref_leaky = add_controlled_leakage(
                reference, clean, level, rng
            )
            leak_gain, leak_active = features_and_gain(
                model, extractor, primary, ref_leaky, clean
            )
            a = leak_active.numpy().astype(bool)
            common = active_np & a
            if not np.any(common):
                row[f"gain_{level:g}db"] = np.nan
                row[f"drop_{level:g}db"] = np.nan
                continue

            lm = float(leak_gain.numpy()[common].mean())
            row[f"gain_{level:g}db"] = lm
            row[f"drop_{level:g}db"] = lm - base_mean

        rows.append(row)

        if (i + 1) % 250 == 0:
            print(f"processed {i + 1}/{len(test)}")

    import csv

    out_csv = out_dir / "leakage_eval.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== ADDITIONAL MIC-2 LEAKAGE TEST ===")
    print(f"samples: {len(rows)}")
    for level in args.levels:
        key = f"drop_{level:g}db"
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        gains = [r[f"gain_{level:g}db"] for r in rows if np.isfinite(r[f"gain_{level:g}db"])]
        print(
            f"added leakage {level:+g} dB: "
            f"speech gain={np.mean(gains):.3f} | "
            f"change vs base={np.mean(vals):+.3f}"
        )

    # Main acceptance stress case: added -6 dB leakage.
    z = [
        r["drop_-6db"]
        for r in rows
        if np.isfinite(r["drop_-6db"])
    ]
    if z:
        print(
            f"\n-6 dB stress case: mean gain change={np.mean(z):+.3f} "
            f"({np.mean(z) * 100.0:+.1f} percentage points)"
        )
        print(
            "Interpretation: the original reference already contains its "
            "dataset-generated leakage, so this measures robustness to an "
            "additional strong leakage component."
        )

    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
