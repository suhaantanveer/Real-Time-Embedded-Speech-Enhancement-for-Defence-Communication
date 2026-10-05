"""V3 clean-speech gain probe.

Tests the current 89-feature V3 model and reports:
- mean/p10/p50/p90 predicted gain
- output RMS vs input RMS
- actual loaded checkpoint/config

This is a model-level regression probe, not a full ESP32 hardware test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve()
ML_DIR = HERE.parents[1]

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from config import G_MIN, N_FEATURES, NUM_BANDS  # noqa: E402
from features import FeatureExtractor  # noqa: E402
from model import TinySpeechEnhancer  # noqa: E402


LEVELS_DBFS = [-50, -45, -40, -35, -30, -25, -20, -15]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


def rms_dbfs(x: np.ndarray) -> float:
    return 20.0 * np.log10(rms(x))


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sr} Hz: {path}")

    return audio.astype(np.float32)


def pick_audio(audio_dir: Path) -> Path:
    wavs = sorted(audio_dir.rglob("*.wav"))

    if not wavs:
        raise FileNotFoundError(f"No WAV files found under {audio_dir}")

    return wavs[0]


def scale_to_dbfs(audio: np.ndarray, level_dbfs: float) -> np.ndarray:
    target_rms = 10.0 ** (level_dbfs / 20.0)
    current_rms = rms(audio)

    return (audio / current_rms * target_rms).astype(np.float32)


def load_checkpoint(model: TinySpeechEnhancer, checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return ckpt


@torch.no_grad()
def run_probe(
    model: TinySpeechEnhancer,
    extractor: FeatureExtractor,
    audio: np.ndarray,
    levels: list[float],
) -> list[dict[str, float]]:

    rows = []

    for level in levels:
        scaled = scale_to_dbfs(audio, level)

        x = torch.from_numpy(scaled).unsqueeze(0)

        # Current V3 pipeline:
        # clean speech -> Mic1 only -> ref_valid = 0
        features, stft, _ = extractor.extract_dual(
            x,
            mic2=None,
            ref_valid=None,
        )

        gains, _ = model(features)

        # [1, frames, 22]
        gain_np = gains.squeeze(0).cpu().numpy()

        # Reconstruct enhanced waveform using the actual V3 ISTFT path.
        enhanced = extractor.apply_gains_and_istft(
            stft,
            gains,
            target_len=scaled.shape[0],
        )

        enhanced_np = enhanced.squeeze(0).cpu().numpy()

        # Trim to common length just in case.
        n = min(len(scaled), len(enhanced_np))
        scaled = scaled[:n]
        enhanced_np = enhanced_np[:n]

        in_rms = rms(scaled)
        out_rms = rms(enhanced_np)

        rows.append(
            {
                "level_dbfs": float(level),
                "mean_gain": float(np.mean(gain_np)),
                "p10_gain": float(np.percentile(gain_np, 10)),
                "p50_gain": float(np.percentile(gain_np, 50)),
                "p90_gain": float(np.percentile(gain_np, 90)),
                "min_gain": float(np.min(gain_np)),
                "max_gain": float(np.max(gain_np)),
                "input_rms_dbfs": float(20.0 * np.log10(in_rms)),
                "output_rms_dbfs": float(20.0 * np.log10(out_rms)),
                "rms_change_db": float(
                    20.0 * np.log10(out_rms / in_rms)
                ),
            }
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        default=ML_DIR / "checkpoints_full_v3" / "best_tiny_enhancer.pth",
    )

    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="One 16 kHz clean speech WAV",
    )

    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory containing clean speech WAV files",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=ML_DIR.parent / "docs" / "baseline_probe_v3.txt",
    )

    args = parser.parse_args()

    if args.audio is not None:
        audio_path = args.audio
    elif args.audio_dir is not None:
        audio_path = pick_audio(args.audio_dir)
    else:
        raise SystemExit(
            "Provide --audio <file.wav> or --audio-dir <directory>"
        )

    audio = load_audio(audio_path)

    model = TinySpeechEnhancer()

    if model.num_features != N_FEATURES:
        raise RuntimeError(
            f"Model expects {model.num_features} features, "
            f"but config says {N_FEATURES}"
        )

    if model.num_bands != NUM_BANDS:
        raise RuntimeError(
            f"Model outputs {model.num_bands} bands, "
            f"but config says {NUM_BANDS}"
        )

    ckpt = load_checkpoint(model, args.model)

    extractor = FeatureExtractor()

    rows = run_probe(
        model,
        extractor,
        audio,
        LEVELS_DBFS,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        f.write("V3 clean-speech gain probe\n")
        f.write("==========================\n\n")
        f.write(f"Checkpoint: {args.model}\n")
        f.write(f"Input audio: {audio_path}\n")
        f.write(f"Epoch: {ckpt.get('epoch')}\n")
        f.write(f"Model features: {N_FEATURES}\n")
        f.write(f"Model bands: {NUM_BANDS}\n")
        f.write(f"G_MIN: {G_MIN}\n")
        f.write("\n")
        f.write(
            "level_dBFS mean_gain p10_gain p50_gain p90_gain "
            "min_gain max_gain input_dBFS output_dBFS rms_change_dB\n"
        )

        for r in rows:
            f.write(
                f"{r['level_dbfs']:>8.0f} "
                f"{r['mean_gain']:.6f} "
                f"{r['p10_gain']:.6f} "
                f"{r['p50_gain']:.6f} "
                f"{r['p90_gain']:.6f} "
                f"{r['min_gain']:.6f} "
                f"{r['max_gain']:.6f} "
                f"{r['input_rms_dbfs']:.2f} "
                f"{r['output_rms_dbfs']:.2f} "
                f"{r['rms_change_db']:.2f}\n"
            )

    print()
    print("=== V3 GAIN PROBE ===")
    print(f"Checkpoint : {args.model}")
    print(f"Epoch      : {ckpt.get('epoch')}")
    print(f"Features   : {N_FEATURES}")
    print(f"Bands      : {NUM_BANDS}")
    print(f"G_MIN      : {G_MIN}")
    print(f"Audio      : {audio_path}")
    print()

    print(
        " Level | mean  |  p10  |  p50  |  p90  | "
        "out dBFS | change"
    )
    print("-" * 65)

    for r in rows:
        print(
            f"{r['level_dbfs']:>5.0f} | "
            f"{r['mean_gain']:.3f} | "
            f"{r['p10_gain']:.3f} | "
            f"{r['p50_gain']:.3f} | "
            f"{r['p90_gain']:.3f} | "
            f"{r['output_rms_dbfs']:.2f} | "
            f"{r['rms_change_db']:+.2f} dB"
        )

    print()
    print(f"Saved: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())