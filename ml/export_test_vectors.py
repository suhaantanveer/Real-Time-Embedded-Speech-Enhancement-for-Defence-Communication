"""Generate multi-frame Python reference vectors for V3 Python/C equivalence.

The C firmware is a streaming implementation. Therefore the reference is a
sequence of frames, with GRU state carried across frames and a 96-sample
analysis/synthesis latency. The first and final streaming output chunks are
not used for the output-error gate; interior chunks are compared against the
corresponding slice of Python's full causal overlap-add result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from features import FeatureExtractor
from model import TinySpeechEnhancer

SR = 16000
FRAME = 160
N_FRAMES = 32


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Trained checkpoint")
    p.add_argument("--frames", type=int, default=N_FRAMES)
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = root.parent / "firmware" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames = max(4, int(args.frames))
    n = n_frames * FRAME

    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    t = np.arange(n, dtype=np.float32) / SR

    # Deterministic continuous two-channel test stream with changing content.
    mic1 = (
        0.38 * np.sin(2 * np.pi * (440.0 + 30.0 * np.sin(2 * np.pi * 0.4 * t)) * t)
        + 0.16 * np.sin(2 * np.pi * 1200.0 * t)
        + 0.05 * np.sin(2 * np.pi * 2100.0 * t)
        + 0.025 * rng.standard_normal(n)
    ).astype(np.float32)
    mic2 = (
        0.18 * np.sin(2 * np.pi * 330.0 * t)
        + 0.08 * np.sin(2 * np.pi * 920.0 * t)
        + 0.05 * rng.standard_normal(n)
    ).astype(np.float32)

    audio1 = torch.from_numpy(mic1).unsqueeze(0)
    audio2 = torch.from_numpy(mic2).unsqueeze(0)
    ref_valid = torch.ones((1, 1), dtype=torch.float32)

    model = TinySpeechEnhancer().cpu().eval()
    ckpt = torch.load(args.model, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain model_state_dict")
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded trained checkpoint: {args.model}")

    extractor = FeatureExtractor()
    with torch.no_grad():
        features, stft, _ = extractor.extract_dual(audio1, audio2, ref_valid)
        gains, _ = model(features)
        enhanced = extractor.apply_gains_and_istft(stft, gains, target_len=n)

    # Flatten frame-major: [frame0 features, frame1 features, ...].
    feat_np = features.squeeze(0).numpy().astype(np.float32)
    gain_np = gains.squeeze(0).numpy().astype(np.float32)
    stft_np = stft.squeeze(0).numpy().astype(np.complex64)  # [129, frames]
    out_np = enhanced.squeeze(0).numpy().astype(np.float32)

    mic1.tofile(out_dir / "test_mic1_seq.bin")
    mic2.tofile(out_dir / "test_mic2_seq.bin")
    feat_np.tofile(out_dir / "expected_features_89_seq.bin")
    gain_np.tofile(out_dir / "expected_gains_22_seq.bin")
    np.real(stft_np).T.astype(np.float32).tofile(out_dir / "expected_stft_real_129_seq.bin")
    np.imag(stft_np).T.astype(np.float32).tofile(out_dir / "expected_stft_imag_129_seq.bin")
    out_np.tofile(out_dir / "expected_output_seq.bin")

    print(f"[OK] Wrote stateful V3 vectors to {out_dir}")
    print(f"     frames:   {n_frames}")
    print(f"     samples:  {n}")
    print("     features: 89/frame")
    print("     gains:    22/frame")
    print("     STFT:     129 bins/frame")
    print("     output:   full offline Python reference")
    print("     streaming output comparison: interior frames only (96-sample latency aligned)")


if __name__ == "__main__":
    main()
