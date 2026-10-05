"""Clean-speech level probe for the trained 89-feature model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve()
ML = HERE.parents[1]
sys.path.insert(0, str(ML))
from front_end import MicFrontEnd  # noqa: E402
from features import FeatureExtractor  # noqa: E402
from model import TinySpeechEnhancer  # noqa: E402
from losses import speech_active_frame_mask  # noqa: E402


def proxy(sr=16000, duration=3.0):
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in [(135, .5), (270, .25), (405, .14), (540, .08), (675, .05)])
    env = .65 + .35 * (.5 + .5 * np.sin(2 * np.pi * 2 * t))
    return (x * env / np.max(np.abs(x)) * .45).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--audio')
    p.add_argument('--out', default=str(HERE.parents[2] / 'docs' / 'v3_gain_probe.txt'))
    args = p.parse_args()

    if args.audio:
        x, sr = sf.read(args.audio, dtype='float32')
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            raise SystemExit('audio must be 16 kHz')
    else:
        x = proxy()

    ckpt = torch.load(args.model, map_location='cpu')
    model = TinySpeechEnhancer().eval()
    model.load_state_dict(ckpt['model_state_dict'])
    ext = FeatureExtractor()

    rows = []
    for db in [-50, -45, -40, -35, -30, -25, -20, -15]:
        target = 10 ** (db / 20)
        y = x / (np.sqrt(np.mean(x * x)) + 1e-12) * target
        fe = MicFrontEnd()
        p1, _, _ = fe.process_audio(y, np.zeros_like(y), apply_guard=True)
        with torch.no_grad():
            feat, _, _ = ext.extract_dual(
                torch.from_numpy(p1)[None, :],
                torch.zeros(1, len(p1)),
                torch.tensor([[0.0]])
            )
            gains, _ = model(feat)
            clean = torch.from_numpy(y)[None, :]
            active = speech_active_frame_mask(clean)
            mean_gain = float(gains.mean(dim=-1)[active].mean())
        rows.append((db, mean_gain, float(gains.min()), float(gains.max())))

    text = (
        'V3 clean-level probe\n'
        'Input: ' + ('real ' + args.audio if args.audio else 'synthetic voiced proxy') + '\n\n'
        + ''.join(
            f'{r[0]:>4} dBFS mean_gain={r[1]:.5f} min={r[2]:.5f} max={r[3]:.5f}\n'
            for r in rows
        )
    )
    Path(args.out).write_text(text, encoding='utf-8')
    for r in rows:
        print(f'{r[0]:>4} dBFS | mean {r[1]:.3f} | min {r[2]:.3f} | max {r[3]:.3f}')


if __name__ == '__main__':
    main()
