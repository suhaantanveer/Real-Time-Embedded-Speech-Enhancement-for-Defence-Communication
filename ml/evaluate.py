"""Evaluation suite for the redesigned speech enhancer."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dataset_loader import DefenceAudioDataset
from features import FeatureExtractor
from losses import compute_si_snr, frame_rms, speech_active_frame_mask
from model import TinySpeechEnhancer


def main(args):
    device = torch.device('cpu')
    ckpt = torch.load(args.model, map_location=device)
    model = TinySpeechEnhancer().eval()
    model.load_state_dict(ckpt['model_state_dict'])
    ext = FeatureExtractor()
    ds = DefenceAudioDataset(args.manifest, 'test', args.duration, return_details=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(len(ds)):
        b = ds[i]
        feat = b['features'][None].to(device)
        stft = b['stft'][None].to(device)
        clean = b['clean'][None].to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            gains, _ = model(feat)
            enhanced = ext.apply_gains_and_istft(stft, gains, target_len=clean.shape[-1])
        elapsed = (time.perf_counter() - t0) * 1000 / max(1, feat.shape[1])
        raw = ext.apply_gains_and_istft(stft, torch.ones_like(gains), target_len=clean.shape[-1])

        has_speech = bool(b['has_speech'].item())
        if has_speech:
            raw_si = float(compute_si_snr(raw, clean).mean())
            enh_si = float(compute_si_snr(enhanced, clean).mean())
            delta = enh_si - raw_si
        else:
            raw_si = float('nan')
            enh_si = float('nan')
            delta = float('nan')

        clean_gain = float('nan')
        clean_level = float('nan')
        if bool(b['is_clean_condition'].item()):
            active = speech_active_frame_mask(clean)
            gain_frames = gains.mean(dim=-1)
            if torch.any(active):
                clean_gain = float(gain_frames[active].mean())
                er = frame_rms(enhanced)
                cr = frame_rms(clean)
                clean_level = float((20.0 * torch.log10((er[active] + 1e-8) / (cr[active] + 1e-8))).mean())

        meta = json.loads(b['meta']) if isinstance(b['meta'], str) else b['meta']
        path = out_dir / f"test_{i:04d}.wav"
        sf.write(path, enhanced.squeeze(0).numpy(), 16000, subtype='PCM_16')
        rows.append({
            'id': meta['id'],
            'condition': meta.get('condition', 'unknown'),
            'noise_class': meta.get('noise_class', 'unknown'),
            'snr_db': meta.get('snr_db'),
            'actual_primary_snr_db': meta.get('actual_primary_snr_db'),
            'raw_si_snr': raw_si,
            'enhanced_si_snr': enh_si,
            'delta_si_snr': delta,
            'clean_speech_mean_gain': clean_gain,
            'clean_level_error_db': clean_level,
            'mean_gain': float(gains.mean()),
            'frame_latency_cpu_ms': elapsed,
            'output': str(path),
        })

    if rows:
        p = out_dir / 'evaluation.csv'
        with p.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        valid = [r for r in rows if np.isfinite(r['delta_si_snr'])]
        print(f"speech-sample mean ΔSI-SNR: {np.mean([r['delta_si_snr'] for r in valid]):+.2f} dB" if valid else 'speech-sample mean ΔSI-SNR: n/a')
        clean_rows = [r for r in rows if np.isfinite(r['clean_speech_mean_gain'])]
        print(f"clean-condition speech-frame mean gain: {np.mean([r['clean_speech_mean_gain'] for r in clean_rows]):.3f}" if clean_rows else 'clean-condition gain: n/a')
        print(f"CPU inference per 10 ms frame: {np.mean([r['frame_latency_cpu_ms'] for r in rows]):.3f} ms")
        print(f"saved {p}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--duration', type=float, default=3.0)
    p.add_argument('--output-dir', default='enhanced_output_v3')
    main(p.parse_args())
