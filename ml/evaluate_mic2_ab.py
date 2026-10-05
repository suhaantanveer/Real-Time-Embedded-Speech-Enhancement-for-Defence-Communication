"""A/B evaluation of the trained model with Mic 2 present vs forced absent.
Run from the handoff_repo root:
  python .\ml\evaluate_mic2_ab.py --model .\ml\checkpoints_full_v3\best_tiny_enhancer.pth --manifest .\dataset\generated_v3\manifest.json --output-dir .\ml\mic2_ab_v3
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from dataset_loader import DefenceAudioDataset
from features import FeatureExtractor
from losses import compute_si_snr
from model import TinySpeechEnhancer


def evaluate_case(model, ext, ds, device):
    rows = []
    for i in range(len(ds)):
        b = ds[i]
        feat = b["features"][None].to(device)
        stft = b["stft"][None].to(device)
        clean = b["clean"][None].to(device)
        with torch.no_grad():
            gains, _ = model(feat)
            enhanced = ext.apply_gains_and_istft(stft, gains, target_len=clean.shape[-1])
            raw = ext.apply_gains_and_istft(stft, torch.ones_like(gains), target_len=clean.shape[-1])

        has_speech = bool(b["has_speech"].item())
        if has_speech:
            raw_si = float(compute_si_snr(raw, clean).mean())
            enh_si = float(compute_si_snr(enhanced, clean).mean())
            delta = enh_si - raw_si
        else:
            raw_si = float("nan")
            enh_si = float("nan")
            delta = float("nan")

        meta = json.loads(b["meta"]) if isinstance(b["meta"], str) else b["meta"]
        rows.append({
            "id": meta["id"],
            "condition": meta.get("condition", "unknown"),
            "noise_class": meta.get("noise_class", "unknown"),
            "actual_primary_snr_db": meta.get("actual_primary_snr_db"),
            "raw_si_snr": raw_si,
            "enhanced_si_snr": enh_si,
            "delta_si_snr": delta,
        })
    return rows


def mean_finite(rows, key):
    vals = [float(r[key]) for r in rows if np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(rows, label):
    speech = [r for r in rows if np.isfinite(r["delta_si_snr"])]
    print(f"\n=== {label} ===")
    print(f"speech samples: {len(speech)}")
    print(f"mean ΔSI-SNR: {mean_finite(speech, 'delta_si_snr'):+.3f} dB")

    buckets = [
        (-5.0, 0.0, "-5..0"),
        (0.0, 5.0, "0..5"),
        (5.0, 10.0, "5..10"),
        (10.0, 15.0, "10..15"),
        (15.0, 20.0, "15..20"),
    ]
    print("SNR bucket | N | ΔSI-SNR")
    for lo, hi, name in buckets:
        sub = [r for r in speech if lo <= float(r["actual_primary_snr_db"]) < hi]
        print(f"{name:>8} | {len(sub):4d} | {mean_finite(sub, 'delta_si_snr'):+.3f} dB")


def main(args):
    device = torch.device("cpu")
    ckpt = torch.load(args.model, map_location=device)
    model = TinySpeechEnhancer().eval()
    model.load_state_dict(ckpt["model_state_dict"])
    ext = FeatureExtractor()

    ds_on = DefenceAudioDataset(args.manifest, "test", args.duration, return_details=True, force_ref_invalid=False)
    ds_off = DefenceAudioDataset(args.manifest, "test", args.duration, return_details=True, force_ref_invalid=True)
    if len(ds_on) != len(ds_off):
        raise RuntimeError("Mic2-on/off datasets differ in length")

    print(f"Evaluating {len(ds_on)} paired test examples...")
    on = evaluate_case(model, ext, ds_on, device)
    off = evaluate_case(model, ext, ds_off, device)
    if any(a["id"] != b["id"] for a, b in zip(on, off)):
        raise RuntimeError("Paired IDs do not match")

    summarize(on, "MIC2 PRESENT")
    summarize(off, "MIC2 ABSENT")

    paired = []
    for a, b in zip(on, off):
        if np.isfinite(a["delta_si_snr"]) and np.isfinite(b["delta_si_snr"]):
            paired.append({
                "id": a["id"],
                "condition": a["condition"],
                "noise_class": a["noise_class"],
                "actual_primary_snr_db": a["actual_primary_snr_db"],
                "delta_with_mic2_db": a["delta_si_snr"],
                "delta_without_mic2_db": b["delta_si_snr"],
                "mic2_gain_db": a["delta_si_snr"] - b["delta_si_snr"],
            })

    def sub_by_snr(lo, hi):
        return [r for r in paired if lo <= float(r["actual_primary_snr_db"]) < hi]

    low = [r for r in paired if float(r["actual_primary_snr_db"]) <= 0.0]
    high = [r for r in paired if float(r["actual_primary_snr_db"]) >= 10.0]
    print("\n=== MIC2 VALUE TEST ===")
    print(f"SNR <= 0 dB:  N={len(low)}  Mic2 advantage={mean_finite(low, 'mic2_gain_db'):+.3f} dB")
    print(f"SNR >=10 dB: N={len(high)}  Mic2 advantage={mean_finite(high, 'mic2_gain_db'):+.3f} dB")
    print("Target reference: low-SNR Mic2 advantage >= +1.5 dB; high-SNR difference within ±1 dB.")

    print("\n=== MIC2 ADVANTAGE BY NOISE CLASS ===")
    for cls in ["ambient", "impulsive", "nonstationary", "stationary"]:
        sub = [r for r in low if r["noise_class"] == cls]
        if sub:
            print(f"{cls:14} N={len(sub):4d} advantage={mean_finite(sub, 'mic2_gain_db'):+.3f} dB")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "mic2_ab.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=paired[0].keys())
        w.writeheader()
        w.writerows(paired)
    print(f"\nSaved {out / 'mic2_ab.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--output-dir", default="ml/mic2_ab_v3")
    main(p.parse_args())
