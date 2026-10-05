
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from pystoi import stoi


def read_mono(path: Path):
    x, fs = sf.read(path, always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


def align(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--output-csv", default="results/stoi_v3/stoi_results.csv")
    args = ap.parse_args()

    root = Path(args.root)
    manifest = json.loads(
        Path(args.manifest).read_text(encoding="utf-8")
    )

    by_id = {e["id"]: e for e in manifest}

    df = pd.read_csv(args.csv)

    rows = []
    skipped = 0

    for _, r in df.iterrows():
        sample_id = r["id"]
        condition = str(r["condition"])

        # STOI needs a speech target, so skip pure noise-only examples.
        if condition == "noise_only":
            skipped += 1
            continue

        e = by_id.get(sample_id)
        if e is None:
            skipped += 1
            continue

        clean_path = root / Path(e["paths"]["clean"])
        primary_path = root / Path(e["paths"]["primary_mic"])
        enhanced_path = root / Path(str(r["output"]))

        try:
            clean, fs_c = read_mono(clean_path)
            primary, fs_p = read_mono(primary_path)
            enhanced, fs_e = read_mono(enhanced_path)

            if not (fs_c == fs_p == fs_e == 16000):
                raise ValueError(
                    f"sample rates {fs_c}/{fs_p}/{fs_e}, expected 16000"
                )

            clean_p, primary = align(clean, primary)
            clean_e, enhanced = align(clean, enhanced)

            # Standard (non-extended) STOI.
            s_in = float(stoi(clean_p, primary, 16000, extended=False))
            s_out = float(stoi(clean_e, enhanced, 16000, extended=False))

            rows.append({
                "id": sample_id,
                "condition": condition,
                "noise_class": e.get("noise_class", ""),
                "snr_db": float(e.get("snr_db", np.nan)),
                "stoi_input": s_in,
                "stoi_output": s_out,
                "delta_stoi": s_out - s_in,
            })
        except Exception as exc:
            skipped += 1
            print(f"[WARN] {sample_id}: {exc}")

    out = pd.DataFrame(rows)

    if out.empty:
        raise SystemExit("No STOI rows were computed.")

    out_path = root / args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    speech = out[out["condition"].isin(["noisy", "near_clean", "clean"])]

    print(f"STOI rows           : {len(out)}")
    print(f"Skipped             : {skipped}")
    print(f"Mean input STOI     : {speech['stoi_input'].mean():.4f}")
    print(f"Mean output STOI    : {speech['stoi_output'].mean():.4f}")
    print(f"Mean ΔSTOI          : {speech['delta_stoi'].mean():+.4f}")

    noisy = out[out["condition"] == "noisy"]
    if len(noisy):
        print(f"Noisy-only ΔSTOI    : {noisy['delta_stoi'].mean():+.4f}")

    # Match the project's existing SNR-bucket convention.
    bins = [-np.inf, 0, 5, 10, 15, 20, np.inf]
    labels = ["<0", "0..5", "5..10", "10..15", "15..20", ">=20"]
    speech = speech.copy()
    speech["snr_bucket"] = pd.cut(
        speech["snr_db"], bins=bins, labels=labels, right=False
    )

    print("\nBy SNR bucket (ΔSTOI):")
    print(
        speech.groupby("snr_bucket", observed=True)["delta_stoi"]
        .agg(["count", "mean"])
        .to_string(float_format=lambda x: f"{x:+.4f}")
    )

    print("\nBy noise class (ΔSTOI):")
    print(
        speech.groupby("noise_class", observed=True)["delta_stoi"]
        .agg(["count", "mean"])
        .to_string(float_format=lambda x: f"{x:+.4f}")
    )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
