
import csv
import json
import wave
from pathlib import Path

import numpy as np

FS = 16000
FULL_SCALE = 32767 / 32768


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        fs = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if channels != 1 or width != 2:
        raise ValueError(f"{path}: expected mono 16-bit WAV")

    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return x, fs


def rms_dbfs(x):
    if len(x) == 0:
        return float("nan")
    return float(20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-12)))


def peak_dbfs(x):
    if len(x) == 0:
        return float("nan")
    return float(20 * np.log10(max(np.max(np.abs(x)), 1e-12)))


def stats(path):
    x, fs = read_wav(path)
    if fs != FS:
        raise ValueError(f"{path}: expected 16 kHz, got {fs}")

    return {
        "duration_s": len(x) / FS,
        "rms_dbfs": rms_dbfs(x),
        "peak_dbfs": peak_dbfs(x),
        "full_scale_pct": float(100 * np.mean(np.abs(x) >= FULL_SCALE)),
    }


def load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def nums(rows, key):
    return np.array(
        [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))],
        dtype=np.float64,
    )


def analyze_run(run_dir):
    meta = load_json(run_dir / "metadata.json", {})
    report = load_json(run_dir / "capture_report.json", {})
    telemetry = load_json(run_dir / "telemetry.json", [])

    p = stats(run_dir / "primary.wav")
    r = stats(run_dir / "ref.wav")
    c = stats(run_dir / "clean.wav")

    lat = nums(telemetry, "lat")
    att = nums(telemetry, "att")
    pin = nums(telemetry, "pin")
    pref = nums(telemetry, "pref")

    row = {
        "run": run_dir.name,
        "scenario": meta.get("scenario", ""),
        "mode": meta.get("mode", ""),
        "mic2": meta.get("mic2", ""),
        "take": meta.get("take", ""),
        "requested_s": meta.get("requested_seconds", ""),
        "elapsed_s": report.get("elapsed_seconds", ""),
        "audio_s": report.get("audio_duration_seconds", ""),
        "packets": report.get("packets_received", ""),
        "expected_packets": report.get("expected_packets", ""),
        "bad_packets": report.get("bad_packets", ""),
        "packet_loss_pct": (
            100 * report.get("missing_packets_estimate", 0)
            / report["expected_packets"]
            if report.get("expected_packets") else ""
        ),
        "primary_rms_dbfs": p["rms_dbfs"],
        "primary_peak_dbfs": p["peak_dbfs"],
        "primary_full_scale_pct": p["full_scale_pct"],
        "ref_rms_dbfs": r["rms_dbfs"],
        "ref_peak_dbfs": r["peak_dbfs"],
        "enhanced_rms_dbfs": c["rms_dbfs"],
        "enhanced_peak_dbfs": c["peak_dbfs"],
        "enhanced_full_scale_pct": c["full_scale_pct"],
        "enhanced_minus_primary_rms_db": c["rms_dbfs"] - p["rms_dbfs"],
        "telemetry_count": len(telemetry),
        "latency_mean_ms": float(np.mean(lat) / 1000) if len(lat) else "",
        "latency_median_ms": float(np.median(lat) / 1000) if len(lat) else "",
        "latency_p95_ms": float(np.percentile(lat, 95) / 1000) if len(lat) else "",
        "latency_max_ms": float(np.max(lat) / 1000) if len(lat) else "",
        "attenuation_mean_db": float(np.mean(att)) if len(att) else "",
        "attenuation_median_db": float(np.median(att)) if len(att) else "",
        "attenuation_p95_db": float(np.percentile(att, 95)) if len(att) else "",
        "mic1_level_mean": float(np.mean(pin)) if len(pin) else "",
        "mic2_level_mean": float(np.mean(pref)) if len(pref) else "",
    }

    if telemetry:
        first = telemetry[0]
        last = telemetry[-1]

        for key in ["qdrop", "qsend", "qmax", "frames", "batches", "maxgap"]:
            row[f"{key}_first"] = first.get(key, "")
            row[f"{key}_last"] = last.get(key, "")

        for key in ["qdrop", "qsend", "frames", "batches"]:
            a, b = first.get(key), last.get(key)
            row[f"{key}_delta"] = (
                b - a
                if isinstance(a, (int, float)) and isinstance(b, (int, float))
                else ""
            )

        row["maxgap_last_ms"] = (
            last["maxgap"] / 1000
            if isinstance(last.get("maxgap"), (int, float))
            else ""
        )

        vad_values = [bool(x["vad"]) for x in telemetry if "vad" in x]
        row["vad_speech_fraction"] = (
            float(np.mean(vad_values)) if vad_values else ""
        )

    return row


def main():
    root = Path("data/real_eval")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    if not root.exists():
        print(f"ERROR: {root} not found")
        return

    rows = []

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        required = [
            run_dir / "metadata.json",
            run_dir / "capture_report.json",
            run_dir / "primary.wav",
            run_dir / "ref.wav",
            run_dir / "clean.wav",
        ]

        if not all(p.exists() for p in required):
            print(f"[SKIP] {run_dir.name}")
            continue

        try:
            rows.append(analyze_run(run_dir))
            print(f"[OK]   {run_dir.name}")
        except Exception as exc:
            print(f"[ERROR] {run_dir.name}: {exc}")

    if not rows:
        print("No complete runs found.")
        return

    fields = sorted({k for row in rows for k in row})

    out = output_dir / "real_hardware_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("REAL HARDWARE DATA SUMMARY")
    print("=" * 70)
    print(f"Runs analyzed : {len(rows)}")
    print(f"Saved         : {out}")
    print()

    for row in rows:
        print(
            f"{row['scenario']:<30} "
            f"RMS Δ={row['enhanced_minus_primary_rms_db']:>7.2f} dB  "
            f"lat={row['latency_mean_ms']} ms  "
            f"packets={row['packets']}"
        )

    print()
    print("RMS Δ = enhanced RMS minus primary RMS.")
    print("This is level change, NOT SNR improvement.")


if __name__ == "__main__":
    main()
