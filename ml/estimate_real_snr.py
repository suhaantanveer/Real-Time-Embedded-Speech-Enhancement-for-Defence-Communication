
import json
import wave
from pathlib import Path

import numpy as np

FS = 16000
FRAME = 160  # 10 ms


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        x = np.frombuffer(
            wf.readframes(wf.getnframes()), dtype="<i2"
        ).astype(np.float64) / 32768.0
        fs = wf.getframerate()
    if fs != FS:
        raise ValueError(f"{path}: expected 16 kHz, got {fs}")
    return x


def frame_rms(x):
    n = len(x) // FRAME
    x = x[:n * FRAME].reshape(n, FRAME)
    return np.sqrt(np.mean(x * x, axis=1) + 1e-12)


def robust_snr_from_recording(primary):
    """
    Estimate SNR from the recording itself:
      - classify relatively quiet frames as noise-only
      - classify louder frames as speech+noise
      - estimate noise power from quiet frames
      - subtract noise power from speech+noise power

    This is an ESTIMATE, not ground-truth SNR.
    It is most defensible for stationary noise.
    """
    rms = frame_rms(primary)
    db = 20 * np.log10(np.maximum(rms, 1e-12))

    # Keep the lower 35% as a conservative noise-floor pool.
    noise_pool = db <= np.percentile(db, 35)

    # Speech/noise pool: upper 50%, excluding quietest frames.
    active_pool = db >= np.percentile(db, 50)

    if np.sum(noise_pool) < 10 or np.sum(active_pool) < 10:
        return None

    p_noise = float(np.median(rms[noise_pool] ** 2))
    p_mix = float(np.median(rms[active_pool] ** 2))

    p_speech = max(p_mix - p_noise, 1e-12)

    snr_db = 10 * np.log10(p_speech / max(p_noise, 1e-12))

    return {
        "noise_floor_rms_dbfs": float(
            20 * np.log10(np.sqrt(p_noise))
        ),
        "speech_plus_noise_rms_dbfs": float(
            20 * np.log10(np.sqrt(p_mix))
        ),
        "estimated_speech_rms_dbfs": float(
            20 * np.log10(np.sqrt(p_speech))
        ),
        "estimated_snr_db": float(snr_db),
        "noise_frames": int(np.sum(noise_pool)),
        "active_frames": int(np.sum(active_pool)),
    }


def main():
    root = Path("data/real_eval")

    if not root.exists():
        print(f"ERROR: {root} not found")
        return

    print("=" * 78)
    print("ESTIMATED REAL-HARDWARE INPUT SNR")
    print("=" * 78)
    print()
    print("IMPORTANT: these are estimates from the recording itself.")
    print("They are NOT ground-truth SNR measurements.")
    print("They are most useful for stationary-noise recordings.")
    print()

    for run_dir in sorted(root.iterdir()):
        primary_path = run_dir / "primary.wav"
        metadata_path = run_dir / "metadata.json"

        if not primary_path.exists() or not metadata_path.exists():
            continue

        meta = json.loads(metadata_path.read_text(encoding="utf-8"))

        scenario = meta.get("scenario", "")

        # Skip noise-only recordings: they have no speech SNR.
        if "speech" not in scenario.lower():
            continue

        try:
            primary = read_wav(primary_path)
            result = robust_snr_from_recording(primary)

            if result is None:
                print(f"[SKIP] {run_dir.name}: insufficient variation")
                continue

            print(f"{scenario:<30}")
            print(
                f"  noise floor          : "
                f"{result['noise_floor_rms_dbfs']:.2f} dBFS"
            )
            print(
                f"  speech+noise level   : "
                f"{result['speech_plus_noise_rms_dbfs']:.2f} dBFS"
            )
            print(
                f"  estimated speech    : "
                f"{result['estimated_speech_rms_dbfs']:.2f} dBFS"
            )
            print(
                f"  ESTIMATED SNR        : "
                f"{result['estimated_snr_db']:.2f} dB"
            )
            print()

        except Exception as e:
            print(f"[ERROR] {run_dir.name}: {e}")


if __name__ == "__main__":
    main()
