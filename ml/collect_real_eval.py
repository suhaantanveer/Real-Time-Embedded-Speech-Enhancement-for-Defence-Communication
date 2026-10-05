import argparse
import json
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import websocket


FS = 16000

# One synchronized packet contains:
#   640 primary samples
#   640 reference samples
#   640 enhanced samples
# each as int16
PACKET_SAMPLES = 640
PACKET_BYTES = PACKET_SAMPLES * 3 * 2


def save_wav(path, samples):
    samples = np.asarray(samples, dtype=np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(samples.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="SIH26052 synchronized real-hardware ANC capture"
    )

    parser.add_argument(
        "--ip",
        default="192.168.4.1",
        help="ESP32 IP address"
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=20.0,
        help="Capture duration"
    )

    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario name, e.g. quiet_speech"
    )

    parser.add_argument(
        "--mode",
        choices=["primary", "dsp", "full"],
        default="full"
    )

    parser.add_argument(
        "--mic2",
        choices=["on", "off"],
        default="on"
    )

    parser.add_argument(
        "--take",
        default="01"
    )

    parser.add_argument(
        "--output",
        default="data/real_eval"
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"{timestamp}_"
        f"{args.scenario}_"
        f"{args.mode}_"
        f"mic2-{args.mic2}_"
        f"take-{args.take}"
    )

    run_dir = Path(args.output) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 70)
    print(" SIH26052 REAL HARDWARE CAPTURE")
    print("=" * 70)

    print(f"IP       : {args.ip}")
    print(f"Duration : {args.seconds:.1f} s")
    print(f"Scenario : {args.scenario}")
    print(f"Mode     : {args.mode}")
    print(f"Mic2     : {args.mic2}")
    print(f"Take     : {args.take}")
    print()

    metadata = {
        "timestamp": timestamp,
        "sample_rate": FS,
        "requested_seconds": args.seconds,
        "scenario": args.scenario,
        "mode": args.mode,
        "mic2": args.mic2,
        "take": args.take,
        "ip": args.ip,
        "packet_bytes": PACKET_BYTES,
    }

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    ws_url = f"ws://{args.ip}/ws"

    print(f"[CONNECTING] {ws_url}")

    ws = websocket.create_connection(
        ws_url,
        timeout=5
    )

    ws.settimeout(1.0)

    # Tell firmware to send:
    # [PRIMARY][REF][CLEAN]
    ws.send("sync_all")

    print("[CONNECTED] sync_all requested")
    print("[RECORDING]")

    primary_chunks = []
    ref_chunks = []
    clean_chunks = []

    telemetry = []

    start = time.monotonic()
    deadline = start + args.seconds

    packet_count = 0
    bad_packets = 0

    try:

        while time.monotonic() < deadline:

            try:
                payload = ws.recv()

            except Exception as exc:
                message = str(exc).lower()

                if "timeout" in message:
                    continue

                print(f"[WS ERROR] {exc}")
                break

            if payload is None:
                break

            # -------------------------------------------------
            # TELEMETRY
            # -------------------------------------------------

            if isinstance(payload, str):

                try:
                    obj = json.loads(payload)

                    if isinstance(obj, dict):
                        telemetry.append(obj)

                except Exception:
                    pass

                continue

            # -------------------------------------------------
            # AUDIO
            # -------------------------------------------------

            payload = bytes(payload)

            if len(payload) != PACKET_BYTES:

                bad_packets += 1

                print(
                    f"[BAD PACKET] "
                    f"{len(payload)} bytes "
                    f"(expected {PACKET_BYTES})"
                )

                continue

            samples = np.frombuffer(
                payload,
                dtype="<i2"
            )

            primary = samples[0:640]
            ref = samples[640:1280]
            clean = samples[1280:1920]

            primary_chunks.append(primary.copy())
            ref_chunks.append(ref.copy())
            clean_chunks.append(clean.copy())

            packet_count += 1

    finally:

        ws.close()

    elapsed = time.monotonic() - start

    # ---------------------------------------------------------
    # COMBINE AUDIO
    # ---------------------------------------------------------

    if primary_chunks:

        primary_audio = np.concatenate(primary_chunks)

        ref_audio = np.concatenate(ref_chunks)

        clean_audio = np.concatenate(clean_chunks)

    else:

        primary_audio = np.empty(0, dtype=np.int16)
        ref_audio = np.empty(0, dtype=np.int16)
        clean_audio = np.empty(0, dtype=np.int16)

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    save_wav(
        run_dir / "primary.wav",
        primary_audio
    )

    save_wav(
        run_dir / "ref.wav",
        ref_audio
    )

    save_wav(
        run_dir / "clean.wav",
        clean_audio
    )

    report = {
        "elapsed_seconds": elapsed,
        "requested_seconds": args.seconds,
        "packets_received": packet_count,
        "bad_packets": bad_packets,
        "audio_samples": int(len(primary_audio)),
        "audio_duration_seconds": float(
            len(primary_audio) / FS
        ),
        "expected_packets": int(
            round(args.seconds * FS / PACKET_SAMPLES)
        ),
        "telemetry_messages": len(telemetry),
    }

    (run_dir / "capture_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    (run_dir / "telemetry.json").write_text(
        json.dumps(telemetry, indent=2),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print(" CAPTURE COMPLETE")
    print("=" * 70)

    print(f"Elapsed        : {elapsed:.2f} s")
    print(f"Packets        : {packet_count}")
    print(f"Bad packets    : {bad_packets}")
    print(f"Audio duration : {len(primary_audio) / FS:.3f} s")
    print(f"Telemetry      : {len(telemetry)} messages")
    print()
    print(f"Saved to       : {run_dir}")
    print()
    print("Files:")
    print("  primary.wav")
    print("  ref.wav")
    print("  clean.wav")
    print("  telemetry.json")
    print("  capture_report.json")
    print("  metadata.json")


if __name__ == "__main__":
    main()