import argparse
import json
import struct
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import websocket
import wave


SAMPLE_RATE = 16000
FRAME_SAMPLES = 160
BATCH_FRAMES = 4
BATCH_SAMPLES = FRAME_SAMPLES * BATCH_FRAMES  # 640
BYTES_PER_CHANNEL = BATCH_SAMPLES * 2
EXPECTED_PACKET_BYTES = BYTES_PER_CHANNEL * 3  # primary + ref + clean
END_LEAD_SECONDS = 0.25


def write_wav(path: Path, samples: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(x.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.4.1")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--outdir", default="captures")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    primary = []
    ref = []
    clean = []
    packets = 0
    bad = 0
    start = None

    url = f"ws://{args.ip}/ws"

    print("=" * 66)
    print(" SIH26052 SYNCHRONIZED LIVE CAPTURE")
    print("=" * 66)
    print(f"Target   : {url}")
    print(f"Duration : {args.seconds:.1f} s")
    print("Packet   : PRIMARY + REF + CLEAN (640 samples each)")
    print(f"Expected : {EXPECTED_PACKET_BYTES} bytes/packet")
    print("")

    ws = None
    try:
        ws = websocket.create_connection(
            url,
            timeout=1.0,
            enable_multithread=True,
        )
        print("[CONNECTED] Requesting sync_all mode...")
        ws.send("sync_all")
        start = time.monotonic()

        deadline = start + args.seconds

        while time.monotonic() < deadline:
            remaining = max(0.05, min(1.0, deadline - time.monotonic()))
            ws.settimeout(remaining)

            try:
                message = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue

            if isinstance(message, str):
                # Telemetry text; ignore for audio capture.
                continue

            if len(message) != EXPECTED_PACKET_BYTES:
                bad += 1
                print(
                    f"[WARN] Unexpected packet size: {len(message)} "
                    f"bytes (expected {EXPECTED_PACKET_BYTES})"
                )
                continue

            x = np.frombuffer(message, dtype="<i2")
            primary.append(x[0:BATCH_SAMPLES].copy())
            ref.append(x[BATCH_SAMPLES:2 * BATCH_SAMPLES].copy())
            clean.append(x[2 * BATCH_SAMPLES:3 * BATCH_SAMPLES].copy())
            packets += 1

        elapsed = time.monotonic() - start

    except Exception as e:
        elapsed = 0.0 if start is None else time.monotonic() - start
        print(f"[WS ERROR] {e}")

    finally:
        try:
            if ws is not None:
                ws.close()
        except Exception:
            pass

    primary_x = np.concatenate(primary) if primary else np.empty(0, dtype=np.int16)
    ref_x = np.concatenate(ref) if ref else np.empty(0, dtype=np.int16)
    clean_x = np.concatenate(clean) if clean else np.empty(0, dtype=np.int16)

    p_path = outdir / f"sync_{ts}_primary.wav"
    r_path = outdir / f"sync_{ts}_ref.wav"
    c_path = outdir / f"sync_{ts}_clean.wav"
    j_path = outdir / f"sync_{ts}_stats.json"

    write_wav(p_path, primary_x)
    write_wav(r_path, ref_x)
    write_wav(c_path, clean_x)

    stats = {
        "timestamp": ts,
        "requested_seconds": args.seconds,
        "elapsed_seconds": elapsed,
        "packets": packets,
        "bad_packets": bad,
        "samples_per_channel": {
            "primary": int(primary_x.size),
            "ref": int(ref_x.size),
            "clean": int(clean_x.size),
        },
        "expected_samples_for_requested_duration": int(round(args.seconds * SAMPLE_RATE)),
        "expected_packet_size_bytes": EXPECTED_PACKET_BYTES,
        "packet_payload_samples_per_channel": BATCH_SAMPLES,
        "sample_rate": SAMPLE_RATE,
    }
    j_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("")
    print("[RESULT]")
    print(f"Elapsed  : {elapsed:.2f} s")
    print(f"Packets  : {packets}")
    print(f"Bad pkts : {bad}")
    print(f"Samples  : {primary_x.size} / channel")
    print(f"Duration : {primary_x.size / SAMPLE_RATE:.3f} s / channel")
    print("")
    print(f"[SAVED] {p_path}")
    print(f"[SAVED] {r_path}")
    print(f"[SAVED] {c_path}")
    print(f"[SAVED] {j_path}")


if __name__ == "__main__":
    main()
