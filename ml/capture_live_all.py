"""
capture_live_all.py
Capture synchronized PRIMARY + REFERENCE + CLEAN audio from the ESP32.

Protocol:
    The ESP32 must support the control command "all". For each 40 ms WebSocket
    binary frame it sends, the payload is exactly:

        PRIMARY[640] + REFERENCE[640] + CLEAN[640]

    Each sample is little-endian signed int16 PCM at 16 kHz.

Usage:
    python ml/capture_live_all.py --ip 192.168.4.1 --seconds 15

Outputs:
    captures/sync_<timestamp>_primary.wav
    captures/sync_<timestamp>_ref.wav
    captures/sync_<timestamp>_clean.wav
    captures/sync_<timestamp>_stats.json
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import wave
from pathlib import Path

try:
    import numpy as np
    import websocket
except ImportError as exc:
    raise SystemExit(
        "Install dependencies first: pip install numpy websocket-client"
    ) from exc


SAMPLE_RATE = 16000
BATCH_SAMPLES = 640  # 40 ms at 16 kHz
BYTES_PER_SAMPLE = 2
CHANNELS = 1
PAYLOAD_BYTES = 3 * BATCH_SAMPLES * BYTES_PER_SAMPLE  # 3840


def save_wav(path: Path, pcm: bytearray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture synchronized primary/ref/clean ESP32 audio."
    )
    parser.add_argument("--ip", default="192.168.4.1")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=Path("captures"))
    args = parser.parse_args()

    if args.seconds <= 0:
        raise SystemExit("--seconds must be > 0")

    ws_url = f"ws://{args.ip}/ws"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = args.output_dir / f"sync_{stamp}"

    primary_pcm = bytearray()
    ref_pcm = bytearray()
    clean_pcm = bytearray()

    state = {
        "start": None,
        "packets": 0,
        "bad_packets": 0,
        "bytes": 0,
        "stopped_by_timeout": False,
        "error": None,
    }

    print("=" * 70)
    print(" SIH26052 SYNCHRONIZED LIVE CAPTURE")
    print("=" * 70)
    print(f"Target   : {ws_url}")
    print(f"Duration : {args.seconds:.1f} s")
    print(f"Payload  : {PAYLOAD_BYTES} bytes/packet")
    print("Order    : PRIMARY[640] + REF[640] + CLEAN[640]")
    print("Output   : three synchronized WAV files")
    print()

    ws = None

    def on_open(sock):
        state["start"] = time.monotonic()
        sock.send("all")
        print("[CONNECTED] Requested synchronized mode: all")

    def on_message(sock, message):
        now = time.monotonic()

        if isinstance(message, bytes):
            if len(message) != PAYLOAD_BYTES:
                state["bad_packets"] += 1
                print(
                    f"\n[WARN] Unexpected packet size: {len(message)} "
                    f"bytes (expected {PAYLOAD_BYTES})"
                )
            else:
                p0 = 0
                p1 = BATCH_SAMPLES * BYTES_PER_SAMPLE
                p2 = 2 * BATCH_SAMPLES * BYTES_PER_SAMPLE
                p3 = 3 * BATCH_SAMPLES * BYTES_PER_SAMPLE

                primary_pcm.extend(message[p0:p1])
                ref_pcm.extend(message[p1:p2])
                clean_pcm.extend(message[p2:p3])
                state["packets"] += 1
                state["bytes"] += len(message)

                elapsed = now - state["start"] if state["start"] else 0.0
                print(
                    f"\r[CAPTURE] {elapsed:6.1f}/{args.seconds:5.1f}s | "
                    f"packets={state['packets']:4d} | "
                    f"samples={state['packets'] * BATCH_SAMPLES:7d}",
                    end="",
                    flush=True,
                )

            if state["start"] is not None and now - state["start"] >= args.seconds:
                state["stopped_by_timeout"] = True
                sock.close()
        else:
            # Ignore telemetry text frames for the capture itself.
            pass

    def on_error(sock, error):
        state["error"] = str(error)
        print(f"\n[WS ERROR] {error}")

    def close_after_duration():
        time.sleep(args.seconds + 0.25)
        if state["start"] is not None and ws is not None:
            if ws.sock is not None and ws.sock.connected:
                state["stopped_by_timeout"] = True
                try:
                    ws.close()
                except Exception:
                    pass

    def on_close(sock, close_status_code, close_msg):
        elapsed = (
            time.monotonic() - state["start"]
            if state["start"] is not None
            else 0.0
        )
        print("\n[DISCONNECTED]")
        print(f"Elapsed time : {elapsed:.2f} s")
        print(f"Packets      : {state['packets']}")
        print(f"Bad packets  : {state['bad_packets']}")
        print(f"Samples/ch   : {state['packets'] * BATCH_SAMPLES}")

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    timer = threading.Thread(target=close_after_duration, daemon=True)
    timer.start()

    try:
        ws.run_forever()
    except KeyboardInterrupt:
        print("\n[STOP] Capture interrupted by user.")
        try:
            ws.close()
        except Exception:
            pass

    primary_path = Path(f"{prefix}_primary.wav")
    ref_path = Path(f"{prefix}_ref.wav")
    clean_path = Path(f"{prefix}_clean.wav")
    stats_path = Path(f"{prefix}_stats.json")

    save_wav(primary_path, primary_pcm)
    save_wav(ref_path, ref_pcm)
    save_wav(clean_path, clean_pcm)

    samples_per_channel = state["packets"] * BATCH_SAMPLES
    expected_packets = int(args.seconds / 0.04)

    stats = {
        "target": ws_url,
        "requested_seconds": args.seconds,
        "expected_packets_40ms": expected_packets,
        "packets": state["packets"],
        "bad_packets": state["bad_packets"],
        "samples_per_channel": samples_per_channel,
        "duration_per_channel_seconds": samples_per_channel / SAMPLE_RATE,
        "payload_bytes_per_packet": PAYLOAD_BYTES,
        "order": "primary + ref + clean",
        "error": state["error"],
        "stopped_by_timeout": state["stopped_by_timeout"],
        "files": {
            "primary": str(primary_path),
            "ref": str(ref_path),
            "clean": str(clean_path),
        },
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print()
    print(f"[SAVED] {primary_path}")
    print(f"[SAVED] {ref_path}")
    print(f"[SAVED] {clean_path}")
    print(f"[SAVED] {stats_path}")
    print()
    print(
        f"Captured {samples_per_channel} synchronized samples/channel "
        f"({samples_per_channel / SAMPLE_RATE:.2f} s)."
    )


if __name__ == "__main__":
    main()
