"""
capture_live.py
Capture the ESP32-S3 WebSocket audio stream directly to a WAV file.

Usage:
    python capture_live.py --ip 192.168.4.1 --mode primary --seconds 30
    python capture_live.py --ip 192.168.4.1 --mode clean   --seconds 30
    python capture_live.py --ip 192.168.4.1 --mode ref     --seconds 30

The ESP32 sends 16 kHz, mono, signed 16-bit PCM in binary WebSocket frames.
This script does NOT play the audio live; it saves exactly what arrived over
the WebSocket so we can diagnose transport/playback dropouts separately.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
import threading
from pathlib import Path

try:
    import websocket
except ImportError:
    raise SystemExit("Install websocket-client first: pip install websocket-client")


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # int16
VALID_MODES = ("clean", "primary", "ref")


def capture_stream(
    esp_ip: str,
    mode: str,
    seconds: float,
    output: Path,
) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")

    ws_url = f"ws://{esp_ip}/ws"
    output.parent.mkdir(parents=True, exist_ok=True)

    wav = wave.open(str(output), "wb")
    wav.setnchannels(CHANNELS)
    wav.setsampwidth(SAMPLE_WIDTH_BYTES)
    wav.setframerate(SAMPLE_RATE)

    state = {
        "start": None,
        "bytes": 0,
        "frames": 0,
        "binary_messages": 0,
        "last_status": 0.0,
    }

    print("=" * 66)
    print(" SIH26052 LIVE CAPTURE")
    print("=" * 66)
    print(f"Target   : {ws_url}")
    print(f"Mode     : {mode}")
    print(f"Duration : {seconds:.1f} s")
    print(f"Output   : {output}")
    print("Saving raw WebSocket PCM; no live playback.")
    print()

    def on_open(ws):
        state["start"] = time.monotonic()
        ws.send(mode)
        print(f"[CONNECTED] Requested mode: {mode}")

    def on_message(ws, message):
        now = time.monotonic()

        if isinstance(message, bytes):
            # ESP32 sends little-endian signed 16-bit PCM.
            if len(message) % 2 != 0:
                print(f"[WARN] Odd-length binary frame: {len(message)} bytes")
                return

            wav.writeframes(message)
            state["bytes"] += len(message)
            state["frames"] += len(message) // 2
            state["binary_messages"] += 1

            if now - state["last_status"] >= 1.0:
                state["last_status"] = now
                elapsed = now - state["start"] if state["start"] else 0.0
                print(
                    f"\r[CAPTURE] {elapsed:6.1f}/{seconds:5.1f}s | "
                    f"{state['binary_messages']:5d} packets | "
                    f"{state['frames']:7d} samples",
                    end="",
                    flush=True,
                )

        else:
            try:
                tel = json.loads(message)
                lat = tel.get("lat", 0)
                att = tel.get("att", 0.0)
                vad = tel.get("vad", False)
                # Print telemetry occasionally without flooding the terminal.
                if now - state["last_status"] >= 1.0:
                    print(
                        f"\r[STATUS] latency={lat}us | "
                        f"atten={att:+.1f}dB | "
                        f"VAD={'speech' if vad else 'noise':5s}      ",
                        end="",
                        flush=True,
                    )
            except (json.JSONDecodeError, TypeError):
                pass

        if state["start"] is not None and now - state["start"] >= seconds:
            ws.close()

    def on_error(ws, error):
        print(f"\n[WS ERROR] {error}")

    def close_after_duration():
        # on_message() may never run when the ESP32 sends no packets.
        # Use an independent timer so --seconds is a hard wall-clock limit.
        time.sleep(seconds)
        if state["start"] is not None:
            try:
                ws.close()
            except Exception:
                pass

    def on_close(ws, close_status_code, close_msg):
        elapsed = (
            time.monotonic() - state["start"]
            if state["start"] is not None
            else 0.0
        )
        print("\n[DISCONNECTED]")
        print(f"Elapsed time : {elapsed:.2f} s")
        print(f"Packets      : {state['binary_messages']}")
        print(f"Samples      : {state['frames']}")
        print(f"Bytes        : {state['bytes']}")

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
    finally:
        wav.close()

    print(f"\n[SAVED] {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture ESP32-S3 WebSocket PCM stream to WAV."
    )
    parser.add_argument(
        "--ip",
        default="192.168.4.1",
        help="ESP32 IP address (default: 192.168.4.1)",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="clean",
        help="Audio source to capture",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="Capture duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV path. Default: captures/<mode>_<timestamp>.wav",
    )
    args = parser.parse_args()

    if args.seconds <= 0:
        raise SystemExit("--seconds must be > 0")

    if args.output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = Path("captures") / f"{args.mode}_{stamp}.wav"

    capture_stream(
        esp_ip=args.ip,
        mode=args.mode,
        seconds=args.seconds,
        output=args.output,
    )


if __name__ == "__main__":
    main()
