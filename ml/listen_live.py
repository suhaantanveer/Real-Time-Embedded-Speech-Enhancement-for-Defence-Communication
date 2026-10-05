"""
listen_live.py
Real-Time Audio Player for Laptop (Windows / Mac / Linux).
Connects to the ESP32-S3 over WiFi and streams the 16 kHz processed audio
directly out of your laptop's speakers or headphones with sub-30ms latency.

Usage:
  python listen_live.py
  python listen_live.py --ip 192.168.4.1
  python listen_live.py --ip 192.168.1.50 --mode clean
"""

import sys
import argparse
import json
import time

try:
    import websocket
except ImportError:
    print("[ERROR] 'websocket-client' is required.")
    print("Install it with: pip install websocket-client")
    sys.exit(1)

try:
    import sounddevice as sd
    import numpy as np
except ImportError:
    print("[ERROR] 'sounddevice' and 'numpy' are required for laptop audio output.")
    print("Install them with: pip install sounddevice numpy")
    sys.exit(1)

SAMPLE_RATE = 16000
CHANNELS = 1

def start_stream(esp_ip="192.168.4.1", mode="clean"):
    ws_url = f"ws://{esp_ip}/ws"
    print("==================================================================")
    print(f"  SIH DEFENCE ANC — REAL-TIME LAPTOP AUDIO MONITOR")
    print(f"  Target: {ws_url} | Channel: {mode.upper()} | Sampling: 16 kHz")
    print("==================================================================")
    print("[INFO] Connecting to ESP32-S3...")

    # Start audio output stream
    audio_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
        blocksize=160,
        latency='low'
    )
    audio_stream.start()

    def on_open(ws):
        print("[CONNECTED] Real-time audio stream active! Audio playing through speakers.")
        print("Press Ctrl+C to stop.\n")
        ws.send(mode)

    def on_message(ws, message):
        if isinstance(message, bytes):
            # Binary 16-bit PCM audio samples
            samples = np.frombuffer(message, dtype=np.int16)

            # sounddevice expects (frames, channels)
            samples = samples.reshape(-1, 1)

            audio_stream.write(samples)
        else:
            # Telemetry text message
            try:
                tel = json.loads(message)
                lat = tel.get('lat', 0)
                att = tel.get('att', 0.0)
                vad = "SPEECH DETECTED (APSA FROZEN)" if tel.get('vad') else "Noise Only (Adapting)"
                print(f"\r[STATUS] Latency: {lat} us | Attenuation: {att:+.1f} dB | VAD: {vad}   ", end="", flush=True)
            except Exception:
                pass

    def on_error(ws, error):
        print(f"\n[WS ERROR] {error}")

    def on_close(ws, close_status_code, close_msg):
        print("\n[DISCONNECTED] Connection closed.")

    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    try:
        ws.run_forever()
    except KeyboardInterrupt:
        print("\n[STOPPING] Audio monitor stopped.")
    finally:
        audio_stream.stop()
        audio_stream.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream real-time audio from ESP32-S3 to laptop speakers")
    parser.add_argument("--ip", type=str, default="192.168.4.1", help="ESP32 IP address (default: 192.168.4.1 for AP mode)")
    parser.add_argument("--mode", type=str, choices=["clean", "primary", "ref"], default="clean", help="Audio channel to monitor")
    args = parser.parse_args()

    start_stream(args.ip, args.mode)
