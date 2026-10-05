
from pathlib import Path
import wave
import numpy as np

RUN = Path(r"data\real_eval\20261004_141813_quiet_speech_full_mic2-on_take-01")
FS = 16000
FRAME = 160  # 10 ms


def read(path):
    with wave.open(str(path), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
        fs = w.getframerate()
    assert fs == FS
    return x


primary = read(RUN / "primary.wav")
clean = read(RUN / "clean.wav")

n = min(len(primary), len(clean))
primary = primary[:n]
clean = clean[:n]

nframes = n // FRAME
primary = primary[:nframes * FRAME].reshape(nframes, FRAME)
clean = clean[:nframes * FRAME].reshape(nframes, FRAME)

prms = np.sqrt(np.mean(primary ** 2, axis=1) + 1e-12)
crms = np.sqrt(np.mean(clean ** 2, axis=1) + 1e-12)

# Detect speech-active frames using primary level.
# Threshold is relative to the loudest part of the recording.
p_db = 20 * np.log10(np.maximum(prms, 1e-12))
threshold = np.percentile(p_db, 40)
active = p_db > threshold

ratio_db = 20 * np.log10(np.maximum(crms, 1e-12) / np.maximum(prms, 1e-12))

print("=" * 65)
print("QUIET SPEECH: SPEECH-ACTIVE LEVEL CHECK")
print("=" * 65)
print(f"Total frames          : {nframes}")
print(f"Active frames         : {int(np.sum(active))}")
print(f"Activity threshold    : {threshold:.2f} dBFS")
print()

for name, arr in [
    ("PRIMARY", prms[active]),
    ("ENHANCED", crms[active]),
    ("LEVEL CHANGE", ratio_db[active]),
]:
    print(f"{name}:")
    print(f"  mean : {np.mean(arr):.4f}" if name != "LEVEL CHANGE" else f"  mean : {np.mean(arr):.2f} dB")
    print(f"  p10  : {np.percentile(arr, 10):.4f}" if name != "LEVEL CHANGE" else f"  p10  : {np.percentile(arr, 10):.2f} dB")
    print(f"  p50  : {np.percentile(arr, 50):.4f}" if name != "LEVEL CHANGE" else f"  p50  : {np.percentile(arr, 50):.2f} dB")
    print(f"  p90  : {np.percentile(arr, 90):.4f}" if name != "LEVEL CHANGE" else f"  p90  : {np.percentile(arr, 90):.2f} dB")
    print()

print("Interpretation:")
print("The earlier -8.66 dB was whole-record RMS.")
print("This test isolates speech-active frames, so pauses/noise-floor suppression")
print("cannot make the speech result look worse than it really is.")
