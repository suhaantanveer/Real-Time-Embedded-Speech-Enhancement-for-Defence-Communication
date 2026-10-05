# SIH26052 ESP32-S3 Firmware — V2

Target: ESP32-S3 DevKitC-1 N16R8, 16 kHz dual digital microphones.

## Default signal path

```text
Mic 1 -> front-end -> 256 FFT / 22 bands -> 89-input GRU -> gains -> ISTFT -> dry/wet safeguard
Mic 2 ------------------------------------^ side information only (optional)
```

Mic 2 is not adaptively subtracted from Mic 1 in the default path. This prevents direct target-speech cancellation from reference-mic leakage.

Legacy APSA/FLANN code remains in the project behind `ENABLE_ADAPTIVE_PREFILTER` for A/B research. It is `0` by default.

## Wiring

Both INMP441 microphones share BCLK and WS. Mic 1 uses Left (`L/R=GND`), Mic 2 uses Right (`L/R=3V3`). Current baseline pins remain GPIO 4/5/6 for BCLK/WS/DIN.

## Serial A/B modes

- `raw`
- `front`
- `ml1` — Mic 1 only, default
- `dual` — Mic 1 + Mic 2 side information
- `apsa`, `dsp`, `legacy` — legacy comparison path when enabled
- `reset`, `help`

Telemetry reports average DSP latency, reference validity, input level/peak, output level, and tracked noise floor.

## Important

The supplied handoff did not contain hardware level measurements or a trained V2 checkpoint. `model_weights.h` is currently exported from a V2 bootstrap checkpoint created by copying the legacy shared weights into the new architecture. It is for build/numerical-equivalence smoke testing only; train a real V2 model before judging audio quality.
