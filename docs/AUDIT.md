# SIH26052 ANC Redesign — Phase 0 Audit

Date: 2026-10-01

## Scope

Audited the supplied `firmware/` and `ml/` handoff before changing behavior. The handoff is a PlatformIO ESP32-S3 N16R8 firmware project plus a PyTorch speech-enhancement project.

## Existing architecture (verified)

### Firmware

`firmware/src/main.cpp` currently runs:

`mic1, mic2 -> APSA (64 taps, P=2) -> FLANN (16 taps, Chebyshev order 3) -> 256-pt causal STFT -> 22 Bark/log-energy + 22 deltas -> GRU -> 22 gains -> causal ISTFT -> output`

The active path uses mic 2 as the adaptive-filter reference. `MONITOR_RAW`, `MONITOR_APSA`, `MONITOR_DSP`, and `MONITOR_FULL` allow live A/B monitoring.

`firmware/src/i2s_audio.c` applies a software preamp gain of `MIC_PREAMP_GAIN` followed by `tanhf()` scaling before the rest of the pipeline. The current configured value is 16x.

`firmware/src/apsa_filter.c` contains an adaptation-freeze flag and supports freezing the adaptive weights.

`firmware/src/flann_filter.c` has no adaptation-freeze flag; it adapts on every sample.

`firmware/src/features_esp32.c` implements a 256-point radix-2 FFT, 22-band Bark filterbank, and causal overlap-add synthesis. The current feature interface is 44 inputs.

`firmware/src/gru_inference.c` implements the exported PyTorch GRU in plain C, including the 64-unit + 48-unit recurrent layers and LayerNorm.

### ML

`ml/model.py` is a 45,462-parameter GRU model with 44 input features and 22 output gains.

`ml/dataset_loader.py` currently loads a manifest containing `clean`, `primary_mic`, and `ref_mic`, runs the Python APSA+FLANN simulator, and trains the GRU on that residual. This means the training path is coupled to the adaptive-filter residual rather than to mic 1 directly.

`ml/dsp_simulator.py` mirrors the current APSA and FLANN code. APSA has frame-level freeze logic; FLANN does not freeze in the firmware, so the simulator and firmware behavior are not fully identical.

`ml/losses.py` currently uses SI-SNR + multi-resolution STFT loss, with an optional mask loss that is not used by `train.py`.

`ml/train.py` trains on the DSP residual and selects the best checkpoint by validation SI-SNR.

No dataset-mixing script was present in the handoff. No source-data manifest was present either.

## Existing issues that motivate the redesign

1. The user-reported listening result says raw mic 1 sounds better than APSA, DSP, and FULL in low noise.
2. Mic 2 contains speech leakage. Using mic 2 as an adaptive reference can therefore cancel desired speech.
3. The firmware has APSA adaptation freeze but no FLANN freeze flag, while the Python simulator applies freeze logic to both stages. This creates a simulation/firmware mismatch.
4. The current speech detector uses `p_primary / p_reference > 1.8` (~2.55 dB) without hangover logic.
5. The supplied handoff reports that the exported GRU often predicts very low gains. This is consistent with the current loss lacking an explicit absolute-level preservation term.
6. The current dataset loader depends on an unavailable mixing script and therefore the exact training data distribution could not be audited.
7. The configured 16x preamp + `tanh` is not supported by an actual current hardware level measurement in this handoff.
8. The FLANN Chebyshev T2/T3 basis can be poorly conditioned near zero because T2 approaches -1 for small inputs; this remains an optional/secondary path in the redesign.

## Baseline checks run in this environment

- PyTorch model import and forward pass: **passed**.
- Current checkpoint loaded: epoch 11, validation SI-SNR recorded in checkpoint: **3.7905600468 dB**.
- Current model parameter count: **45,462**.
- PlatformIO CLI is **not installed in this environment**, so the ESP32 firmware could not be rebuilt here.
- The supplied `firmware/test_numerical_equivalence.exe` is a Windows executable and cannot be executed in this Linux environment (`Exec format error`).
- No current clean-speech source dataset was present, so a real hardware/dataset level probe could not be run yet. A synthetic-vowel regression probe is provided separately as a software-only diagnostic.

## Dataset state

The handoff contains no `mix_dataset.py`-style source and no `dataset/` directory. Therefore Phase 2 creates a deterministic dataset builder from local speech/noise/RIR directories instead of inventing a dataset that was not supplied.

## Redesign decision

Default path:

`Mic 1 -> DC block + HPF + impulse guard -> STFT/features`

`Mic 2 -> DC block + HPF -> STFT/features (optional)`

`Mic 1 + Mic 2 features -> small GRU -> band gains -> ISTFT -> dry/wet safeguard -> output`

Mic 2 is side information only and is never directly subtracted from mic 1 in the default path. APSA/FLANN remain available behind `ENABLE_ADAPTIVE_PREFILTER=0` for controlled comparison.

## What cannot be honestly claimed yet

- Real-time ESP32-S3 latency for the redesigned pipeline.
- Achieved SI-SDR/STOI improvements on the intended training/test corpus.
- Hardware-level microphone calibration and actual input RMS distribution.
- That mic 2 gives a specific quantitative benefit.

Those are measurement targets for later phases, not current results.
