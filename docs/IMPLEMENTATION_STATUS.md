# SIH26052 ANC Redesign — Implementation Status

## Completed in this handoff

- Mic-1-primary + optional Mic-2 side-information architecture.
- Matching Python/ESP32 DC blocker + 100 Hz HPF + mic-1 impulse guard.
- 89-feature dual-microphone input and 48k-parameter GRU enhancer.
- Deterministic speech/noise/RIR dataset preparation and manifest generation.
- Speaker-disjoint speech pools; current dataset generation now also partitions noise and RIR source files disjointly by train/validation/test.
- Dataset mixer uses speech-active cropping, active-speech SNR scaling, Mic-2 noise tied to the Mic-1 noise component, variable speech leakage, and explicit measured SNR metadata.
- Training loop uses noise-only-safe SI-SNR, clean-speech-only preservation metrics, fixed stratified validation, Mic-2 A/B validation, and clean-preservation-aware checkpoint selection.
- Numerical-equivalence tests and host-side front-end/model checks remain part of the pipeline.

## Important corrections made before final training

- Impulse guard now compares against the previous frame EMA and uses a separate 0.90 peak threshold; the clean target uses only the linear DC/HPF path.
- Mic-2 environmental noise is tied to the Mic-1 noise level instead of receiving an unrelated raw-corpus level.
- Noise and RIR source files are held out across train/val/test.
- Faulty-reference examples keep `ref_valid=1` and use unrelated noise rather than being indistinguishable from normal dropout.
- Validation excludes noise-only samples from SI-SNR, evaluates clean gain on speech-active clean-condition frames, and uses a fixed stratified subset of at least 500 examples.

## Not yet claimed

- No full V3 training result is established yet.
- No SI-SDR/STOI performance result is established yet.
- No real microphone input-level calibration is established.
- No ESP32-S3 end-to-end latency/memory result is established.
- Hindi remains training-only until reliable speaker/client IDs are available in the validation/test sources.
