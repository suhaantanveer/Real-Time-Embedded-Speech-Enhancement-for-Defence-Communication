# Pre-training V3 fixes

This package incorporates the review findings that should be resolved before the real 25k-example training run.

## Dataset / front-end fixes

1. **Impulse guard**
   - Energy trigger compares against the previous EMA before the current frame is folded into the EMA.
   - Separate absolute peak threshold is now 0.90 instead of using the 20 dB setting as an amplitude threshold.
   - The first frame initializes the EMA without triggering.
   - Python and ESP32 C use the same constants.
   - Clean targets receive the same causal DC blocker + 100 Hz HPF, but never the impulse guard.

2. **Mic-2 environmental-noise coupling**
   - Mic-2 environmental noise is scaled from the actual Mic-1 noise component with a configurable -3 to +10 dB path offset.
   - Clean-primary examples do not receive an unrelated loud environmental noise signal on Mic 2.
   - Variable speech leakage remains intentionally randomized.

3. **Held-out acoustic sources**
   - Noise files are deterministically partitioned 80/10/10, stratified by noise class.
   - RIR files are deterministically partitioned 80/10/10.
   - No source-file overlap is allowed across train/validation/test.

4. **Noise sampling**
   - The four intended classes (stationary, nonstationary, impulsive, ambient) are sampled evenly rather than letting the largest corpus dominate.
   - The manually tagged `review` noise pool is excluded until it is explicitly audited for speech contamination.

5. **RIR realism**
   - The early speech RIR is aligned to the dominant arrival before trimming.
   - Speech, Mic-1 noise, and Mic-2 noise use independently sampled RIRs.
   - Noise uses the full RIR rather than discarding late reverberation.

6. **Level handling**
   - Speech target RMS range is capped at -45 to -15 dBFS for headroom.
   - Any final global safety scaling is transparent and recorded in the manifest as `mix_scale_db` rather than being an implicit compressor.
   - Manifest records requested and measured primary SNR plus leakage and Mic-2 level offset.

## Loader / loss / validation fixes

- Clean/primary/reference crops use the same temporal crop.
- Faulty-reference examples keep `ref_valid=1` with unrelated noise; normal dropout uses `ref_valid=0`.
- Noise-only samples are excluded from SI-SNR loss and validation SI-SNR metrics.
- Clean-speech preservation is measured only on speech-active frames of clean-condition samples.
- Multi-resolution STFT loss uses normalized spectral convergence.
- Checkpoints are selected by noisy-speech SI-SNR improvement penalized by clean-speech gain/level distortion.
- Validation uses a fixed stratified subset of at least 500 samples and runs a Mic-2-present vs Mic-2-absent A/B comparison.
- DataLoader worker seeding is explicit for multi-worker loading.

## Verified in this package

- Python syntax checks pass for the changed ML files.
- Python/C front-end equivalence test passes after the guard change.
- Synthetic dataset smoke test reports zero noise-source overlap and zero RIR-source overlap.
- Synthetic one-epoch training smoke test completes and saves a checkpoint.

## Not yet claimed

No real V3 training metric is established. The full dataset must be regenerated with the new mixer before training.


## V3.1 noise/SNR hardening (2026-10-03)

- Noise crops are now rejected when the filtered acoustic path is effectively silent.
- This prevents sparse impulsive recordings (impact/thunder) from producing near-zero noise energy when a random 3-second crop misses the event.
- Mic-1 noise is hard-rescaled after speech level scaling to the requested target SNR.
- Every `noisy` and `near_clean` sample now has a hard SNR invariant: absolute target-vs-actual error must be <= 0.25 dB or generation aborts. The `clean` condition has zero noise by construction, so its SNR is not treated as a finite-valued metric.
- Mic-2 noise paths are also checked for usable energy before scaling.
- Manifest records `snr_error_db`, `noise_attempts`, and `mic2_noise_attempts` for auditability.
