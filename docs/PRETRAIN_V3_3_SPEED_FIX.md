# Pre-training V3.3 speed fix

The training pipeline was functionally correct but CPU-bound on 3-second clips.
V3.3 keeps the same DSP/features/model semantics and accelerates only execution:

- `ml/front_end.py`: vectorized DC blocker + HPF with `scipy.signal.lfilter`; impulse-guard gain envelope is vectorized per frame.
- `ml/features.py`: vectorized ISTFT overlap-add with `torch.nn.functional.fold`; numerical output matches the previous implementation exactly on the regression test.
- `ml/train.py`: non-blocking CUDA transfers and defaults of batch size 16 / 4 data-loader workers.

Regression checks:
- Front-end with and without guard: max difference 0.0 on 16k random test audio.
- Feature/STFT: max difference 0.0.
- ISTFT after arbitrary gains: max difference 0.0.

This does not change the dataset or training objective.
