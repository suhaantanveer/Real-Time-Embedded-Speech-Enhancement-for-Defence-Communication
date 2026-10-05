# V3.2 pre-training fixes

- Exclude noise-only examples from the relative MR-STFT reconstruction loss; keep their ideal-mask/gain loss so they still teach suppression.
- Multi-resolution STFT loss now returns per-sample values and handles zero-target samples safely.
- Added `--overfit` to `train.py` to disable Mic-2 dropout/fault augmentation for a deterministic 32-sample memorization test.
