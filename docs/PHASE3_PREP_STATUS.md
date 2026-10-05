# Phase 3 preparation status

## Completed

- Phase-1 speech corpus: 6,148 usable files / 10.231 hours.
- LibriSpeech speaker-disjoint manifests: 2,176 train / 261 val / 257 test, 40 speakers.
- Common Voice Hindi: 3,454 files, kept as training-only because speaker/client IDs were unavailable.
- `ml/data/mix_dataset.py` now consumes `speech.csv` and the generated speaker-disjoint manifests instead of inferring speaker IDs from directory names.
- Generated sample counts are explicit per split (`--train-samples`, `--val-samples`, `--test-samples`) so the large Hindi training corpus cannot accidentally consume almost the entire generated dataset.
- Smoke test passed: 14 generated examples (8 train / 3 val / 3 test), Hindi appeared only in train, and LibriSpeech speakers remained separated.
- One-epoch CPU training smoke test passed on the synthetic smoke corpus; this is only a plumbing check, not a model-performance result.

## Not yet claimed

- No real noise corpus has been integrated yet.
- No full V2 training has been run.
- No SI-SDR/STOI performance result is established.
- No ESP32-S3 latency result is established.
