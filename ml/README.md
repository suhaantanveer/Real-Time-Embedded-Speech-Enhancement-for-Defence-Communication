# SIH26052 — V2 Embedded Speech Enhancement

The redesigned default path is **Mic-1-primary speech enhancement** with optional Mic-2 side information.

```text
Mic 1 -> DC block -> 100 Hz HPF -> impulse guard -> STFT -> 44 mic1 features --+
                                                                                    +-> 89-input GRU -> 22 gains -> ISTFT -> dry/wet safeguard
Mic 2 -> DC block -> 100 Hz HPF -> STFT -> 44 optional side features ------------+
```

Mic 2 is never directly subtracted from Mic 1 in the default path. `ref_valid=0` disables its contribution.

## Model

- Inputs: 89 per-frame features
- 64-unit projection / GRU layer
- 48-unit GRU layer
- 22 band gains
- Gain floor `G_MIN=0.1`
- Current architecture: 48,342 trainable parameters
- Causal 256-point STFT, 160-sample hop, 22 Bark bands

## Training data

The repository did not contain the original dataset mixer or source data. Use `ml/data/mix_dataset.py` with locally supplied speech/noise/RIR corpora.

Example:

```powershell
python ml/data/mix_dataset.py `
  --speech-dir <speech> `
  --speech-metadata <speech>\metadata\speech.csv `
  --noise-dir <noise> `
  --rir-dir <rir> `
  --out-dir dataset/generated `
  --train-samples 50000 `
  --val-samples 5000 `
  --test-samples 5000

python ml/train.py --manifest dataset/generated/manifest.json --epochs 50 --batch-size 4
python ml/export.py --model checkpoints_v2/best_tiny_enhancer.pth --output-dir exported
```

The Phase-1 speech metadata is authoritative for speaker identity and split membership.
LibriSpeech uses the generated speaker-disjoint train/validation/test manifests. Common Voice Hindi with unknown speaker IDs is training-only until speaker/client metadata is recovered.
The builder also creates mic-2 leakage, controlled SNR buckets, and noise-class labels.

## Validation targets

These are targets, not current measured results:

- Clean speech mean gain >= 0.9
- Clean output RMS within +/-1.5 dB of target
- At 0 dB SNR: >=8 dB SI-SDR improvement and >=0.10 STOI improvement
- At <=0 dB SNR: Mic-2-present should beat Mic-2-absent by >=1.5 dB SI-SDR
- At >=10 dB SNR: Mic-2-present and absent within 1 dB
- Complete ESP32 frame processing average <4 ms, worst <8 ms

## Useful checks

```powershell
python ml/tools/gain_level_probe.py --model checkpoints/best_tiny_enhancer.pth
python ml/tools/v2_gain_level_probe.py --model checkpoints_v2/best_tiny_enhancer.pth
python ml/export_test_vectors.py
python ml/evaluate.py --model checkpoints_v2/best_tiny_enhancer.pth --manifest dataset/generated/manifest.json
```

`checkpoints/bootstrap_v2_from_legacy.pth` exists only to allow firmware smoke tests while a real V2 model is being trained. It is explicitly **not** a V2 performance result.
