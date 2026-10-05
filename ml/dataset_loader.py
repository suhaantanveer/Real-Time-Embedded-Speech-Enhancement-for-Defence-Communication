"""Dataset loader for the redesigned Mic-1 primary + optional Mic-2 model."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from config import FRAME_SIZE, MIC2_DROPOUT_RATE, MIC2_FAULT_RATE, SAMPLE_RATE
from features import FeatureExtractor
from front_end import MicFrontEnd


class DefenceAudioDataset(Dataset):
    def __init__(self, manifest_path: str | os.PathLike, split: str = "train",
                 target_duration_sec: float = 3.0, sample_rate: int = SAMPLE_RATE,
                 mic2_dropout: float = MIC2_DROPOUT_RATE, return_details: bool = False,
                 force_ref_invalid: bool = False, base_seed: int = 12345):
        self.sample_rate = sample_rate
        self.target_len = int(target_duration_sec * sample_rate)
        self.split = split
        self.mic2_dropout = float(mic2_dropout if split == "train" else 0.0)
        self.mic2_fault_rate = float(MIC2_FAULT_RATE if split == "train" else 0.0)
        self.return_details = return_details
        self.force_ref_invalid = bool(force_ref_invalid)
        self.base_seed = int(base_seed)
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        self.samples = [e for e in entries if e.get("split") == split]
        self.extractor = FeatureExtractor()
        print(f"Loaded {len(self.samples)} samples for split '{split}' from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _mono_16k(path: str) -> np.ndarray:
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Expected 16 kHz WAV: {path} (got {sr} Hz)")
        return np.asarray(x, dtype=np.float32)

    def _rng(self, idx: int) -> np.random.Generator:
        worker = get_worker_info()
        worker_seed = 0 if worker is None else int(worker.seed)
        return np.random.default_rng(self.base_seed + idx * 1_000_003 + worker_seed)

    def _aligned_crop(self, clean: np.ndarray, primary: np.ndarray, reference: np.ndarray,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = min(len(clean), len(primary), len(reference))
        if n >= self.target_len:
            if self.split == "train":
                start = int(rng.integers(0, n - self.target_len + 1))
            else:
                start = (n - self.target_len) // 2
            end = start + self.target_len
            return clean[start:end].astype(np.float32), primary[start:end].astype(np.float32), reference[start:end].astype(np.float32)
        pad = self.target_len - n
        return (np.pad(clean[:n], (0, pad)).astype(np.float32),
                np.pad(primary[:n], (0, pad)).astype(np.float32),
                np.pad(reference[:n], (0, pad)).astype(np.float32))

    def __getitem__(self, idx: int):
        entry = self.samples[idx]
        rng = self._rng(idx)
        clean_raw = self._mono_16k(entry["paths"]["clean"])
        primary = self._mono_16k(entry["paths"]["primary_mic"])
        reference = self._mono_16k(entry["paths"]["ref_mic"])
        clean, primary, reference = self._aligned_crop(clean_raw, primary, reference, rng)

        # Input front-end includes the impulse guard on mic 1.
        fe = MicFrontEnd(self.sample_rate)
        primary_fe, ref_fe, impulse_frames = fe.process_audio(primary, reference, apply_guard=True)
        # Target follows the same linear DC/HPF path but is never impulse-guarded.
        clean_fe = MicFrontEnd(self.sample_rate)
        clean_target, _, _ = clean_fe.process_audio(clean, np.zeros_like(clean), apply_guard=False)

        ref_valid = 1.0
        unrelated_ref = False
        if self.force_ref_invalid:
            ref_fe[:] = 0.0
            ref_valid = 0.0
        elif self.split == "train":
            u = float(rng.random())
            if u < self.mic2_dropout:
                ref_fe[:] = 0.0
                ref_valid = 0.0
            elif u < self.mic2_dropout + self.mic2_fault_rate:
                # Faulty mic: it remains marked present, but carries unrelated noise.
                ref_fe = (rng.standard_normal(len(ref_fe)).astype(np.float32) * 0.01)
                ref_valid = 1.0
                unrelated_ref = True

        primary_t = torch.from_numpy(primary_fe).unsqueeze(0)
        ref_t = torch.from_numpy(ref_fe).unsqueeze(0)
        clean_t = torch.from_numpy(clean_target).unsqueeze(0)
        with torch.no_grad():
            features, stft, _ = self.extractor.extract_dual(
                primary_t, ref_t, torch.tensor([[ref_valid]], dtype=torch.float32)
            )
            clean_stft, clean_power = self.extractor._stft(clean_t)
            noise = primary_fe - clean_target
            noise_t = torch.from_numpy(noise).unsqueeze(0)
            _, noise_power = self.extractor._stft(noise_t)
            clean_bands = self.extractor._band_log_energy(clean_power)
            noise_bands = self.extractor._band_log_energy(noise_power)
            clean_energy = torch.exp(clean_bands)
            noise_energy = torch.exp(noise_bands)
            ideal = torch.sqrt(clean_energy / (clean_energy + noise_energy + 1e-8))
            ideal = torch.clamp(ideal, min=0.1, max=1.0)

        condition = str(entry.get("condition", "unknown"))
        item = {
            "features": features.squeeze(0),
            "stft": stft.squeeze(0),
            "clean": clean_t.squeeze(0),
            "primary": primary_t.squeeze(0),
            "ideal_mask": ideal.squeeze(0),
            "ref_valid": torch.tensor(ref_valid, dtype=torch.float32),
            "has_speech": torch.tensor(0.0 if condition == "noise_only" else 1.0, dtype=torch.float32),
            "is_clean_condition": torch.tensor(1.0 if condition == "clean" else 0.0, dtype=torch.float32),
            "impulse_frames": torch.tensor(impulse_frames, dtype=torch.int32),
            "unrelated_ref": torch.tensor(1 if unrelated_ref else 0, dtype=torch.int32),
            "meta": json.dumps(entry, ensure_ascii=False),
        }
        if self.return_details:
            return item
        return item["features"], item["stft"], item["clean"], item["ideal_mask"]


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def get_dataloaders(manifest_path: str, batch_size: int = 4,
                    target_duration_sec: float = 3.0, num_workers: int = 0):
    train_set = DefenceAudioDataset(manifest_path, "train", target_duration_sec)
    val_set = DefenceAudioDataset(manifest_path, "val", target_duration_sec)
    test_set = DefenceAudioDataset(manifest_path, "test", target_duration_sec)
    kwargs = dict(num_workers=num_workers, worker_init_fn=seed_worker,
                  pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(test_set, batch_size=1, shuffle=False, **kwargs),
    )
