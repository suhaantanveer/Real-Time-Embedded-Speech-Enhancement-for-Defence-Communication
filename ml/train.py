"""Training loop for the redesigned Mic-1-primary speech enhancer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import G_MIN
from dataset_loader import DefenceAudioDataset, seed_worker
from features import FeatureExtractor
from losses import CompositeEnhancementLoss, compute_si_snr, frame_rms, speech_active_frame_mask
from model import TinySpeechEnhancer


def snr_bin(value) -> str:
    if value is None:
        return "unknown"
    v = float(value)
    if v <= 0:
        return "<=0"
    if v < 10:
        return "0_10"
    if v < 20:
        return "10_20"
    return ">=20"


def stratified_indices(dataset: DefenceAudioDataset, max_items: int = 500, seed: int = 1234) -> list[int]:
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(dataset.samples):
        key = f"{e.get('condition','unknown')}|{e.get('noise_class','unknown')}|{snr_bin(e.get('snr_db'))}"
        groups.setdefault(key, []).append(i)
    rng = np.random.default_rng(seed)
    for g in groups.values():
        rng.shuffle(g)
    # Round-robin across strata so the subset is not dominated by the first rows.
    selected: list[int] = []
    while len(selected) < min(max_items, len(dataset)):
        progressed = False
        for key in sorted(groups):
            bucket = groups[key]
            if selected and len(selected) >= max_items:
                break
            if bucket:
                selected.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    return selected[:max_items]


def loader_for(dataset, indices, batch_size, shuffle=False, num_workers=0):
    subset = Subset(dataset, indices) if indices is not None else dataset
    kwargs = dict(num_workers=num_workers, worker_init_fn=seed_worker,
                  pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=False, **kwargs)


def batch_metrics(model: TinySpeechEnhancer, loader, extractor: FeatureExtractor,
                  device: torch.device) -> dict[str, float]:
    model.eval()
    raw_si = []
    enh_si = []
    clean_gain_vals = []
    clean_level_vals = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            stft = batch["stft"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            gains, _ = model(features)
            enhanced = extractor.apply_gains_and_istft(stft, gains, target_len=clean.shape[-1])
            raw = extractor.apply_gains_and_istft(stft, torch.ones_like(gains), target_len=clean.shape[-1])

            has_speech = batch["has_speech"].to(device) > 0.5
            if torch.any(has_speech):
                raw_si.extend(compute_si_snr(raw[has_speech], clean[has_speech]).detach().cpu().tolist())
                enh_si.extend(compute_si_snr(enhanced[has_speech], clean[has_speech]).detach().cpu().tolist())

            clean_cond = batch["is_clean_condition"].to(device) > 0.5
            if torch.any(clean_cond):
                clean_frames = speech_active_frame_mask(clean[clean_cond])
                per_frame_gain = gains[clean_cond].mean(dim=-1)
                if torch.any(clean_frames):
                    clean_gain_vals.extend(per_frame_gain[clean_frames].detach().cpu().tolist())
                er = frame_rms(enhanced[clean_cond])
                cr = frame_rms(clean[clean_cond])
                if torch.any(clean_frames):
                    clean_level_vals.extend((20.0 * torch.log10((er[clean_frames] + 1e-8) / (cr[clean_frames] + 1e-8))).detach().cpu().tolist())

    raw_mean = float(np.mean(raw_si)) if raw_si else float("nan")
    enh_mean = float(np.mean(enh_si)) if enh_si else float("nan")
    clean_gain = float(np.mean(clean_gain_vals)) if clean_gain_vals else float("nan")
    clean_level = float(np.mean(clean_level_vals)) if clean_level_vals else float("nan")
    return {
        "raw_si_snr": raw_mean,
        "enhanced_si_snr": enh_mean,
        "si_snr_improvement": enh_mean - raw_mean if np.isfinite(raw_mean) and np.isfinite(enh_mean) else float("nan"),
        "clean_speech_mean_gain": clean_gain,
        "clean_level_error_db": clean_level,
    }


def train(manifest_path: str, epochs: int = 50, batch_size: int = 4, lr: float = 1e-3,
          output_dir: str = "checkpoints", device_name: str | None = None,
          num_workers: int = 0, val_items: int = 500, overfit: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    train_set = DefenceAudioDataset(manifest_path, split="train", target_duration_sec=3.0, return_details=True, base_seed=9001)
    if overfit:
        # Deterministic, no-augmentation memorization test.
        train_set.mic2_dropout = 0.0
        train_set.mic2_fault_rate = 0.0
    val_set = DefenceAudioDataset(manifest_path, split="val", target_duration_sec=3.0, return_details=True, base_seed=9002)
    val_no_ref_set = DefenceAudioDataset(manifest_path, split="val", target_duration_sec=3.0, return_details=True,
                                         force_ref_invalid=True, base_seed=9002)
    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError("Training/validation manifest is empty")

    val_indices = stratified_indices(val_set, max_items=val_items, seed=777)
    print(f"[INFO] Validation subset: {len(val_indices)} fixed stratified examples")
    train_loader = loader_for(train_set, None, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = loader_for(val_set, val_indices, batch_size, shuffle=False, num_workers=num_workers)
    val_no_ref_loader = loader_for(val_no_ref_set, val_indices, batch_size, shuffle=False, num_workers=num_workers)

    model = TinySpeechEnhancer().to(device)
    extractor = FeatureExtractor()
    criterion = CompositeEnhancementLoss(w_mag=1.0, w_gain=1.0, w_level=0.5, w_si=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f"[INFO] Parameters: {model.get_parameter_count():,}")

    best_score = -float("inf")
    best_path = os.path.join(output_dir, "best_tiny_enhancer.pth")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "mag": 0.0, "gain": 0.0, "level": 0.0, "si": 0.0}
        steps = 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            stft = batch["stft"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            ideal = batch["ideal_mask"].to(device, non_blocking=True)
            has_speech = batch["has_speech"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            gains, _ = model(features)
            enhanced = extractor.apply_gains_and_istft(stft, gains, target_len=clean.shape[-1])
            loss, stats = criterion(enhanced, clean, gains, ideal, has_speech=has_speech)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["mag"] += float(stats["loss_mag"])
            totals["gain"] += float(stats["loss_gain"])
            totals["level"] += float(stats["loss_level"])
            totals["si"] += float(stats["si_snr"])
            steps += 1

        scheduler.step()
        metrics = batch_metrics(model, val_loader, extractor, device)
        no_ref = batch_metrics(model, val_no_ref_loader, extractor, device)
        # Checkpoint score rewards real noisy-speech improvement and explicitly
        # penalizes clean-speech distortion rather than treating all frames equally.
        score = metrics["si_snr_improvement"]
        if np.isfinite(metrics["clean_speech_mean_gain"]):
            score -= 5.0 * abs(metrics["clean_speech_mean_gain"] - 1.0)
        if np.isfinite(metrics["clean_level_error_db"]):
            score -= 0.10 * abs(metrics["clean_level_error_db"])

        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **{k: v / max(1, steps) for k, v in totals.items()},
            **metrics,
            "mic2_present_si_snr_improvement": metrics["si_snr_improvement"],
            "mic2_absent_si_snr_improvement": no_ref["si_snr_improvement"],
            "mic2_delta_si_snr": metrics["si_snr_improvement"] - no_ref["si_snr_improvement"],
            "validation_score": score,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs} | loss={row['loss']:.4f} | "
            f"val ΔSI-SNR={metrics['si_snr_improvement']:+.2f} dB | "
            f"clean gain={metrics['clean_speech_mean_gain']:.3f} | "
            f"clean level={metrics['clean_level_error_db']:+.2f} dB | score={score:.3f}"
        )
        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": row,
                "validation_score": score,
                "config": {"g_min": G_MIN, "features": 89, "hidden1": 64, "hidden2": 48},
            }, best_path)
            print(f"  -> saved {best_path}")

    hist_path = Path(output_dir) / "training_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[DONE] Best validation score: {best_score:.4f}; checkpoint: {best_path}")
    return best_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", default="checkpoints_v3")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-items", type=int, default=500)
    p.add_argument("--overfit", action="store_true", help="Disable training-time mic2 augmentation for the tiny memorization test")
    args = p.parse_args()
    train(args.manifest, args.epochs, args.batch_size, args.lr, args.output_dir, args.device, args.num_workers, args.val_items)
