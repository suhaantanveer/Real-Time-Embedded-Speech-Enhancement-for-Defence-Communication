"""Losses for speech preservation and enhancement."""
from __future__ import annotations

import torch
import torch.nn as nn


def compute_si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target * target, dim=-1, keepdim=True) + eps
    proj = (dot / target_energy) * target
    noise = estimate - proj
    return 10.0 * torch.log10((torch.sum(proj * proj, dim=-1) + eps) /
                              (torch.sum(noise * noise, dim=-1) + eps))


def speech_active_frame_mask(clean_audio: torch.Tensor, threshold_db: float = -35.0,
                             frame: int = 160) -> torch.Tensor:
    """Return a [B,Tframes] active-speech mask using the clean target only."""
    n = clean_audio.shape[-1] // frame
    if n == 0:
        return clean_audio.new_zeros((*clean_audio.shape[:-1], 0), dtype=torch.bool)
    trimmed = clean_audio[..., :n * frame].reshape(*clean_audio.shape[:-1], n, frame)
    power = torch.mean(trimmed * trimmed, dim=-1)
    peak = torch.amax(power, dim=-1, keepdim=True)
    threshold = peak * (10.0 ** (threshold_db / 10.0))
    return power > threshold


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(256, 128, 64), hop_sizes=(128, 64, 32), win_lengths=(256, 128, 64)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Return one loss value per sample. This is important because the clean
        # target is exactly zero for noise-only examples; relative spectral
        # convergence is undefined there and can explode. The caller can mask
        # those samples and still use gain/band losses to teach suppression.
        total = x.new_zeros(x.shape[0])
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            w = torch.hann_window(win, device=x.device)
            X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                           window=w, return_complex=True)
            Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win,
                           window=w, return_complex=True)
            mx = torch.abs(X)
            my = torch.abs(Y)
            numerator = torch.sqrt(torch.sum((mx - my) ** 2, dim=(-2, -1)) + 1e-8)
            target_norm = torch.sqrt(torch.sum(my ** 2, dim=(-2, -1)))
            safe = target_norm > 1e-6
            spectral_convergence = torch.zeros_like(target_norm)
            spectral_convergence[safe] = numerator[safe] / (target_norm[safe] + 1e-8)
            # Log-magnitude loss is finite for silence because of the floor.
            log_mag = torch.mean(torch.abs(torch.log(mx + 1e-5) - torch.log(my + 1e-5)), dim=(-2, -1))
            total = total + spectral_convergence + log_mag
        return total / len(self.fft_sizes)


def frame_rms(x: torch.Tensor, frame: int = 160) -> torch.Tensor:
    n = x.shape[-1] // frame
    if n == 0:
        return x.new_zeros((*x.shape[:-1], 0))
    trimmed = x[..., :n * frame].reshape(*x.shape[:-1], n, frame)
    return torch.sqrt(torch.mean(trimmed * trimmed, dim=-1) + 1e-10)


class CompositeEnhancementLoss(nn.Module):
    def __init__(self, w_mag: float = 1.0, w_gain: float = 1.0, w_level: float = 0.5, w_si: float = 0.1):
        super().__init__()
        self.mr_stft = MultiResolutionSTFTLoss()
        self.w_mag = w_mag
        self.w_gain = w_gain
        self.w_level = w_level
        self.w_si = w_si

    def forward(self, est_audio: torch.Tensor, clean_audio: torch.Tensor,
                est_gains: torch.Tensor | None = None,
                ideal_mask: torch.Tensor | None = None,
                has_speech: torch.Tensor | None = None):
        mag_per_sample = self.mr_stft(est_audio, clean_audio)
        if has_speech is None:
            speech_sel = torch.ones(est_audio.shape[0], dtype=torch.bool, device=est_audio.device)
        else:
            speech_sel = has_speech.reshape(-1) > 0.5
        # Do not apply reconstruction spectral loss to noise-only examples.
        # Their zero target makes relative spectral convergence meaningless;
        # their ideal-mask/gain loss still teaches the model to suppress noise.
        l_mag = torch.mean(mag_per_sample[speech_sel]) if torch.any(speech_sel) else est_audio.new_tensor(0.0)
        if est_gains is not None and ideal_mask is not None:
            l_gain = torch.mean(torch.abs(est_gains - ideal_mask))
        else:
            l_gain = est_audio.new_tensor(0.0)

        est_rms = frame_rms(est_audio)
        clean_rms = frame_rms(clean_audio)
        active = speech_active_frame_mask(clean_audio)
        if torch.any(active):
            l_level = torch.mean(torch.abs(
                20.0 * torch.log10(est_rms[active] + 1e-8) -
                20.0 * torch.log10(clean_rms[active] + 1e-8)
            )) / 20.0
        else:
            l_level = est_audio.new_tensor(0.0)

        per_sample_si = compute_si_snr(est_audio, clean_audio)
        if torch.any(speech_sel):
            si = torch.mean(per_sample_si[speech_sel])
        else:
            si = est_audio.new_tensor(0.0)

        total = self.w_mag * l_mag + self.w_gain * l_gain + self.w_level * l_level - self.w_si * si
        stats = {
            "loss_mag": l_mag.detach(),
            "loss_gain": l_gain.detach(),
            "loss_level": l_level.detach(),
            "si_snr": si.detach(),
        }
        return total, stats
