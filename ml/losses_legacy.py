"""
losses.py
Loss functions for SIH speech enhancement:
1. Scale-Invariant Signal-to-Noise Ratio (SI-SNR) Loss
2. Multi-Resolution Spectral Magnitude Loss (L1 on Linear & Log Spectrograms)
3. Ideal Ratio Mask (IRM) Target Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
    Higher is better (returns dB).
    """
    # Zero-mean normalization
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # Projection: s_target = (<estimate, target> / ||target||^2) * target
    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    proj = (dot / s_target_energy) * target
    
    # Noise residual: e_noise = estimate - proj
    e_noise = estimate - proj
    
    # SI-SNR calculation
    proj_energy = torch.sum(proj ** 2, dim=-1) + eps
    noise_energy = torch.sum(e_noise ** 2, dim=-1) + eps
    
    si_snr = 10.0 * torch.log10(proj_energy / noise_energy)
    return si_snr


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(256, 128, 64), hop_sizes=(128, 64, 32), win_lengths=(256, 128, 64)):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            w = torch.hann_window(win).to(x.device)
            X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=w, return_complex=True)
            Y = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window=w, return_complex=True)
            
            mag_X = torch.abs(X)
            mag_Y = torch.abs(Y)
            
            # Linear spectral loss + Log spectral loss
            sc_loss = torch.mean(torch.abs(mag_X - mag_Y))
            log_loss = torch.mean(torch.abs(torch.log(mag_X + 1e-5) - torch.log(mag_Y + 1e-5)))
            loss += sc_loss + log_loss
            
        return loss / len(self.fft_sizes)


class CompositeEnhancementLoss(nn.Module):
    def __init__(self):
        super(CompositeEnhancementLoss, self).__init__()
        self.mr_stft = MultiResolutionSTFTLoss()

    def forward(self, est_audio: torch.Tensor, clean_audio: torch.Tensor, 
                est_gains: torch.Tensor = None, ideal_mask: torch.Tensor = None) -> tuple:
        # 1. Negative SI-SNR (we minimize loss, so negative SI-SNR)
        si_snr_val = torch.mean(compute_si_snr(est_audio, clean_audio))
        loss_si_snr = -si_snr_val
        
        # 2. Multi-Resolution Spectral Loss
        loss_spectral = self.mr_stft(est_audio, clean_audio)
        
        # 3. Optional Mask Loss
        loss_mask = 0.0
        if est_gains is not None and ideal_mask is not None:
            loss_mask = F.l1_loss(est_gains, ideal_mask)
            
        total_loss = loss_si_snr + 2.0 * loss_spectral + 5.0 * loss_mask
        return total_loss, si_snr_val
