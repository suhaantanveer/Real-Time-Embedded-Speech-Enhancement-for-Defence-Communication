"""Causal 16 kHz STFT features for the Mic-1 primary + optional Mic-2 design."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from config import BARK_FREQS, HOP_LENGTH, MIC2_CALIBRATION_DB, N_FFT, NUM_BANDS, SAMPLE_RATE, WIN_LENGTH


def get_bark_filterbank(n_fft: int = N_FFT, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    num_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, num_bins)
    filterbank = np.zeros((NUM_BANDS, num_bins), dtype=np.float32)
    edges = [0.0] + BARK_FREQS
    for b in range(NUM_BANDS):
        f_low = edges[b]
        f_center = edges[b + 1]
        f_high = edges[b + 2] if b + 2 < len(edges) else sample_rate / 2
        for k, f in enumerate(fft_freqs):
            if f_low <= f < f_center:
                filterbank[b, k] = (f - f_low) / (f_center - f_low + 1e-7)
            elif f_center <= f <= f_high:
                filterbank[b, k] = (f_high - f) / (f_high - f_center + 1e-7)
    filterbank /= np.sum(filterbank, axis=1, keepdims=True) + 1e-7
    return filterbank


class FeatureExtractor:
    def __init__(self, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                 win_length: int = WIN_LENGTH, sample_rate: int = SAMPLE_RATE,
                 mic2_calibration_db: np.ndarray | None = None):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.filterbank = torch.from_numpy(get_bark_filterbank(n_fft, sample_rate))
        self.calibration_db = np.asarray(
            MIC2_CALIBRATION_DB if mic2_calibration_db is None else mic2_calibration_db,
            dtype=np.float32,
        )
        if self.calibration_db.shape != (NUM_BANDS,):
            raise ValueError("mic2_calibration_db must contain 22 values")
        # sqrt-Hann analysis/synthesis gives unity window product at conventional 50% overlap;
        # the implementation keeps explicit OLA normalization for 160-sample hop.
        w = torch.hann_window(win_length)
        self.win_ana = torch.sqrt(w)
        self.win_syn = torch.sqrt(w)
        self.ola_win = self.win_ana * self.win_syn
        self.pad_left = win_length - hop_length

    def _stft(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = audio.device
        frames = F.pad(audio, (self.pad_left, 0)).unfold(-1, self.win_length, self.hop_length)
        stft = torch.fft.rfft(frames * self.win_ana.to(device), n=self.n_fft, dim=-1)
        power = torch.abs(stft) ** 2
        return stft, power

    def _band_log_energy(self, power: torch.Tensor) -> torch.Tensor:
        fb = self.filterbank.to(power.device)
        return torch.log(torch.matmul(power, fb.T) + 1e-6)

    def extract_dual(self, mic1: torch.Tensor, mic2: torch.Tensor | None = None,
                     ref_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return 89-dim features, mic1 STFT and mic1 log energies.

        Feature layout per frame:
        0:22   mic1 log band energy
        22:44  mic1 first-order delta
        44:66  mic2 log band energy (zeros if absent)
        66:88  calibrated mic1-minus-mic2 log energy
        88     ref_valid flag
        """
        if mic1.ndim != 2:
            raise ValueError("mic1 must be [batch, samples]")
        if mic2 is None:
            mic2 = torch.zeros_like(mic1)
            present = torch.zeros((mic1.shape[0], 1), dtype=mic1.dtype, device=mic1.device)
        else:
            if mic2.shape != mic1.shape:
                raise ValueError("mic1 and mic2 must have identical shape")
            if ref_valid is None:
                present = torch.ones((mic1.shape[0], 1), dtype=mic1.dtype, device=mic1.device)
            else:
                present = ref_valid.reshape(-1, 1).to(mic1.dtype)

        stft1, p1 = self._stft(mic1)
        _, p2 = self._stft(mic2)
        log1 = self._band_log_energy(p1)
        log2 = self._band_log_energy(p2)
        delta1 = torch.zeros_like(log1)
        if log1.shape[1] > 1:
            delta1[:, 1:, :] = log1[:, 1:, :] - log1[:, :-1, :]

        calib = torch.as_tensor(self.calibration_db, dtype=log1.dtype, device=log1.device).view(1, 1, NUM_BANDS) / 10.0 * np.log(10.0)
        level_diff = log1 - (log2 + calib)
        if present is not None:
            # Broadcast the validity flag across time/bands.
            log2 = log2 * present.unsqueeze(1)
            level_diff = level_diff * present.unsqueeze(1)
        flag = present.expand(-1, log1.shape[1]).unsqueeze(-1)
        features = torch.cat([log1, delta1, log2, level_diff, flag], dim=-1)
        return features, stft1.transpose(1, 2), log1

    def extract_features(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mic-1-only compatibility wrapper."""
        features, stft, _ = self.extract_dual(audio, None, None)
        return features, stft

    def apply_gains_and_istft(self, stft_complex: torch.Tensor,
                              band_gains: torch.Tensor,
                              target_len: int | None = None) -> torch.Tensor:
        device = stft_complex.device
        fb = self.filterbank.to(device)
        expand_weights = fb.T / (torch.sum(fb.T, dim=1, keepdim=True) + 1e-7)
        bin_gains = torch.matmul(band_gains, expand_weights.T).transpose(1, 2)
        enhanced = stft_complex * bin_gains
        frames = torch.fft.irfft(enhanced.transpose(1, 2), n=self.n_fft, dim=-1)
        frames = frames * self.win_syn.to(device)

        bsz, n_frames, _ = frames.shape
        total_len = (n_frames - 1) * self.hop_length + self.win_length

        # Vectorized overlap-add. The previous implementation launched one
        # tensor update per STFT frame, which made training GPU-bound on kernel
        # launch overhead for a tiny model. F.fold performs the same OLA in one
        # batched operation.
        frames_2d = frames
        cols = frames_2d.transpose(1, 2).unsqueeze(2)  # [B, win, 1, n_frames]
        out = F.fold(
            cols.reshape(bsz, self.win_length, 1, n_frames).squeeze(2),
            output_size=(1, total_len),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze(1).squeeze(1)

        ola_cols = self.ola_win.to(device).view(1, self.win_length, 1, 1).expand(bsz, -1, 1, n_frames)
        ola = F.fold(
            ola_cols.reshape(bsz, self.win_length, 1, n_frames).squeeze(2),
            output_size=(1, total_len),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze(1).squeeze(1)
        ola_1d = ola[0]
        valid = ola_1d > 1e-4
        out[:, valid] /= ola_1d[valid]
        out = out[:, self.pad_left:]
        if target_len is not None:
            out = out[:, :target_len]
            if out.shape[-1] < target_len:
                out = F.pad(out, (0, target_len - out.shape[-1]))
        return out
