"""
features.py
Causal feature extraction and time-frequency reconstruction:
1. Short-Time Fourier Transform (STFT): 256-point causal window (16 ms) with 160-sample hop (10 ms).
2. 22-Band Bark/Mel Filterbank: Reduces frequency bins to 22 band energies matching RNNoise efficiency.
3. Band Gain Interpolation: Maps 22 predicted neural gains back to 129 linear FFT bins for synthesis.
"""

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
N_FFT = 256
HOP_LENGTH = 160  # 10 ms hop
WIN_LENGTH = 256  # 16 ms window
NUM_BANDS = 22

# Bark scale band division cutoffs (in Hz) from 100 Hz to 8000 Hz
BARK_FREQS = [
    100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 
    1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6350, 7700, 8000
]

def get_bark_filterbank(n_fft: int = N_FFT, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Constructs a 22-band Bark triangular filterbank matrix: (num_bands, n_fft // 2 + 1).
    """
    num_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, num_bins)
    filterbank = np.zeros((NUM_BANDS, num_bins), dtype=np.float32)
    
    # Edges of the 22 bands
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
                
    # Normalize each band filter
    band_sums = np.sum(filterbank, axis=1, keepdims=True) + 1e-7
    filterbank = filterbank / band_sums
    return filterbank


class FeatureExtractor:
    def __init__(self, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, 
                 win_length: int = WIN_LENGTH, sample_rate: int = SAMPLE_RATE):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.filterbank = torch.from_numpy(get_bark_filterbank(n_fft, sample_rate))
        
        # Causal square-root Hann window for perfect analysis-synthesis overlap-add
        w = torch.hann_window(win_length)
        self.win_ana = torch.sqrt(w)
        self.win_syn = torch.sqrt(w)
        self.ola_win = self.win_ana * self.win_syn
        self.pad_left = win_length - hop_length  # 96 samples of past history
        
    def extract_features(self, audio: torch.Tensor) -> tuple:
        """
        Extracts STFT complex spectrum and 22-band log energies using strictly causal framing.
        Uses 96 samples of past history + 160 new samples with zero future lookahead.
        
        Args:
            audio: (Batch, Samples) float32 tensor
        Returns:
            (log_band_energies, stft_complex)
            - log_band_energies: (Batch, Frames, 22)
            - stft_complex: (Batch, Bins, Frames) complex tensor
        """
        device = audio.device
        win_ana = self.win_ana.to(device)
        filterbank = self.filterbank.to(device)
        
        # Causal past-history padding (96 samples on the left, 0 on the right)
        audio_padded = F.pad(audio, (self.pad_left, 0))
        
        # Frame unfolding: (Batch, Frames, 256)
        frames = audio_padded.unfold(dimension=-1, size=self.win_length, step=self.hop_length)
        
        # Windowed frames
        w_frames = frames * win_ana
        
        # RFFT: (Batch, Frames, 129)
        stft = torch.fft.rfft(w_frames, n=self.n_fft, dim=-1)
        stft_complex = stft.transpose(1, 2)  # (Batch, 129, Frames)
        
        # Power spectrum: |X|^2
        power_t = torch.abs(stft) ** 2  # (Batch, Frames, 129)
        
        # Apply Bark filterbank: (Batch, Frames, 22)
        band_energies = torch.matmul(power_t, filterbank.T)
        log_band_energies = torch.log(band_energies + 1e-6)
        
        return log_band_energies, stft_complex
        
    def apply_gains_and_istft(self, stft_complex: torch.Tensor, 
                              band_gains: torch.Tensor, 
                              target_len: int = None) -> torch.Tensor:
        """
        Interpolates 22 predicted band gains to 129 FFT bins, applies gain mask,
        and reconstructs enhanced audio via causal overlap-add.
        
        Args:
            stft_complex: (Batch, 129, Frames) complex tensor
            band_gains: (Batch, Frames, 22) float in [0.0, 1.0]
            target_len: Target output audio length in samples
        """
        device = stft_complex.device
        win_syn = self.win_syn.to(device)
        ola_win = self.ola_win.to(device)
        filterbank = self.filterbank.to(device)
        
        # Matrix to expand 22 band gains back to 129 FFT bins
        expand_weights = filterbank.T / (torch.sum(filterbank.T, dim=1, keepdim=True) + 1e-7)
        
        # (Batch, Frames, 22) @ (22, 129) => (Batch, Frames, 129)
        bin_gains = torch.matmul(band_gains, expand_weights.T)
        bin_gains = bin_gains.transpose(1, 2)  # (Batch, 129, Frames)
        
        # Apply mask
        enhanced_stft = stft_complex * bin_gains  # (Batch, 129, Frames)
        enhanced_stft_t = enhanced_stft.transpose(1, 2)  # (Batch, Frames, 129)
        
        # Inverse RFFT: (Batch, Frames, 256)
        rec_frames = torch.fft.irfft(enhanced_stft_t, n=self.n_fft, dim=-1) * win_syn
        
        # Causal overlap-add synthesis
        batch_size, num_frames, _ = rec_frames.shape
        total_len = (num_frames - 1) * self.hop_length + self.win_length
        
        out = torch.zeros((batch_size, total_len), device=device, dtype=rec_frames.dtype)
        ola = torch.zeros(total_len, device=device, dtype=rec_frames.dtype)
        
        for f in range(num_frames):
            idx = f * self.hop_length
            out[:, idx:idx + self.win_length] += rec_frames[:, f, :]
            ola[idx:idx + self.win_length] += ola_win
            
        valid = ola > 1e-4
        out[:, valid] = out[:, valid] / ola[valid]
        
        # Strip past-history left padding to align with original time axis
        enhanced_audio = out[:, self.pad_left:]
        if target_len is not None:
            enhanced_audio = enhanced_audio[:, :target_len]
            if enhanced_audio.shape[-1] < target_len:
                enhanced_audio = F.pad(enhanced_audio, (0, target_len - enhanced_audio.shape[-1]))
                
        return enhanced_audio
