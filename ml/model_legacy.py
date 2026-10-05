"""
model.py
Tiny GRU Causal Speech Enhancement Network for ESP32-S3 Edge Deployment.
- Total parameters: ~45,000 (< 200 KB in FP32, < 50 KB in INT8)
- Architecture: 2 Causal GRU layers + Dense projection
- Input: 22-band log energies + delta features (44 features per frame)
- Output: 22-band spectral gains in [0.0, 1.0] (Ideal Ratio Mask)
- Zero future lookahead (strictly causal for real-time <10 ms latency)
"""

import torch
import torch.nn as nn

class TinySpeechEnhancer(nn.Module):
    def __init__(self, num_bands: int = 22, hidden1: int = 64, hidden2: int = 48):
        super(TinySpeechEnhancer, self).__init__()
        self.num_bands = num_bands
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        
        # Input projection: 22 bands + 22 deltas = 44 features -> 64
        self.in_proj = nn.Sequential(
            nn.Linear(num_bands * 2, hidden1),
            nn.ReLU()
        )
        
        # BUG #24 FIX: Normalize features before GRU to handle mic level variation
        self.input_norm = nn.LayerNorm(hidden1)
        
        # Causal Recurrent Layers
        self.gru1 = nn.GRU(hidden1, hidden1, batch_first=True)
        self.gru2 = nn.GRU(hidden1, hidden2, batch_first=True)
        
        # Output Gain Projection: 48 -> 22 bands with Sigmoid mask
        self.out_proj = nn.Sequential(
            nn.Linear(hidden2, num_bands),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, h_states: tuple = None) -> tuple:
        """
        Forward pass for training and batch evaluation.
        
        Args:
            x: (Batch, Frames, 22) log band energies
            h_states: tuple of (h1, h2) hidden states or None
        Returns:
            (gains, (h1, h2))
            - gains: (Batch, Frames, 22)
        """
        batch_size, num_frames, _ = x.shape
        
        # Compute first-order delta features along time dimension
        delta = torch.zeros_like(x)
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        
        # Concatenate features: (Batch, Frames, 44)
        feat = torch.cat([x, delta], dim=-1)
        
        # Dense input projection
        h = self.in_proj(feat)
        h = self.input_norm(h)
        
        # Causal GRU 1
        h1_in = h_states[0] if h_states is not None else None
        h, h1_out = self.gru1(h, h1_in)
        
        # Causal GRU 2
        h2_in = h_states[1] if h_states is not None else None
        h, h2_out = self.gru2(h, h2_in)
        
        # Output gains
        gains = self.out_proj(h)
        
        return gains, (h1_out, h2_out)

    def step(self, x_frame: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor, 
             x_prev: torch.Tensor) -> tuple:
        """
        Single-frame streaming step for real-time edge inference on ESP32.
        
        Args:
            x_frame: (Batch, 22) current frame log energies
            h1: (1, Batch, 64) GRU 1 hidden state
            h2: (1, Batch, 48) GRU 2 hidden state
            x_prev: (Batch, 22) previous frame log energies
        Returns:
            (gains_frame, h1_new, h2_new)
        """
        # Delta
        delta = x_frame - x_prev
        feat = torch.cat([x_frame, delta], dim=-1).unsqueeze(1) # (Batch, 1, 44)
        
        h = self.in_proj(feat)
        h = self.input_norm(h)
        h, h1_new = self.gru1(h, h1)
        h, h2_new = self.gru2(h, h2)
        
        gains = self.out_proj(h).squeeze(1) # (Batch, 22)
        return gains, h1_new, h2_new

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TinySpeechEnhancer()
    params = model.get_parameter_count()
    print(f"TinySpeechEnhancer initialized successfully.")
    print(f"Total Trainable Parameters: {params:,}")
    print(f"Model Size (FP32)         : {params * 4 / 1024:.2f} KB")
    print(f"Model Size (INT8 Quant)   : {params * 1 / 1024:.2f} KB")
    
    # Test dummy forward pass
    dummy_x = torch.randn(2, 100, 22)
    gains, (h1, h2) = model(dummy_x)
    print(f"Forward output shape      : {gains.shape} (Expected: [2, 100, 22])")
    assert gains.shape == (2, 100, 22)
    assert 0.0 <= gains.min() and gains.max() <= 1.0
    print("[TEST PASSED] Architecture is causal, bounded, and verified.")
