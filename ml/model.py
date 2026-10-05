"""Small causal GRU speech enhancer for ESP32-S3."""
from __future__ import annotations

import torch
import torch.nn as nn

from config import G_MIN, HIDDEN1, HIDDEN2, N_FEATURES, NUM_BANDS


class TinySpeechEnhancer(nn.Module):
    def __init__(self, num_features: int = N_FEATURES, num_bands: int = NUM_BANDS,
                 hidden1: int = HIDDEN1, hidden2: int = HIDDEN2, g_min: float = G_MIN):
        super().__init__()
        self.num_features = num_features
        self.num_bands = num_bands
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.g_min = float(g_min)
        self.in_proj = nn.Linear(num_features, hidden1)
        self.relu = nn.ReLU()
        self.input_norm = nn.LayerNorm(hidden1)
        self.gru1 = nn.GRU(hidden1, hidden1, batch_first=True)
        self.gru2 = nn.GRU(hidden1, hidden2, batch_first=True)
        self.out_proj = nn.Linear(hidden2, num_bands)

    def _gains(self, z: torch.Tensor) -> torch.Tensor:
        return self.g_min + (1.0 - self.g_min) * torch.sigmoid(z)

    def forward(self, features: torch.Tensor, h_states: tuple[torch.Tensor, torch.Tensor] | None = None):
        if features.ndim != 3 or features.shape[-1] != self.num_features:
            raise ValueError(f"expected [batch, frames, {self.num_features}] features")
        h = self.input_norm(self.relu(self.in_proj(features)))
        h1_in = h_states[0] if h_states is not None else None
        h, h1 = self.gru1(h, h1_in)
        h2_in = h_states[1] if h_states is not None else None
        h, h2 = self.gru2(h, h2_in)
        gains = self._gains(self.out_proj(h))
        return gains, (h1, h2)

    def step(self, feature_frame: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor):
        if feature_frame.ndim == 1:
            feature_frame = feature_frame.unsqueeze(0)
        h = self.input_norm(self.relu(self.in_proj(feature_frame))).unsqueeze(1)
        h, h1_new = self.gru1(h, h1)
        h, h2_new = self.gru2(h, h2)
        gains = self._gains(self.out_proj(h.squeeze(1)))
        return gains, h1_new, h2_new

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TinySpeechEnhancer()
    print(f"Trainable parameters: {model.get_parameter_count():,}")
    x = torch.randn(2, 100, N_FEATURES)
    g, _ = model(x)
    print("forward:", tuple(g.shape), "range:", float(g.min()), float(g.max()))
    assert g.shape == (2, 100, NUM_BANDS)
    assert float(g.min()) >= G_MIN
    assert float(g.max()) <= 1.0
