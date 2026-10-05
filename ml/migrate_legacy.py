"""Create a V2 bootstrap checkpoint from the legacy trained model.

Only the shared 44 mic-1 features and recurrent/output weights are copied.
The extra mic-2/ref-valid input columns are initialized to zero so the bootstrap
model initially behaves like the legacy model's mic-1 path, with the new gain floor.
This is a bridge for firmware smoke tests, not a trained V2 result.
"""
from __future__ import annotations
from pathlib import Path
import torch

from model import TinySpeechEnhancer

root = Path(__file__).resolve().parent
legacy = torch.load(root / "checkpoints" / "best_tiny_enhancer.pth", map_location="cpu")
new_model = TinySpeechEnhancer()
new_state = new_model.state_dict()
old = legacy["model_state_dict"]

# New input projection: copy old 44-feature columns into the first 44 columns.
new_state["in_proj.weight"][:, :44] = old["in_proj.0.weight"]
new_state["in_proj.weight"][:, 44:] = 0.0
new_state["in_proj.bias"] = old["in_proj.0.bias"]
new_state["input_norm.weight"] = old["input_norm.weight"]
new_state["input_norm.bias"] = old["input_norm.bias"]
new_state["gru1.weight_ih_l0"] = old["gru1.weight_ih_l0"]
new_state["gru1.weight_hh_l0"] = old["gru1.weight_hh_l0"]
new_state["gru1.bias_ih_l0"] = old["gru1.bias_ih_l0"]
new_state["gru1.bias_hh_l0"] = old["gru1.bias_hh_l0"]
new_state["gru2.weight_ih_l0"] = old["gru2.weight_ih_l0"]
new_state["gru2.weight_hh_l0"] = old["gru2.weight_hh_l0"]
new_state["gru2.bias_ih_l0"] = old["gru2.bias_ih_l0"]
new_state["gru2.bias_hh_l0"] = old["gru2.bias_hh_l0"]
new_state["out_proj.weight"] = old["out_proj.0.weight"]
new_state["out_proj.bias"] = old["out_proj.0.bias"]

new_model.load_state_dict(new_state)
out = root / "checkpoints" / "bootstrap_v2_from_legacy.pth"
torch.save({
    "epoch": 0,
    "model_state_dict": new_model.state_dict(),
    "bootstrap_from": str(root / "checkpoints" / "best_tiny_enhancer.pth"),
    "warning": "UNTRAINED V2 bootstrap. Do not report this as V2 performance.",
}, out)
print(out)
