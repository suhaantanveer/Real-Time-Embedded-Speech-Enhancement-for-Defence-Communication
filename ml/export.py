"""Export the redesigned 89-feature TinySpeechEnhancer to C and TorchScript."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from config import G_MIN, HIDDEN1, HIDDEN2, N_FEATURES, NUM_BANDS
from model import TinySpeechEnhancer

SCRIPT_DIR = Path(__file__).resolve().parent


def export_pure_c_weights(model: TinySpeechEnhancer, output_header_path: str | os.PathLike) -> None:
    state = model.state_dict()
    output_header_path = Path(output_header_path)
    output_header_path.parent.mkdir(parents=True, exist_ok=True)
    with output_header_path.open("w", encoding="utf-8") as f:
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
        f.write("// SIH26052 redesigned Mic-1-primary TinySpeechEnhancer.\n")
        f.write("// Generate this file from a TRAINED V2 checkpoint before demo/final use.\n")
        f.write(f"#define MODEL_N_FEATURES {N_FEATURES}\n#define MODEL_HIDDEN1 {HIDDEN1}\n")
        f.write(f"#define MODEL_HIDDEN2 {HIDDEN2}\n#define MODEL_N_BANDS {NUM_BANDS}\n")
        f.write(f"#define MODEL_G_MIN {G_MIN:.8f}f\n\n#include <stdint.h>\n\n")
        for name, tensor in state.items():
            clean = name.replace(".", "_")
            arr = tensor.detach().cpu().numpy().reshape(-1)
            f.write(f"// {name} shape={list(tensor.shape)}\n")
            f.write(f"static const float {clean}[{len(arr)}] = {{\n")
            for i, value in enumerate(arr):
                if i % 8 == 0:
                    f.write("    ")
                f.write(f"{float(value):.8e}f, ")
                if i % 8 == 7:
                    f.write("\n")
            if len(arr) % 8:
                f.write("\n")
            f.write("};\n\n")
        f.write("#endif\n")


class TraceableWrapper(torch.nn.Module):
    def __init__(self, model: TinySpeechEnhancer):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gains, _ = self.model(x)
        return gains


def load_model(checkpoint: str | os.PathLike | None) -> TinySpeechEnhancer:
    model = TinySpeechEnhancer().cpu().eval()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
    return model


def export_all(checkpoint: str, output_dir: str = "exported") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint)
    export_pure_c_weights(model, out / "model_weights.h")
    fw_header = SCRIPT_DIR.parent / "firmware" / "include" / "model_weights.h"
    export_pure_c_weights(model, fw_header)
    wrapper = TraceableWrapper(model)
    dummy = torch.randn(1, 100, N_FEATURES)
    traced = torch.jit.trace(wrapper, dummy)
    traced.save(str(out / "tiny_enhancer_v2.pt"))
    print(f"Exported {model.get_parameter_count():,} parameters")
    print(f"C header: {fw_header}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", default=str(SCRIPT_DIR / "exported"))
    args = p.parse_args()
    export_all(args.model, args.output_dir)
