"""
export.py
Exports trained TinySpeechEnhancer for ESP32-S3 Edge Deployment:
1. Pure C header exporter: Exports exact weights as const C arrays for zero-dependency inference on ESP32.
2. TorchScript traced model (.pt).
3. ONNX export (if onnx is installed).
"""

import os
import argparse
import torch
from model import TinySpeechEnhancer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def export_pure_c_weights(model: TinySpeechEnhancer, output_header_path: str = "model_weights.h"):
    """
    Exports neural network weights as plain C const arrays.
    Allows running the forward pass directly on ESP32-S3 with zero external ML dependencies!
    """
    state = model.state_dict()
    
    with open(output_header_path, "w") as f:
        f.write("#ifndef MODEL_WEIGHTS_H\n")
        f.write("#define MODEL_WEIGHTS_H\n\n")
        f.write("// ==============================================================================\n")
        f.write("// SIH Defence ANC — TinySpeechEnhancer FP32 Weights for ESP32-S3\n")
        f.write("// Architecture: 44 -> Dense(64) -> GRU(64) -> GRU(48) -> Dense(22)\n")
        f.write("// ==============================================================================\n\n")
        f.write("#include <stdint.h>\n\n")
        
        for name, tensor in state.items():
            clean_name = name.replace(".", "_")
            np_arr = tensor.cpu().numpy().flatten()
            
            f.write(f"// Layer: {name} (shape: {list(tensor.shape)}, total: {len(np_arr)})\n")
            f.write(f"static const float {clean_name}[{len(np_arr)}] = {{\n")
            
            for i, val in enumerate(np_arr):
                if i % 8 == 0:
                    f.write("    ")
                f.write(f"{val:.7e}f, ")
                if i % 8 == 7:
                    f.write("\n")
            if len(np_arr) % 8 != 0:
                f.write("\n")
            f.write("};\n\n")
            
        f.write("#endif // MODEL_WEIGHTS_H\n")
        
    size_kb = os.path.getsize(output_header_path) / 1024.0
    print(f"[OK] Exported standalone C weights header to: {output_header_path} ({size_kb:.1f} KB)")


class TraceableWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        gains, _ = self.model(x)
        return gains


def export_torchscript(model: TinySpeechEnhancer, output_path: str = "tiny_enhancer.pt"):
    try:
        model.eval()
        wrapper = TraceableWrapper(model)
        dummy_x = torch.randn(1, 100, 22)
        traced = torch.jit.trace(wrapper, dummy_x)
        traced.save(output_path)
        size_kb = os.path.getsize(output_path) / 1024.0
        print(f"[OK] Exported TorchScript model to: {output_path} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"[NOTE] TorchScript trace skipped ({type(e).__name__}).")


def export_onnx(model: TinySpeechEnhancer, output_path: str = "tiny_enhancer.onnx"):
    try:
        model.eval()
        wrapper = TraceableWrapper(model)
        dummy_input = torch.randn(1, 100, 22, dtype=torch.float32)
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['log_energies'],
            output_names=['band_gains']
        )
        size_kb = os.path.getsize(output_path) / 1024.0
        print(f"[OK] Exported ONNX model to: {output_path} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"[NOTE] ONNX exporter skipped.")


def export_all(model_path: str, output_dir: str = "exported"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cpu')
    model = TinySpeechEnhancer().to(device)
    
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[INFO] Loaded trained weights from: {model_path}")
    else:
        print(f"[INFO] Using initialized weights.")
        
    model.eval()
    
    # 1. Export Pure C Weights Header
    c_header_path = os.path.join(output_dir, "model_weights.h")
    export_pure_c_weights(model, c_header_path)
    
    # Copy to firmware/include (anchored to script location, works from any CWD)
    fw_header = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "firmware", "include", "model_weights.h"))
    if os.path.exists(os.path.dirname(fw_header)):
        export_pure_c_weights(model, fw_header)
        
    # 2. Export TorchScript
    ts_path = os.path.join(output_dir, "tiny_enhancer.pt")
    export_torchscript(model, ts_path)
    
    # 3. Export ONNX
    onnx_path = os.path.join(output_dir, "tiny_enhancer.onnx")
    export_onnx(model, onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to C weights header and TorchScript")
    parser.add_argument("--model", type=str, default="checkpoints/best_tiny_enhancer.pth", help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, default="exported", help="Output directory")
    args = parser.parse_args()
    
    export_all(args.model, args.output_dir)
