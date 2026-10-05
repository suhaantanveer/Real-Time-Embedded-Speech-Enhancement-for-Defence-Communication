"""
export_test_vectors.py
Generates fixed numerical test vectors from PyTorch to verify bit-level
equivalence between PyTorch and the ESP32 C implementation (Bug #26, #33).
"""

import os
import torch
import numpy as np
from features import FeatureExtractor
from model import TinySpeechEnhancer

def generate_test_vectors(output_dir: str = "../firmware/test"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Deterministic random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 2. Instantiate PyTorch modules
    extractor = FeatureExtractor()
    model = TinySpeechEnhancer()
    
    # Load trained checkpoint if exists
    ckpt_path = "checkpoints/best_tiny_enhancer.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[INFO] Loaded trained weights from {ckpt_path}")
    model.eval()
    
    # 3. Create a synthetic test frame of 160 samples
    # e.g. combination of two sinusoids + white noise
    t = np.linspace(0, 0.01, 160, endpoint=False) # 10 ms @ 16 kHz
    audio_test = (0.5 * np.sin(2 * np.pi * 440 * t) + 
                  0.3 * np.sin(2 * np.pi * 1200 * t) + 
                  0.05 * np.random.randn(160)).astype(np.float32)
                  
    audio_t = torch.from_numpy(audio_test).unsqueeze(0) # (1, 160)
    
    # 4. Run PyTorch feature extraction
    with torch.no_grad():
        log_energies, stft_complex = extractor.extract_features(audio_t)
        
        # Delta calculation for frame 0 is 0
        delta = torch.zeros_like(log_energies)
        features_44 = torch.cat([log_energies, delta], dim=-1) # (1, 1, 44)
        
        # Run PyTorch model forward step
        gains, _ = model(log_energies)
        
        # Reconstruct audio
        clean_out = extractor.apply_gains_and_istft(stft_complex, gains, target_len=160)

    # 5. Save raw binary arrays for C test
    audio_test.tofile(os.path.join(output_dir, "test_audio_in_160.bin"))
    
    features_np = features_44.squeeze(0).squeeze(0).numpy().astype(np.float32)
    features_np.tofile(os.path.join(output_dir, "expected_features_44.bin"))
    
    gains_np = gains.squeeze(0).squeeze(0).numpy().astype(np.float32)
    gains_np.tofile(os.path.join(output_dir, "expected_gains_22.bin"))
    
    stft_real_np = torch.real(stft_complex).squeeze(0).squeeze(-1).numpy().astype(np.float32)
    stft_imag_np = torch.imag(stft_complex).squeeze(0).squeeze(-1).numpy().astype(np.float32)
    stft_real_np.tofile(os.path.join(output_dir, "expected_stft_real_129.bin"))
    stft_imag_np.tofile(os.path.join(output_dir, "expected_stft_imag_129.bin"))
    
    print(f"[OK] Successfully exported test vectors to {output_dir}:")
    print(f"     test_audio_in_160.bin       : {audio_test.shape} float32")
    print(f"     expected_features_44.bin    : {features_np.shape} float32")
    print(f"     expected_gains_22.bin       : {gains_np.shape} float32")
    print(f"     expected_stft_real_129.bin  : {stft_real_np.shape} float32")
    print(f"     expected_stft_imag_129.bin  : {stft_imag_np.shape} float32")

if __name__ == "__main__":
    generate_test_vectors()
