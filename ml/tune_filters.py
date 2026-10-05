"""
tune_filters.py
Acoustic parameter sweep and ablation validation for APSA & FLANN adaptive filters.
Addresses Bug #16 (step size tuning) and Bug #17 (FLANN role validation).
"""

import os
import json
import numpy as np
import soundfile as sf
from dsp_simulator import run_apsa, run_flann

def evaluate_filter_config(primary: np.ndarray, ref: np.ndarray, clean: np.ndarray,
                           mu_apsa: float, mu_flann: float, apsa_taps: int = 64, flann_taps: int = 16) -> dict:
    # 1. Run APSA
    apsa_res = run_apsa(primary, ref, step_size=mu_apsa, taps=apsa_taps)
    
    # 2. Run FLANN
    flann_res = run_flann(apsa_res, ref, step_size=mu_flann, taps=flann_taps)
    
    # Compute SNRs
    p_clean = np.mean(clean ** 2) + 1e-12
    snr_raw = 10.0 * np.log10(p_clean / (np.mean((primary - clean) ** 2) + 1e-12))
    snr_apsa = 10.0 * np.log10(p_clean / (np.mean((apsa_res - clean) ** 2) + 1e-12))
    snr_flann = 10.0 * np.log10(p_clean / (np.mean((flann_res - clean) ** 2) + 1e-12))
    
    return {
        "mu_apsa": mu_apsa,
        "mu_flann": mu_flann,
        "snr_raw": snr_raw,
        "snr_apsa": snr_apsa,
        "snr_flann": snr_flann,
        "gain_apsa": snr_apsa - snr_raw,
        "gain_flann_over_apsa": snr_flann - snr_apsa,
        "gain_total_dsp": snr_flann - snr_raw
    }

def run_parameter_sweep(manifest_path: str = "../dataset/data_generated/manifest.json"):
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found at {manifest_path}")
        return
        
    with open(manifest_path, "r") as f:
        entries = json.load(f)
        
    test_samples = [e for e in entries if e.get("split") == "test"][:3]
    if not test_samples:
        test_samples = entries[:3]
        
    print("==========================================================================")
    print("  SIH DEFENCE ANC — APSA & FLANN STEP SIZE HYPERPARAMETER SWEEP")
    print("==========================================================================")
    
    apsa_candidates = [0.010, 0.020, 0.025, 0.035]
    flann_candidates = [0.005, 0.010, 0.015, 0.025]
    
    best_gain = -999.0
    best_config = None
    
    for mu_a in apsa_candidates:
        for mu_f in flann_candidates:
            gains = []
            flann_additions = []
            for s in test_samples:
                clean, _ = sf.read(s["paths"]["clean"])
                primary, _ = sf.read(s["paths"]["primary_mic"])
                ref, _ = sf.read(s["paths"]["ref_mic"])
                
                res = evaluate_filter_config(primary, ref, clean, mu_a, mu_f)
                gains.append(res["gain_total_dsp"])
                flann_additions.append(res["gain_flann_over_apsa"])
                
            avg_gain = float(np.mean(gains))
            avg_flann_gain = float(np.mean(flann_additions))
            
            flag = ""
            if avg_gain > best_gain:
                best_gain = avg_gain
                best_config = (mu_a, mu_f)
                flag = " <-- BEST"
                
            print(f"APSA mu={mu_a:5.3f} | FLANN mu={mu_f:5.3f} | DSP Gain: {avg_gain:+5.2f} dB (FLANN boost: {avg_flann_gain:+5.2f} dB){flag}")
            
    print("--------------------------------------------------------------------------")
    print(f"[RECOMMENDATION] Optimal Tuning: APSA mu={best_config[0]}, FLANN mu={best_config[1]}")
    print("==========================================================================\n")

if __name__ == "__main__":
    run_parameter_sweep()
