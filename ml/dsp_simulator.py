"""
dsp_simulator.py
Python port of the ESP32 APSA + FLANN adaptive filters.

PURPOSE: Generate the correct GRU training input.
  The ML model must be trained on APSA→FLANN residual, NOT raw primary mic.
  This matches exactly what main.cpp feeds into the (future) GRU stage.

All constants are intentionally mirrored from firmware/include/audio_config.h.
If you change a constant there, change it here too (see Bug #26 checklist).

PERFORMANCE NOTE:
  Pure Python sample loop is ~0.5–2 s per 3-second clip. For small datasets
  (<200 samples) this is acceptable. For large training runs, precompute
  residuals once with precompute_residuals.py (future work).
"""

import numpy as np

# ============================================================
# MUST MATCH firmware/include/audio_config.h EXACTLY
# ============================================================
APSA_FILTER_TAPS      = 64
APSA_PROJECTION_ORDER = 2
APSA_STEP_SIZE        = 0.010
APSA_REGULARIZATION   = 1e-4

FLANN_FILTER_TAPS     = 16
FLANN_EXPANSION_ORDER = 3
FLANN_TOTAL_WEIGHTS   = FLANN_FILTER_TAPS * FLANN_EXPANSION_ORDER  # 48
FLANN_STEP_SIZE       = 0.025


def _sign(val: float) -> float:
    """Matches sign_val() in apsa_filter.c / flann_filter.c."""
    if val > 1e-6:
        return 1.0
    if val < -1e-6:
        return -1.0
    return 0.0


def run_apsa(primary: np.ndarray, reference: np.ndarray,
             step_size: float = APSA_STEP_SIZE,
             regularization: float = APSA_REGULARIZATION,
             taps: int = APSA_FILTER_TAPS,
             proj_order: int = APSA_PROJECTION_ORDER) -> np.ndarray:
    """
    Affine Projection Sign Algorithm adaptive filter.
    Direct Python translation of apsa_filter.c :: apsa_process_sample().

    Args:
        primary  : d[n] — primary mic (speech + acoustic-path noise)
        reference: x[n] — reference mic (ambient noise + speech leakage)
    Returns:
        residual : e[n] = d[n] - ŷ[n]  (linear noise cancelled)
    """
    N = min(len(primary), len(reference))
    total_buf = taps + proj_order

    weights  = np.zeros(taps,      dtype=np.float64)
    x_buf    = np.zeros(total_buf, dtype=np.float64)
    d_past   = np.zeros(proj_order, dtype=np.float64)
    buf_idx  = 0
    residual = np.zeros(N, dtype=np.float32)

    frame_size = 160
    for n in range(N):
        # Frame-level VAD double-talk check every 160 samples (matches main.cpp)
        if n % frame_size == 0:
            frame_end = min(n + frame_size, N)
            p_pri = float(np.mean(primary[n:frame_end] ** 2))
            p_ref = float(np.mean(reference[n:frame_end] ** 2))
            ratio = p_pri / (p_ref + 1e-6)
            freeze = (ratio > 1.8) and (p_pri > 1e-4)
            
        # --- Insert new reference sample (mirrored ring buffer) ---
        buf_idx = (buf_idx - 1 + total_buf) % total_buf
        x_buf[buf_idx] = reference[n]

        # --- Shift desired-signal history ---
        d_past[1] = d_past[0]
        d_past[0] = primary[n]

        # --- Build two tap vectors via vectorised numpy indexing ---
        idx0 = buf_idx
        idx1 = (buf_idx + 1) % total_buf

        # Causal tap indices for x0 and x1
        indices0 = np.arange(taps)
        indices1 = np.arange(1, taps + 1)
        x0 = x_buf[(idx0 + indices0) % total_buf]
        x1 = x_buf[(idx0 + indices1) % total_buf]

        y0 = float(np.dot(weights, x0))
        y1 = float(np.dot(weights, x1))

        r00 = regularization + float(np.dot(x0, x0))
        r11 = regularization + float(np.dot(x1, x1))
        r01 = float(np.dot(x0, x1))

        e0 = d_past[0] - y0
        e1 = d_past[1] - y1

        det = r00 * r11 - r01 * r01
        if det > 1e-10 and not freeze:
            inv_det = 1.0 / det
            inv_r00 =  r11 * inv_det
            inv_r11 =  r00 * inv_det
            inv_r01 = -r01 * inv_det

            s0 = _sign(e0)
            s1 = _sign(e1)

            v0 = inv_r00 * s0 + inv_r01 * s1
            v1 = inv_r01 * s0 + inv_r11 * s1

            weights += step_size * (v0 * x0 + v1 * x1)

        residual[n] = float(e0)

    return residual


def run_flann(apsa_residual: np.ndarray, reference: np.ndarray,
              step_size: float = FLANN_STEP_SIZE,
              taps: int = FLANN_FILTER_TAPS,
              order: int = FLANN_EXPANSION_ORDER) -> np.ndarray:
    """
    Functional Link Adaptive Neural Network filter.
    Direct Python translation of flann_filter.c :: flann_process_sample().

    Chebyshev expansion (interleaved per tap, matching C layout):
      phi = [T1(x[n]), T2(x[n]), T3(x[n]),
             T1(x[n-1]), T2(x[n-1]), T3(x[n-1]), ...]

    Args:
        apsa_residual: e[n] from APSA (linear residual)
        reference    : x[n] reference mic signal
    Returns:
        residual: e_final[n] after non-linear cancellation
    """
    N = min(len(apsa_residual), len(reference))
    total_w = taps * order

    weights   = np.zeros(total_w, dtype=np.float64)
    x_history = np.zeros(taps,    dtype=np.float64)
    hist_idx  = 0
    residual  = np.zeros(N, dtype=np.float32)

    frame_size = 160
    for n in range(N):
        # Frame-level VAD double-talk check every 160 samples
        if n % frame_size == 0:
            frame_end = min(n + frame_size, N)
            p_res = float(np.mean(apsa_residual[n:frame_end] ** 2))
            p_ref = float(np.mean(reference[n:frame_end] ** 2))
            ratio = p_res / (p_ref + 1e-6)
            freeze = (ratio > 1.8) and (p_res > 1e-4)
            
        # Insert reference sample
        hist_idx = (hist_idx - 1 + taps) % taps
        x_history[hist_idx] = reference[n]

        # Build tapped delay line
        tap_indices = (hist_idx + np.arange(taps)) % taps
        u = x_history[tap_indices]   # shape (K,)

        # Chebyshev expansion interleaved: [T1,T2,T3, T1,T2,T3, ...]
        t1 = u
        t2 = 2.0 * u * u - 1.0
        t3 = 4.0 * u * u * u - 3.0 * u

        # Stack columns then flatten → matches C phi_idx++ pattern
        phi = np.column_stack([t1, t2, t3]).flatten()  # shape (K*3,)

        norm_sq = max(float(np.dot(phi, phi)), 1e-4)

        y_nl    = float(np.dot(weights, phi))
        e_final = float(apsa_residual[n]) - y_nl

        sgn_e = _sign(e_final)
        if not freeze:
            weights += (step_size * sgn_e / norm_sq) * phi

        residual[n] = float(e_final)

    return residual


def simulate_dsp_pipeline(primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Full APSA → FLANN pipeline.
    Mirrors dsp_processing_task() in firmware/src/main.cpp exactly.

    Args:
        primary  : primary mic audio  (speech + acoustic-path noise)
        reference: reference mic audio (ambient noise + speech leakage)
    Returns:
        dsp_out : GRU-ready residual signal
    """
    primary   = primary.astype(np.float64)
    reference = reference.astype(np.float64)
    apsa_out  = run_apsa(primary, reference)
    flann_out = run_flann(apsa_out.astype(np.float64), reference)
    return flann_out.astype(np.float32)


# ----------------------------------------------------------------
# Quick self-test
# ----------------------------------------------------------------
if __name__ == "__main__":
    import time
    sr   = 16000
    dur  = 3.0
    N    = int(sr * dur)

    primary   = np.random.randn(N).astype(np.float32) * 0.3
    reference = np.random.randn(N).astype(np.float32) * 0.5

    t0  = time.perf_counter()
    out = simulate_dsp_pipeline(primary, reference)
    t1  = time.perf_counter()

    print(f"[OK] DSP pipeline test passed in {(t1-t0)*1000:.1f} ms")
    print(f"     Input RMS : {float(np.sqrt(np.mean(primary**2))):.4f}")
    print(f"     Output RMS: {float(np.sqrt(np.mean(out**2))):.4f}")
    assert out.shape == primary.shape, "Shape mismatch!"
    print("[PASS] dsp_simulator.py self-test complete.")
