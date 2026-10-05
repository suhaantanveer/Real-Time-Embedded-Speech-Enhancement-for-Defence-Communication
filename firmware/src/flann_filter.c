#include "flann_filter.h"
#include <string.h>
#include <math.h>

static inline float sign_val(float val) {
    if (val > 1e-6f) return 1.0f;
    if (val < -1e-6f) return -1.0f;
    return 0.0f;
}

void flann_init(flann_filter_t *flann, float step_size) {
    if (!flann) return;
    memset(flann, 0, sizeof(flann_filter_t));
    flann->step_size = (step_size > 0.0f) ? step_size : FLANN_STEP_SIZE;
    flann->history_idx = 0;
    flann->freeze_adaptation = false;
}

void flann_reset(flann_filter_t *flann) {
    if (!flann) return;
    memset(flann->weights, 0, sizeof(flann->weights));
    memset(flann->x_history, 0, sizeof(flann->x_history));
    flann->history_idx = 0;
    flann->freeze_adaptation = false;
}

void flann_set_adaptation_freeze(flann_filter_t *flann, bool freeze) {
    if (!flann) return;
    flann->freeze_adaptation = freeze;
}

float flann_process_sample(flann_filter_t *flann, float apsa_residual, float ref_sample) {
    const int K = FLANN_FILTER_TAPS;
    
    // Insert new reference sample into history
    flann->history_idx = (flann->history_idx - 1 + K) % K;
    flann->x_history[flann->history_idx] = ref_sample;
    
    // Functional Link Expansion using Chebyshev orthogonal polynomials
    // T_1(u) = u
    // T_2(u) = 2*u^2 - 1
    // T_3(u) = 4*u^3 - 3*u
    float phi[FLANN_TOTAL_WEIGHTS];
    float norm_sq = 1e-4f;
    
    int phi_idx = 0;
    for (int k = 0; k < K; k++) {
        float u = flann->x_history[(flann->history_idx + k) % K];
        
        float t1 = u;
        float t2 = 2.0f * u * u - 1.0f;
        float t3 = 4.0f * u * u * u - 3.0f * u;
        
        phi[phi_idx++] = t1;
        phi[phi_idx++] = t2;
        phi[phi_idx++] = t3;
        
        norm_sq += (t1 * t1) + (t2 * t2) + (t3 * t3);
    }
    
    // Estimate non-linear noise residual
    float y_nl = 0.0f;
    for (int i = 0; i < FLANN_TOTAL_WEIGHTS; i++) {
        y_nl += flann->weights[i] * phi[i];
    }
    
    // Final error after non-linear cancellation
    float e_final = apsa_residual - y_nl;
    
    // Normalized Sign-LMS update
    float sgn_e = sign_val(e_final);
    float step = (flann->step_size * sgn_e) / norm_sq;
    
    for (int i = 0; i < FLANN_TOTAL_WEIGHTS; i++) {
        flann->weights[i] += step * phi[i];
    }
    
    return e_final;
}

void flann_process_block(flann_filter_t *flann, const float *apsa_res_in, 
                         const float *ref_in, float *clean_out, int num_samples) {
    if (!flann || !apsa_res_in || !ref_in || !clean_out) return;
    for (int i = 0; i < num_samples; i++) {
        clean_out[i] = flann_process_sample(flann, apsa_res_in[i], ref_in[i]);
    }
}
