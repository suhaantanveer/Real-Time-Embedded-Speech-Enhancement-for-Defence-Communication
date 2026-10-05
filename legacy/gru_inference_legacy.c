#include "gru_inference.h"
#include "model_weights.h"
#include <string.h>
#include <math.h>

static inline float sigmoidf_fast(float x) {
    if (x > 15.0f) return 1.0f;
    if (x < -15.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static inline float reluf_fast(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

void gru_inference_init(gru_state_t *state) {
    if (!state) return;
    memset(state, 0, sizeof(gru_state_t));
}

void gru_inference_reset(gru_state_t *state) {
    if (!state) return;
    memset(state->h1, 0, sizeof(state->h1));
    memset(state->h2, 0, sizeof(state->h2));
}

void gru_inference_step(gru_state_t *state, const float *features_44, float *gains_22) {
    if (!state || !features_44 || !gains_22) return;
    
    // =========================================================================
    // 1. Input Linear Projection: in_proj (44 -> 64) + ReLU
    // =========================================================================
    float h_proj[GRU_HIDDEN1_DIM];
    for (int i = 0; i < GRU_HIDDEN1_DIM; i++) {
        float sum = in_proj_0_bias[i];
        const float *w_row = &in_proj_0_weight[i * GRU_INPUT_DIM];
        for (int j = 0; j < GRU_INPUT_DIM; j++) {
            sum += w_row[j] * features_44[j];
        }
        h_proj[i] = reluf_fast(sum);
    }
    
    // =========================================================================
    // 2. LayerNorm(64): (x - mean) / sqrt(var + eps) * weight + bias
    // =========================================================================
    float mean = 0.0f;
    for (int i = 0; i < GRU_HIDDEN1_DIM; i++) {
        mean += h_proj[i];
    }
    mean /= (float)GRU_HIDDEN1_DIM;
    
    float var = 0.0f;
    for (int i = 0; i < GRU_HIDDEN1_DIM; i++) {
        float diff = h_proj[i] - mean;
        var += diff * diff;
    }
    var /= (float)GRU_HIDDEN1_DIM;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    
    float h_norm[GRU_HIDDEN1_DIM];
    for (int i = 0; i < GRU_HIDDEN1_DIM; i++) {
        h_norm[i] = (h_proj[i] - mean) * inv_std * input_norm_weight[i] + input_norm_bias[i];
    }
    
    // =========================================================================
    // 3. Causal GRU Layer 1: 64 inputs -> 64 hidden states
    // =========================================================================
    // PyTorch GRU gate ordering: r (reset), z (update), n (new)
    float r1[GRU_HIDDEN1_DIM];
    float z1[GRU_HIDDEN1_DIM];
    float n1[GRU_HIDDEN1_DIM];
    
    const int H1 = GRU_HIDDEN1_DIM;
    
    // Reset gate r and Update gate z
    for (int j = 0; j < H1; j++) {
        // Reset gate: row j in [0, H1)
        float gate_r = gru1_bias_ih_l0[j] + gru1_bias_hh_l0[j];
        const float *w_ih_r = &gru1_weight_ih_l0[j * H1];
        const float *w_hh_r = &gru1_weight_hh_l0[j * H1];
        for (int k = 0; k < H1; k++) {
            gate_r += w_ih_r[k] * h_norm[k] + w_hh_r[k] * state->h1[k];
        }
        r1[j] = sigmoidf_fast(gate_r);
        
        // Update gate: row (j + H1) in [H1, 2*H1)
        int idx_z = j + H1;
        float gate_z = gru1_bias_ih_l0[idx_z] + gru1_bias_hh_l0[idx_z];
        const float *w_ih_z = &gru1_weight_ih_l0[idx_z * H1];
        const float *w_hh_z = &gru1_weight_hh_l0[idx_z * H1];
        for (int k = 0; k < H1; k++) {
            gate_z += w_ih_z[k] * h_norm[k] + w_hh_z[k] * state->h1[k];
        }
        z1[j] = sigmoidf_fast(gate_z);
    }
    
    // New gate n
    for (int j = 0; j < H1; j++) {
        int idx_n = j + 2 * H1;
        float gate_ih = gru1_bias_ih_l0[idx_n];
        const float *w_ih_n = &gru1_weight_ih_l0[idx_n * H1];
        for (int k = 0; k < H1; k++) {
            gate_ih += w_ih_n[k] * h_norm[k];
        }
        
        float gate_hh = gru1_bias_hh_l0[idx_n];
        const float *w_hh_n = &gru1_weight_hh_l0[idx_n * H1];
        for (int k = 0; k < H1; k++) {
            gate_hh += w_hh_n[k] * state->h1[k];
        }
        
        n1[j] = tanhf(gate_ih + r1[j] * gate_hh);
    }
    
    // Hidden state update: h1_new = (1 - z1) * n1 + z1 * h1_old
    float h1_out[GRU_HIDDEN1_DIM];
    for (int j = 0; j < H1; j++) {
        h1_out[j] = (1.0f - z1[j]) * n1[j] + z1[j] * state->h1[j];
        state->h1[j] = h1_out[j];
    }
    
    // =========================================================================
    // 4. Causal GRU Layer 2: 64 inputs -> 48 hidden states
    // =========================================================================
    const int In2 = GRU_HIDDEN1_DIM; // 64
    const int H2  = GRU_HIDDEN2_DIM; // 48
    
    float r2[GRU_HIDDEN2_DIM];
    float z2[GRU_HIDDEN2_DIM];
    float n2[GRU_HIDDEN2_DIM];
    
    for (int j = 0; j < H2; j++) {
        // Reset gate
        float gate_r = gru2_bias_ih_l0[j] + gru2_bias_hh_l0[j];
        const float *w_ih_r = &gru2_weight_ih_l0[j * In2];
        const float *w_hh_r = &gru2_weight_hh_l0[j * H2];
        for (int k = 0; k < In2; k++) {
            gate_r += w_ih_r[k] * h1_out[k];
        }
        for (int k = 0; k < H2; k++) {
            gate_r += w_hh_r[k] * state->h2[k];
        }
        r2[j] = sigmoidf_fast(gate_r);
        
        // Update gate
        int idx_z = j + H2;
        float gate_z = gru2_bias_ih_l0[idx_z] + gru2_bias_hh_l0[idx_z];
        const float *w_ih_z = &gru2_weight_ih_l0[idx_z * In2];
        const float *w_hh_z = &gru2_weight_hh_l0[idx_z * H2];
        for (int k = 0; k < In2; k++) {
            gate_z += w_ih_z[k] * h1_out[k];
        }
        for (int k = 0; k < H2; k++) {
            gate_z += w_hh_z[k] * state->h2[k];
        }
        z2[j] = sigmoidf_fast(gate_z);
    }
    
    for (int j = 0; j < H2; j++) {
        int idx_n = j + 2 * H2;
        float gate_ih = gru2_bias_ih_l0[idx_n];
        const float *w_ih_n = &gru2_weight_ih_l0[idx_n * In2];
        for (int k = 0; k < In2; k++) {
            gate_ih += w_ih_n[k] * h1_out[k];
        }
        
        float gate_hh = gru2_bias_hh_l0[idx_n];
        const float *w_hh_n = &gru2_weight_hh_l0[idx_n * H2];
        for (int k = 0; k < H2; k++) {
            gate_hh += w_hh_n[k] * state->h2[k];
        }
        
        n2[j] = tanhf(gate_ih + r2[j] * gate_hh);
    }
    
    // Hidden state update: h2_new = (1 - z2) * n2 + z2 * h2_old
    float h2_out[GRU_HIDDEN2_DIM];
    for (int j = 0; j < H2; j++) {
        h2_out[j] = (1.0f - z2[j]) * n2[j] + z2[j] * state->h2[j];
        state->h2[j] = h2_out[j];
    }
    
    // =========================================================================
    // 5. Output Projection: out_proj (48 -> 22) + Sigmoid mask
    // =========================================================================
    for (int i = 0; i < GRU_OUTPUT_DIM; i++) {
        float sum = out_proj_0_bias[i];
        const float *w_row = &out_proj_0_weight[i * H2];
        for (int j = 0; j < H2; j++) {
            sum += w_row[j] * h2_out[j];
        }
        gains_22[i] = sigmoidf_fast(sum);
    }
}
