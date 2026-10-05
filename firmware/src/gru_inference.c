#include "gru_inference.h"
#include "model_weights.h"
#include <string.h>
#include <math.h>

static inline float sigmoidf_fast(float x) {
    if (x > 15.0f) return 1.0f;
    if (x < -15.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}
static inline float relu_fast(float x) { return x > 0.0f ? x : 0.0f; }

void gru_inference_init(gru_state_t *state) {
    if (state) memset(state, 0, sizeof(*state));
}
void gru_inference_reset(gru_state_t *state) {
    if (state) memset(state, 0, sizeof(*state));
}

void gru_inference_step(gru_state_t *state, const float *features_89, float *gains_22) {
    if (!state || !features_89 || !gains_22) return;
    float proj[GRU_HIDDEN1_DIM];
    for (int i = 0; i < GRU_HIDDEN1_DIM; ++i) {
        float s = in_proj_bias[i];
        for (int j = 0; j < MODEL_INPUT_FEATURES; ++j) s += in_proj_weight[i * MODEL_INPUT_FEATURES + j] * features_89[j];
        proj[i] = relu_fast(s);
    }

    float mean = 0.0f;
    for (int i = 0; i < GRU_HIDDEN1_DIM; ++i) mean += proj[i];
    mean /= (float)GRU_HIDDEN1_DIM;
    float var = 0.0f;
    for (int i = 0; i < GRU_HIDDEN1_DIM; ++i) { float d = proj[i] - mean; var += d * d; }
    var /= (float)GRU_HIDDEN1_DIM;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    float x[GRU_HIDDEN1_DIM];
    for (int i = 0; i < GRU_HIDDEN1_DIM; ++i)
        x[i] = (proj[i] - mean) * inv_std * input_norm_weight[i] + input_norm_bias[i];

    float r1[GRU_HIDDEN1_DIM], z1[GRU_HIDDEN1_DIM], n1[GRU_HIDDEN1_DIM];
    for (int j = 0; j < GRU_HIDDEN1_DIM; ++j) {
        float sr = gru1_bias_ih_l0[j] + gru1_bias_hh_l0[j];
        float sz = gru1_bias_ih_l0[j + GRU_HIDDEN1_DIM] + gru1_bias_hh_l0[j + GRU_HIDDEN1_DIM];
        for (int k = 0; k < GRU_HIDDEN1_DIM; ++k) {
            sr += gru1_weight_ih_l0[j * GRU_PROJ_INPUT_DIM + k] * x[k] + gru1_weight_hh_l0[j * GRU_HIDDEN1_DIM + k] * state->h1[k];
            sz += gru1_weight_ih_l0[(j + GRU_HIDDEN1_DIM) * GRU_PROJ_INPUT_DIM + k] * x[k] + gru1_weight_hh_l0[(j + GRU_HIDDEN1_DIM) * GRU_HIDDEN1_DIM + k] * state->h1[k];
        }
        r1[j] = sigmoidf_fast(sr);
        z1[j] = sigmoidf_fast(sz);
    }
    float h1_new[GRU_HIDDEN1_DIM];
    for (int j = 0; j < GRU_HIDDEN1_DIM; ++j) {
        float si = gru1_bias_ih_l0[j + 2 * GRU_HIDDEN1_DIM];
        float sh = gru1_bias_hh_l0[j + 2 * GRU_HIDDEN1_DIM];
        for (int k = 0; k < GRU_HIDDEN1_DIM; ++k) {
            si += gru1_weight_ih_l0[(j + 2 * GRU_HIDDEN1_DIM) * GRU_PROJ_INPUT_DIM + k] * x[k];
            sh += gru1_weight_hh_l0[(j + 2 * GRU_HIDDEN1_DIM) * GRU_HIDDEN1_DIM + k] * state->h1[k];
        }
        float n = tanhf(si + r1[j] * sh);
        h1_new[j] = (1.0f - z1[j]) * n + z1[j] * state->h1[j];
    }

    float r2[GRU_HIDDEN2_DIM], z2[GRU_HIDDEN2_DIM], n2[GRU_HIDDEN2_DIM];
    for (int j = 0; j < GRU_HIDDEN2_DIM; ++j) {
        float sr = gru2_bias_ih_l0[j] + gru2_bias_hh_l0[j];
        float sz = gru2_bias_ih_l0[j + GRU_HIDDEN2_DIM] + gru2_bias_hh_l0[j + GRU_HIDDEN2_DIM];
        for (int k = 0; k < GRU_HIDDEN1_DIM; ++k) {
            sr += gru2_weight_ih_l0[j * GRU_HIDDEN1_DIM + k] * h1_new[k];
            sz += gru2_weight_ih_l0[(j + GRU_HIDDEN2_DIM) * GRU_HIDDEN1_DIM + k] * h1_new[k];
        }
        for (int k = 0; k < GRU_HIDDEN2_DIM; ++k) {
            sr += gru2_weight_hh_l0[j * GRU_HIDDEN2_DIM + k] * state->h2[k];
            sz += gru2_weight_hh_l0[(j + GRU_HIDDEN2_DIM) * GRU_HIDDEN2_DIM + k] * state->h2[k];
        }
        r2[j] = sigmoidf_fast(sr);
        z2[j] = sigmoidf_fast(sz);
    }
    float h2_new[GRU_HIDDEN2_DIM];
    for (int j = 0; j < GRU_HIDDEN2_DIM; ++j) {
        float si = gru2_bias_ih_l0[j + 2 * GRU_HIDDEN2_DIM];
        float sh = gru2_bias_hh_l0[j + 2 * GRU_HIDDEN2_DIM];
        for (int k = 0; k < GRU_HIDDEN1_DIM; ++k) si += gru2_weight_ih_l0[(j + 2 * GRU_HIDDEN2_DIM) * GRU_HIDDEN1_DIM + k] * h1_new[k];
        for (int k = 0; k < GRU_HIDDEN2_DIM; ++k) sh += gru2_weight_hh_l0[(j + 2 * GRU_HIDDEN2_DIM) * GRU_HIDDEN2_DIM + k] * state->h2[k];
        float n = tanhf(si + r2[j] * sh);
        h2_new[j] = (1.0f - z2[j]) * n + z2[j] * state->h2[j];
    }

    for (int i = 0; i < GRU_OUTPUT_DIM; ++i) {
        float s = out_proj_bias[i];
        for (int j = 0; j < GRU_HIDDEN2_DIM; ++j) s += out_proj_weight[i * GRU_HIDDEN2_DIM + j] * h2_new[j];
        gains_22[i] = G_MIN + (1.0f - G_MIN) * sigmoidf_fast(s);
    }
    memcpy(state->h1, h1_new, sizeof(h1_new));
    memcpy(state->h2, h2_new, sizeof(h2_new));
}
