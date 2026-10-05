#ifndef GRU_INFERENCE_H
#define GRU_INFERENCE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GRU_INPUT_DIM   44  // 22 log band energies + 22 deltas
#define GRU_HIDDEN1_DIM 64  // Layer 1 hidden state dimension
#define GRU_HIDDEN2_DIM 48  // Layer 2 hidden state dimension
#define GRU_OUTPUT_DIM  22  // 22 predicted spectral band gains

typedef struct {
    float h1[GRU_HIDDEN1_DIM]; // Hidden state of GRU Layer 1
    float h2[GRU_HIDDEN2_DIM]; // Hidden state of GRU Layer 2
} gru_state_t;

/**
 * Initializes the GRU inference engine and clears hidden states.
 */
void gru_inference_init(gru_state_t *state);

/**
 * Resets the recurrent hidden states to zero.
 */
void gru_inference_reset(gru_state_t *state);

/**
 * Performs single-frame causal inference:
 * Input:  44 float features (22 log energies + 22 deltas)
 * Output: 22 float spectral gains in [0.0, 1.0]
 * Updates internal hidden states h1 and h2 in-place.
 */
void gru_inference_step(gru_state_t *state, const float *features_44, float *gains_22);

#ifdef __cplusplus
}
#endif

#endif // GRU_INFERENCE_H
