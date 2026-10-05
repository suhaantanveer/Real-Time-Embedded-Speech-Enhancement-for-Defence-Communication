#ifndef GRU_INFERENCE_H
#define GRU_INFERENCE_H

#include <stdint.h>
#include <stdbool.h>
#include "audio_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float h1[GRU_HIDDEN1_DIM];
    float h2[GRU_HIDDEN2_DIM];
} gru_state_t;

void gru_inference_init(gru_state_t *state);
void gru_inference_reset(gru_state_t *state);
void gru_inference_step(gru_state_t *state, const float *features_89, float *gains_22);

#ifdef __cplusplus
}
#endif
#endif
