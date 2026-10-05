#ifndef FRONT_END_H
#define FRONT_END_H

#include <stdbool.h>
#include "audio_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float dc_prev_x[2];
    float dc_prev_y[2];
    float hp_x1[2];
    float hp_x2[2];
    float hp_y1[2];
    float hp_y2[2];
    float hp_b0;
    float hp_b1;
    float hp_b2;
    float hp_a1;
    float hp_a2;
    float energy_ema;
    bool energy_ema_initialized;
    float guard_gain;
    float guard_target;
    bool last_impulse_trigger;
    uint32_t impulse_frames;
} front_end_state_t;

void front_end_init(front_end_state_t *state);
void front_end_reset(front_end_state_t *state);
void front_end_process_frame(front_end_state_t *state,
                             const float *mic1_in,
                             const float *mic2_in,
                             float *mic1_out,
                             float *mic2_out,
                             int n);

#ifdef __cplusplus
}
#endif
#endif
