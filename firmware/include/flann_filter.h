#ifndef FLANN_FILTER_H
#define FLANN_FILTER_H

#include <stdint.h>
#include <stdbool.h>
#include "audio_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float weights[FLANN_TOTAL_WEIGHTS]; // Adaptive weights for non-linear basis
    float x_history[FLANN_FILTER_TAPS]; // Past reference samples
    int   history_idx;                  // Circular buffer index
    float step_size;                    // Adaptation rate
    bool freeze_adaptation;              // Freeze weight updates without clearing state
} flann_filter_t;

/**
 * Initializes the FLANN filter instance.
 */
void flann_init(flann_filter_t *flann, float step_size);

/**
 * Resets weights and history.
 */
void flann_reset(flann_filter_t *flann);

/**
 * Enables/disables weight adaptation while continuing to filter.
 */
void flann_set_adaptation_freeze(flann_filter_t *flann, bool freeze);

/**
 * Processes one sample through FLANN:
 * Takes APSA linear residual error and cancels non-linear acoustic harmonics.
 * 
 * @param flann Pointer to FLANN filter state
 * @param apsa_residual Linear residual e(n) from APSA stage
 * @param ref_sample Current reference noise sample x(n)
 * @return Final enhanced speech sample e_clean(n)
 */
float flann_process_sample(flann_filter_t *flann, float apsa_residual, float ref_sample);

/**
 * Processes a full block of samples.
 */
void flann_process_block(flann_filter_t *flann, const float *apsa_res_in, 
                         const float *ref_in, float *clean_out, int num_samples);

#ifdef __cplusplus
}
#endif

#endif // FLANN_FILTER_H
