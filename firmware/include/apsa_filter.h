#ifndef APSA_FILTER_H
#define APSA_FILTER_H

#include <stdint.h>
#include <stdbool.h>
#include "audio_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float weights[APSA_FILTER_TAPS];                   // Adaptive filter coefficients w(n)
    float x_buffer[APSA_FILTER_TAPS + APSA_PROJECTION_ORDER]; // Circular reference buffer
    int   buffer_idx;                                  // Ring buffer pointer
    float d_past[APSA_PROJECTION_ORDER];               // Past desired samples
    float step_size;                                   // Adaptation step size (mu)
    float regularization;                              // Delta to avoid singularity
    bool  freeze_adaptation;                           // VAD speech-protection freeze
} apsa_filter_t;

/**
 * Initializes the APSA adaptive filter instance.
 */
void apsa_init(apsa_filter_t *filter, float step_size, float reg);

/**
 * Resets the filter coefficients and history buffers.
 */
void apsa_reset(apsa_filter_t *filter);

/**
 * Processes a single audio sample through the APSA filter.
 * 
 * @param filter Pointer to APSA filter state
 * @param d_sample Primary Mic sample (Desired speech + noise) in range [-1.0f, +1.0f]
 * @param x_sample Reference Mic sample (Environmental noise) in range [-1.0f, +1.0f]
 * @return Enhanced error sample e(n) = d(n) - y(n)
 */
float apsa_process_sample(apsa_filter_t *filter, float d_sample, float x_sample);

/**
 * Processes an entire block of audio frames (e.g. 160 samples / 10 ms).
 */
void apsa_process_block(apsa_filter_t *filter, const float *d_in, const float *x_in, 
                        float *e_out, int num_samples);

/**
 * Controls whether coefficient updates are active or frozen (for speech leakage protection).
 */
void apsa_set_adaptation_freeze(apsa_filter_t *filter, bool freeze);

#ifdef __cplusplus
}
#endif

#endif // APSA_FILTER_H
