#ifndef FEATURES_ESP32_H
#define FEATURES_ESP32_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "audio_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FFT_SIZE             256
#define FFT_BINS             (FFT_SIZE / 2 + 1) // 129
#define FFT_HISTORY_SIZE     (FFT_SIZE - AUDIO_FRAME_SIZE) // 96
#define BARK_NUM_BANDS       22
#define MODEL_INPUT_FEATURES (BARK_NUM_BANDS * 2) // 44 (22 energies + 22 deltas)

typedef struct {
    // 96-sample history buffer for causal STFT
    float history[FFT_HISTORY_SIZE];
    // 96-sample tail buffer for overlap-add synthesis
    float overlap_tail[FFT_HISTORY_SIZE];
    // Previous frame log energies for first-order delta calculation
    float prev_log_energies[BARK_NUM_BANDS];
    
    // Triangular Bark filterbank: (22 bands, 129 bins)
    float filterbank[BARK_NUM_BANDS][FFT_BINS];
    // Gain expansion matrix: (129 bins, 22 bands)
    float expand_weights[FFT_BINS][BARK_NUM_BANDS];
    
    // Causal analysis and synthesis square-root Hann windows
    float win_ana[FFT_SIZE];
    float win_syn[FFT_SIZE];
    float ola_sum[FFT_SIZE];
    
    // Twiddle tables for 256-point FFT
    float sin_table[FFT_SIZE / 2];
    float cos_table[FFT_SIZE / 2];
    uint8_t bit_rev[FFT_SIZE];
    
    bool is_first_frame;
} esp32_features_t;

/**
 * Initializes the feature extraction and synthesis engine.
 * Computes the Bark filterbank, window coefficients, and FFT twiddles.
 */
void features_esp32_init(esp32_features_t *feat);

/**
 * Resets the streaming state buffers (history, overlap-add, prev energies).
 */
void features_esp32_reset(esp32_features_t *feat);

/**
 * Extracts 44 features (22 log energies + 22 deltas) from 160 new audio samples.
 * Also returns the 129 complex FFT bins for subsequent synthesis.
 */
void features_esp32_extract(esp32_features_t *feat, const float *new_audio_160,
                            float *features_44, float *fft_real_129, float *fft_imag_129);

/**
 * Applies 22 predicted band gains to the 129 complex bins and performs
 * inverse FFT with causal overlap-add reconstruction to yield 160 clean output samples.
 */
void features_esp32_synthesize(esp32_features_t *feat, const float *fft_real_129,
                               const float *fft_imag_129, const float *band_gains_22,
                               float *clean_out_160);

#ifdef __cplusplus
}
#endif

#endif // FEATURES_ESP32_H
