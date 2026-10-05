#ifndef FEATURES_ESP32_H
#define FEATURES_ESP32_H

#include <stdint.h>
#include <stdbool.h>
#include "audio_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float history1[FFT_HISTORY_SIZE];
    float history2[FFT_HISTORY_SIZE];
    float overlap_tail[FFT_HISTORY_SIZE];
    float prev_log_energies[BARK_NUM_BANDS];
    float filterbank[BARK_NUM_BANDS][FFT_BINS];
    float expand_weights[FFT_BINS][BARK_NUM_BANDS];
    float win_ana[FFT_SIZE];
    float win_syn[FFT_SIZE];
    float ola_sum[FFT_SIZE];
    float sin_table[FFT_SIZE / 2];
    float cos_table[FFT_SIZE / 2];
    uint8_t bit_rev[FFT_SIZE];
    bool is_first_frame;
} esp32_features_t;

void features_esp32_init(esp32_features_t *feat);
void features_esp32_reset(esp32_features_t *feat);

// Feature layout: 22 mic1 logs + 22 mic1 deltas + 22 mic2 logs +
// 22 calibrated mic1-minus-mic2 log differences + ref_valid.
void features_esp32_extract(esp32_features_t *feat,
                            const float *mic1_160,
                            const float *mic2_160,
                            bool ref_valid,
                            float *features_89,
                            float *fft_real_129,
                            float *fft_imag_129);

void features_esp32_synthesize(esp32_features_t *feat,
                               const float *fft_real_129,
                               const float *fft_imag_129,
                               const float *band_gains_22,
                               float *clean_out_160);

#ifdef __cplusplus
}
#endif
#endif
