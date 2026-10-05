#ifndef AUDIO_CONFIG_H
#define AUDIO_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AUDIO_SAMPLE_RATE          16000
#define AUDIO_FRAME_SIZE           160
#define AUDIO_BITS_PER_SAMPLE      16
#define AUDIO_CHANNELS             2

#define AUDIO_RING_BUFFER_FRAMES   32

#define I2S_MIC_BCLK_PIN           4
#define I2S_MIC_WS_PIN             5
#define I2S_MIC_DIN_PIN            6
#define I2S_SWAP_MIC_CHANNELS      false

// Phase-0 baseline retained until an actual hardware RMS measurement is made.
// The new path no longer uses tanhf() as an implicit compressor.
#define MIC_PREAMP_GAIN            4.0f

#define I2S_SPK_BCLK_PIN           15
#define I2S_SPK_WS_PIN             16
#define I2S_SPK_DOUT_PIN           17

// Legacy APSA/FLANN path is retained for A/B experiments but is OFF by default.
#define ENABLE_ADAPTIVE_PREFILTER  0
#define APSA_FILTER_TAPS           64
#define APSA_PROJECTION_ORDER      2
#define APSA_STEP_SIZE             0.010f
#define APSA_REGULARIZATION        1e-4f
#define FLANN_FILTER_TAPS          16
#define FLANN_EXPANSION_ORDER      3
#define FLANN_TOTAL_WEIGHTS        (FLANN_FILTER_TAPS * FLANN_EXPANSION_ORDER)
#define FLANN_STEP_SIZE            0.025f

// Front-end
#define DC_BLOCK_COEFF             0.995f
#define HPF_CUTOFF_HZ             100.0f
#define IMPULSE_THRESHOLD_DB       20.0f
#define IMPULSE_PEAK_THRESHOLD     0.90f
#define IMPULSE_EMA_ALPHA          0.10f
#define IMPULSE_MAX_ATTENUATION_DB 12.0f
#define IMPULSE_ATTACK_MS           1.0f
#define IMPULSE_RELEASE_MS         40.0f

// STFT / model
#define FFT_SIZE                   256
#define FFT_BINS                   (FFT_SIZE / 2 + 1)
#define FFT_HISTORY_SIZE           (FFT_SIZE - AUDIO_FRAME_SIZE)
#define BARK_NUM_BANDS             22
#define MODEL_INPUT_FEATURES       89
#define GRU_HIDDEN1_DIM            64
#define GRU_PROJ_INPUT_DIM         GRU_HIDDEN1_DIM
#define GRU_HIDDEN2_DIM            48
#define GRU_OUTPUT_DIM             22
#define G_MIN                      0.10f

// Noise-floor protection. These values are starting points and must be tuned from hardware data.
#define DRY_WET_NOISE_FLOOR_DBFS  -55.0f
#define DRY_WET_RELEASE_MS         80.0f
#define DRY_WET_CROSSFADE_MS      100.0f

#ifdef __cplusplus
}
#endif

#endif
