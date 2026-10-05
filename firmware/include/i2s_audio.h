#ifndef I2S_AUDIO_H
#define I2S_AUDIO_H

#include <stdint.h>
#include <stdbool.h>
#include "audio_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *primary_buf;   // Mic 1 (Speech + Noise)
    float *ref_buf;       // Mic 2 (Environmental Noise)
    int    capacity;      // Capacity in samples
    int    head;          // Write index
    int    tail;          // Read index
    int    count;         // Current samples available
} audio_ring_buffer_t;

/**
 * Initializes the I2S peripheral for dual INMP441 / SPH0645 microphones.
 * Configured as Stereo 16 kHz: Left = Primary Mic, Right = Reference Mic.
 */
bool i2s_audio_init(void);

/**
 * Reads a block of interleaved stereo samples from I2S DMA,
 * de-interleaves them into primary and reference float arrays [-1.0f, +1.0f].
 * 
 * @param primary_out Buffer for Primary Mic (length = num_samples)
 * @param ref_out Buffer for Reference Mic (length = num_samples)
 * @param num_samples Number of samples to read (e.g. AUDIO_FRAME_SIZE = 160)
 * @return true if read succeeded, false on timeout/error
 */
bool i2s_audio_read_frame(float *primary_out, float *ref_out, int num_samples);

/**
 * Initializes the secondary I2S peripheral (I2S_NUM_1) for audio TX / DAC output.
 * Pins: BCLK=15, WS=16, DOUT=17 (compatible with MAX98357A, PCM5102, tactical headsets).
 */
bool i2s_speaker_init(void);

/**
 * Transmits a frame of enhanced float audio [-1.0f, +1.0f] to the DAC output.
 * Converts to 16-bit signed PCM and writes to I2S_NUM_1 DMA buffer.
 */
bool i2s_audio_write_frame(const float *clean_in, int num_samples);

/**
 * Allocates and initializes the audio ring buffer in ESP32-S3 Octal PSRAM.
 */
audio_ring_buffer_t* audio_ring_buffer_create(int num_frames);

/**
 * Frees the ring buffer memory.
 */
void audio_ring_buffer_destroy(audio_ring_buffer_t *rb);

#ifdef __cplusplus
}
#endif

#endif // I2S_AUDIO_H
