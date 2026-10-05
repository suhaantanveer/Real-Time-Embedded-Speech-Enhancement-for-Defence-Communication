#ifndef AUDIO_CONFIG_H
#define AUDIO_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================
// Audio Sampling Configuration
// ==============================================================================
#define AUDIO_SAMPLE_RATE          16000    // 16 kHz tactical communications standard
#define AUDIO_FRAME_SIZE           160      // 10 ms frame (160 samples @ 16 kHz)
#define AUDIO_BITS_PER_SAMPLE      16       // 16-bit PCM
#define AUDIO_CHANNELS             2        // Stereo: Left = Primary Mic, Right = Reference Mic

// Ring buffer capacity (in frames) in Octal PSRAM
#define AUDIO_RING_BUFFER_FRAMES   32

// ==============================================================================
// ESP32-S3 I2S Pin Assignments (Adjust to match your breadboard wiring)
// ==============================================================================
// Dual INMP441 / SPH0645 Microphones connect to the SAME BCLK and WS lines.
// Mic 1 (Primary, soldier mouth): L/R pin tied to GND (Left Channel).
// Mic 2 (Reference, ambient noise): L/R pin tied to 3.3V (Right Channel).
#define I2S_MIC_BCLK_PIN           4        // Bit Clock (SCK)
#define I2S_MIC_WS_PIN             5        // Word Select / Left-Right Clock (WS/LRCK)
#define I2S_MIC_DIN_PIN            6        // Serial Data In (SD/DIN)

// Invert Left/Right channel mapping if physical mic breadboard wiring is inverted
#define I2S_SWAP_MIC_CHANNELS      false

// Digital pre-amp gain for INMP441 MEMS microphone (16x = +24 dB)
// Brings conversational voice from raw MEMS level (~0.003) to standard DSP level (~0.05 to 0.4)
#define MIC_PREAMP_GAIN            16.0f

// Optional Audio DAC / Headset Output (e.g. MAX98357A or PCM5102)
#define I2S_SPK_BCLK_PIN           15
#define I2S_SPK_WS_PIN             16
#define I2S_SPK_DOUT_PIN           17

// ==============================================================================
// DSP Filter Parameters
// ==============================================================================
#define APSA_FILTER_TAPS           64       // Number of adaptive FIR taps (32 - 64)
#define APSA_PROJECTION_ORDER      2        // Affine Projection Order P = 2
#define APSA_STEP_SIZE             0.010f   // Convergence step size mu (tuned via hyperparameter sweep)
#define APSA_REGULARIZATION        1e-4f    // Delta to avoid division by zero

#define FLANN_FILTER_TAPS          16       // FLANN input taps
#define FLANN_EXPANSION_ORDER      3        // Chebyshev polynomial order (1, 2, 3)
#define FLANN_TOTAL_WEIGHTS        (FLANN_FILTER_TAPS * FLANN_EXPANSION_ORDER) // 48 weights
#define FLANN_STEP_SIZE            0.025f   // FLANN step size (tuned via hyperparameter sweep)

#ifdef __cplusplus
}
#endif

#endif // AUDIO_CONFIG_H
