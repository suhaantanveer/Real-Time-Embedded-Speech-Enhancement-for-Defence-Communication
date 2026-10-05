/**
 * Stateful host-side equivalence test for SIH26052 V3.
 *
 * Verifies the actual streaming path over many consecutive frames:
 *   - both-mic FFTs/features, including Mic-2 history
 *   - continuous GRU recurrent state
 *   - gains
 *   - causal streaming synthesis with its 96-sample latency
 *
 * Python's full causal overlap-add reference is used for the output check.
 * The first and final streaming chunks are excluded because they correspond
 * to the left warm-up and right-edge tail of the finite reference sequence.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "../include/audio_config.h"
#include "../include/features_esp32.h"
#include "../include/gru_inference.h"

#define TEST_FRAMES 32
#define LATENCY_SAMPLES FFT_HISTORY_SIZE

static bool load_binary_file(const char *filename, void *dst, size_t expected_bytes) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("[ERROR] Failed to open %s\n", filename);
        return false;
    }
    size_t n = fread(dst, 1, expected_bytes, f);
    fclose(f);
    if (n != expected_bytes) {
        printf("[ERROR] %s: expected %zu bytes, read %zu\n", filename, expected_bytes, n);
        return false;
    }
    return true;
}

static float max_abs_err(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float e = fabsf(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

int main(void) {
    printf("============================================================\n");
    printf(" SIH26052 V3 STATEFUL PYTHON <-> C NUMERICAL EQUIVALENCE\n");
    printf("============================================================\n");

    float mic1[TEST_FRAMES * AUDIO_FRAME_SIZE];
    float mic2[TEST_FRAMES * AUDIO_FRAME_SIZE];
    float py_features[TEST_FRAMES * MODEL_INPUT_FEATURES];
    float py_gains[TEST_FRAMES * BARK_NUM_BANDS];
    float py_stft_r[TEST_FRAMES * FFT_BINS];
    float py_stft_i[TEST_FRAMES * FFT_BINS];
    float py_out[TEST_FRAMES * AUDIO_FRAME_SIZE];

    if (!load_binary_file("test/test_mic1_seq.bin", mic1, sizeof(mic1)) ||
        !load_binary_file("test/test_mic2_seq.bin", mic2, sizeof(mic2)) ||
        !load_binary_file("test/expected_features_89_seq.bin", py_features, sizeof(py_features)) ||
        !load_binary_file("test/expected_gains_22_seq.bin", py_gains, sizeof(py_gains)) ||
        !load_binary_file("test/expected_stft_real_129_seq.bin", py_stft_r, sizeof(py_stft_r)) ||
        !load_binary_file("test/expected_stft_imag_129_seq.bin", py_stft_i, sizeof(py_stft_i)) ||
        !load_binary_file("test/expected_output_seq.bin", py_out, sizeof(py_out))) {
        printf("[FAIL] Missing vectors. Run ml/export_test_vectors.py first.\n");
        return 1;
    }

    esp32_features_t feat;
    features_esp32_init(&feat);
    gru_state_t state;
    gru_inference_init(&state);

    float c_features[TEST_FRAMES * MODEL_INPUT_FEATURES];
    float c_gains[TEST_FRAMES * BARK_NUM_BANDS];
    float c_stft_r[TEST_FRAMES * FFT_BINS];
    float c_stft_i[TEST_FRAMES * FFT_BINS];
    float c_out[TEST_FRAMES * AUDIO_FRAME_SIZE];

    for (int f = 0; f < TEST_FRAMES; ++f) {
        float *cf = &c_features[f * MODEL_INPUT_FEATURES];
        float *cg = &c_gains[f * BARK_NUM_BANDS];
        float *cr = &c_stft_r[f * FFT_BINS];
        float *ci = &c_stft_i[f * FFT_BINS];
        float *co = &c_out[f * AUDIO_FRAME_SIZE];

        features_esp32_extract(
            &feat,
            &mic1[f * AUDIO_FRAME_SIZE],
            &mic2[f * AUDIO_FRAME_SIZE],
            true,
            cf, cr, ci
        );
        gru_inference_step(&state, cf, cg);
        features_esp32_synthesize(&feat, cr, ci, cg, co);
    }

    float feat_err = max_abs_err(c_features, py_features, TEST_FRAMES * MODEL_INPUT_FEATURES);
    float gain_err = max_abs_err(c_gains, py_gains, TEST_FRAMES * BARK_NUM_BANDS);
    float fft_r_err = max_abs_err(c_stft_r, py_stft_r, TEST_FRAMES * FFT_BINS);
    float fft_i_err = max_abs_err(c_stft_i, py_stft_i, TEST_FRAMES * FFT_BINS);

    // C streaming output for frame f corresponds to the Python timeline slice:
    // [f*160-96, f*160+64). Skip first frame (warm-up) and last frame (right tail).
    float out_err = 0.0f;
    int compared_frames = 0;
    for (int f = 1; f < TEST_FRAMES - 1; ++f) {
        int start = f * AUDIO_FRAME_SIZE - LATENCY_SAMPLES;
        const float *c = &c_out[f * AUDIO_FRAME_SIZE];
        const float *p = &py_out[start];
        float e = max_abs_err(c, p, AUDIO_FRAME_SIZE);
        if (e > out_err) out_err = e;
        compared_frames++;
    }

    printf("1. Front-end / feature extraction\n");
    printf("   frames tested            : %d\n", TEST_FRAMES);
    printf("   max FFT real error       : %.9f\n", fft_r_err);
    printf("   max FFT imag error       : %.9f\n", fft_i_err);
    printf("   max 89-feature error     : %.9f\n", feat_err);
    printf("2. GRU inference (stateful)\n");
    printf("   max gain error           : %.9f\n", gain_err);
    printf("3. Streaming ISTFT/output\n");
    printf("   aligned frames compared  : %d\n", compared_frames);
    printf("   max output error         : %.9f\n", out_err);

    bool pass = true;
    if (feat_err >= 1e-4f) pass = false;
    if (gain_err >= 1e-3f) pass = false;
    if (fft_r_err >= 1e-4f || fft_i_err >= 1e-4f) pass = false;
    if (out_err >= 1e-4f) pass = false;

    printf("------------------------------------------------------------\n");
    if (pass) {
        printf("[PASS] V3 stateful Python/C numerical equivalence passed.\n");
        return 0;
    }

    printf("[FAIL] V3 stateful numerical equivalence failed.\n");
    return 1;
}
