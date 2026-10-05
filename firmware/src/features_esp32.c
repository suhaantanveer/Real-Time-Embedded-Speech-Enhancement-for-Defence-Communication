#include "features_esp32.h"
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const float BARK_FREQS[BARK_NUM_BANDS] = {
    100.0f, 200.0f, 300.0f, 400.0f, 510.0f, 630.0f, 770.0f, 920.0f, 1080.0f, 1270.0f,
    1480.0f, 1720.0f, 2000.0f, 2320.0f, 2700.0f, 3150.0f, 3700.0f, 4400.0f, 5300.0f, 6350.0f, 7700.0f, 8000.0f
};

static const float MIC2_CALIBRATION_DB[BARK_NUM_BANDS] = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

static uint8_t reverse_bits_8(uint8_t x) {
    uint8_t b = 0;
    for (int i = 0; i < 8; ++i) {
        b = (uint8_t)((b << 1) | (x & 1u));
        x >>= 1;
    }
    return b;
}

static void fft_radix2(float *real, float *imag, const esp32_features_t *feat, bool inverse) {
    for (int i = 0; i < FFT_SIZE; ++i) {
        int j = feat->bit_rev[i];
        if (j > i) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }
    for (int len = 2; len <= FFT_SIZE; len <<= 1) {
        int half = len >> 1;
        float angle = (inverse ? 2.0f : -2.0f) * (float)M_PI / (float)len;
        float wr_step = cosf(angle);
        float wi_step = sinf(angle);
        for (int i = 0; i < FFT_SIZE; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (int j = 0; j < half; ++j) {
                int u = i + j;
                int v = i + j + half;
                float vr = real[v] * wr - imag[v] * wi;
                float vi = real[v] * wi + imag[v] * wr;
                real[v] = real[u] - vr;
                imag[v] = imag[u] - vi;
                real[u] += vr;
                imag[u] += vi;
                float next_wr = wr * wr_step - wi * wi_step;
                wi = wr * wi_step + wi * wr_step;
                wr = next_wr;
            }
        }
    }
    if (inverse) {
        const float inv = 1.0f / (float)FFT_SIZE;
        for (int i = 0; i < FFT_SIZE; ++i) {
            real[i] *= inv;
            imag[i] *= inv;
        }
    }
}

void features_esp32_init(esp32_features_t *feat) {
    if (!feat) return;
    memset(feat, 0, sizeof(*feat));
    for (int i = 0; i < FFT_SIZE; ++i) {
        feat->bit_rev[i] = reverse_bits_8((uint8_t)i);

        float hann =
            0.5f * (1.0f -
            cosf(2.0f * (float)M_PI * (float)i / (float)FFT_SIZE));

        feat->win_ana[i] = sqrtf(hann);
        feat->win_syn[i] = feat->win_ana[i];
    }

    /* Calculate OLA normalization only after the full windows are initialized. */
    for (int i = 0; i < FFT_SIZE; ++i) {
        feat->ola_sum[i] =
            feat->win_ana[i] * feat->win_syn[i];

        if (i < FFT_HISTORY_SIZE) {
            int prev = i + AUDIO_FRAME_SIZE;
            feat->ola_sum[i] +=
                feat->win_ana[prev] * feat->win_syn[prev];
        }

        if (feat->ola_sum[i] < 1e-4f)
            feat->ola_sum[i] = 1.0f;
    }

    float edges[BARK_NUM_BANDS + 1];
    edges[0] = 0.0f;
    for (int b = 0; b < BARK_NUM_BANDS; ++b) edges[b + 1] = BARK_FREQS[b];
    const float bin_hz = ((float)AUDIO_SAMPLE_RATE / 2.0f) / (float)(FFT_BINS - 1);
    for (int b = 0; b < BARK_NUM_BANDS; ++b) {
        float low = edges[b];
        float center = edges[b + 1];
        float high = (b + 2 <= BARK_NUM_BANDS) ? edges[b + 2] : ((float)AUDIO_SAMPLE_RATE / 2.0f);
        float sum = 0.0f;
        for (int k = 0; k < FFT_BINS; ++k) {
            float f = (float)k * bin_hz;
            float w = 0.0f;
            if (f >= low && f < center) w = (f - low) / (center - low + 1e-7f);
            else if (f >= center && f <= high) w = (high - f) / (high - center + 1e-7f);
            feat->filterbank[b][k] = w;
            sum += w;
        }
        if (sum > 1e-7f) {
            float inv = 1.0f / sum;
            for (int k = 0; k < FFT_BINS; ++k) feat->filterbank[b][k] *= inv;
        }
    }
    for (int k = 0; k < FFT_BINS; ++k) {
        float sum = 0.0f;
        for (int b = 0; b < BARK_NUM_BANDS; ++b) sum += feat->filterbank[b][k];
        float inv = sum > 1e-7f ? 1.0f / sum : 0.0f;
        for (int b = 0; b < BARK_NUM_BANDS; ++b) feat->expand_weights[k][b] = feat->filterbank[b][k] * inv;
    }
    feat->is_first_frame = true;
}

void features_esp32_reset(esp32_features_t *feat) {
    if (!feat) return;
    memset(feat->history1, 0, sizeof(feat->history1));
    memset(feat->history2, 0, sizeof(feat->history2));
    memset(feat->overlap_tail, 0, sizeof(feat->overlap_tail));
    memset(feat->prev_log_energies, 0, sizeof(feat->prev_log_energies));
    feat->is_first_frame = true;
}

static void assemble_and_fft(const float *history, const float *new_samples,
                            const float *win, float *real, float *imag,
                            bool use_history) {
    memset(imag, 0, FFT_SIZE * sizeof(float));
    if (use_history) {
        for (int i = 0; i < FFT_HISTORY_SIZE; ++i) real[i] = history[i] * win[i];
    } else {
        for (int i = 0; i < FFT_HISTORY_SIZE; ++i) real[i] = 0.0f;
    }
    for (int i = 0; i < AUDIO_FRAME_SIZE; ++i) real[FFT_HISTORY_SIZE + i] = new_samples[i] * win[FFT_HISTORY_SIZE + i];
}

static void calc_log_bands(const float *real, const float *imag, const float filterbank[BARK_NUM_BANDS][FFT_BINS], float *logs) {
    for (int b = 0; b < BARK_NUM_BANDS; ++b) {
        float e = 0.0f;
        for (int k = 0; k < FFT_BINS; ++k) {
            float p = real[k] * real[k] + imag[k] * imag[k];
            e += p * filterbank[b][k];
        }
        logs[b] = logf(e + 1e-6f);
    }
}

void features_esp32_extract(esp32_features_t *feat,
                            const float *mic1_160,
                            const float *mic2_160,
                            bool ref_valid,
                            float *features_89,
                            float *fft_real_129,
                            float *fft_imag_129) {
    if (!feat || !mic1_160 || !mic2_160 || !features_89 || !fft_real_129 || !fft_imag_129) return;
    float real1[FFT_SIZE], imag1[FFT_SIZE];
    float real2[FFT_SIZE], imag2[FFT_SIZE];
    assemble_and_fft(feat->history1, mic1_160, feat->win_ana, real1, imag1, true);
    assemble_and_fft(feat->history2, mic2_160, feat->win_ana, real2, imag2, true);
    fft_radix2(real1, imag1, feat, false);
    fft_radix2(real2, imag2, feat, false);

    memcpy(fft_real_129, real1, FFT_BINS * sizeof(float));
    memcpy(fft_imag_129, imag1, FFT_BINS * sizeof(float));

    float log1[BARK_NUM_BANDS], log2[BARK_NUM_BANDS];
    calc_log_bands(real1, imag1, feat->filterbank, log1);
    calc_log_bands(real2, imag2, feat->filterbank, log2);

    const float db_to_neper = 0.1f * 2.302585092994046f;
    for (int b = 0; b < BARK_NUM_BANDS; ++b) {
        features_89[b] = log1[b];
        features_89[BARK_NUM_BANDS + b] = feat->is_first_frame ? 0.0f : (log1[b] - feat->prev_log_energies[b]);
        float calibrated_log2 = log2[b] + MIC2_CALIBRATION_DB[b] * db_to_neper;
        features_89[2 * BARK_NUM_BANDS + b] = ref_valid ? log2[b] : 0.0f;
        features_89[3 * BARK_NUM_BANDS + b] = ref_valid ? (log1[b] - calibrated_log2) : 0.0f;
        feat->prev_log_energies[b] = log1[b];
    }
    features_89[4 * BARK_NUM_BANDS] = ref_valid ? 1.0f : 0.0f;
    feat->is_first_frame = false;

    memcpy(feat->history1, &mic1_160[AUDIO_FRAME_SIZE - FFT_HISTORY_SIZE], sizeof(feat->history1));
    memcpy(feat->history2, &mic2_160[AUDIO_FRAME_SIZE - FFT_HISTORY_SIZE], sizeof(feat->history2));
}

void features_esp32_synthesize(esp32_features_t *feat,
                               const float *fft_real_129,
                               const float *fft_imag_129,
                               const float *band_gains_22,
                               float *clean_out_160) {
    if (!feat || !fft_real_129 || !fft_imag_129 || !band_gains_22 || !clean_out_160) return;
    float gains[FFT_BINS];
    for (int k = 0; k < FFT_BINS; ++k) {
        float g = 0.0f;
        for (int b = 0; b < BARK_NUM_BANDS; ++b) g += band_gains_22[b] * feat->expand_weights[k][b];
        gains[k] = g;
    }

    float real[FFT_SIZE], imag[FFT_SIZE];
    real[0] = fft_real_129[0] * gains[0]; imag[0] = 0.0f;
    real[128] = fft_real_129[128] * gains[128]; imag[128] = 0.0f;
    for (int k = 1; k < 128; ++k) {
        float r = fft_real_129[k] * gains[k];
        float im = fft_imag_129[k] * gains[k];
        real[k] = r; imag[k] = im;
        real[FFT_SIZE - k] = r; imag[FFT_SIZE - k] = -im;
    }
    fft_radix2(real, imag, feat, true);
    for (int i = 0; i < FFT_SIZE; ++i) real[i] *= feat->win_syn[i];

    for (int i = 0; i < FFT_HISTORY_SIZE; ++i) clean_out_160[i] = (real[i] + feat->overlap_tail[i]) / feat->ola_sum[i];
    for (int i = FFT_HISTORY_SIZE; i < AUDIO_FRAME_SIZE; ++i) clean_out_160[i] = real[i] / feat->ola_sum[i];
    for (int i = 0; i < FFT_HISTORY_SIZE; ++i) feat->overlap_tail[i] = real[AUDIO_FRAME_SIZE + i];
}
