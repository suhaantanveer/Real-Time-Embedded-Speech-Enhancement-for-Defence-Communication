#include "features_esp32.h"
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 22 Bark frequency band cutoff points (Hz) matching features.py exactly
static const float BARK_FREQS[BARK_NUM_BANDS] = {
    100.0f, 200.0f, 300.0f, 400.0f, 510.0f, 630.0f, 770.0f, 920.0f, 1080.0f, 1270.0f,
    1480.0f, 1720.0f, 2000.0f, 2320.0f, 2700.0f, 3150.0f, 3700.0f, 4400.0f, 5300.0f, 6350.0f, 7700.0f, 8000.0f
};

static uint8_t reverse_bits_8(uint8_t x) {
    uint8_t b = 0;
    for (int i = 0; i < 8; i++) {
        b = (b << 1) | (x & 1);
        x >>= 1;
    }
    return b;
}

static void fft_radix2(float *real, float *imag, const esp32_features_t *feat, bool inverse) {
    // 1. Bit-reversal permutation
    for (int i = 0; i < FFT_SIZE; i++) {
        int j = feat->bit_rev[i];
        if (j > i) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }
    
    // 2. Cooley-Tukey butterfly stages (log2(256) = 8 stages)
    for (int len = 2; len <= FFT_SIZE; len <<= 1) {
        int half_len = len >> 1;
        float angle = (inverse ? 2.0f : -2.0f) * (float)M_PI / (float)len;
        float w_step_r = cosf(angle);
        float w_step_i = sinf(angle);
        
        for (int i = 0; i < FFT_SIZE; i += len) {
            float wr = 1.0f;
            float wi = 0.0f;
            for (int j = 0; j < half_len; j++) {
                int u_idx = i + j;
                int v_idx = i + j + half_len;
                
                float vr = real[v_idx] * wr - imag[v_idx] * wi;
                float vi = real[v_idx] * wi + imag[v_idx] * wr;
                
                real[v_idx] = real[u_idx] - vr;
                imag[v_idx] = imag[u_idx] - vi;
                real[u_idx] = real[u_idx] + vr;
                imag[u_idx] = imag[u_idx] + vi;
                
                float next_wr = wr * w_step_r - wi * w_step_i;
                wi = wr * w_step_i + wi * w_step_r;
                wr = next_wr;
            }
        }
    }
    
    // 3. Normalization for inverse FFT
    if (inverse) {
        float inv_n = 1.0f / (float)FFT_SIZE;
        for (int i = 0; i < FFT_SIZE; i++) {
            real[i] *= inv_n;
            imag[i] *= inv_n;
        }
    }
}

void features_esp32_init(esp32_features_t *feat) {
    if (!feat) return;
    memset(feat, 0, sizeof(esp32_features_t));
    
    // 1. Bit-reversal and window tables
    for (int i = 0; i < FFT_SIZE; i++) {
        feat->bit_rev[i] = reverse_bits_8((uint8_t)i);
        
        // Causal square-root Hann window matching PyTorch torch.hann_window(256, periodic=True)
        float hann = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)FFT_SIZE));
        feat->win_ana[i] = sqrtf(hann);
        feat->win_syn[i] = feat->win_ana[i];
    }
    
    // Overlap-add window sum for 160-sample hop with 256-sample window
    // Overlap region is the first 96 samples (overlaps with previous tail)
    for (int i = 0; i < FFT_SIZE; i++) {
        feat->ola_sum[i] = feat->win_ana[i] * feat->win_syn[i];
        if (i < FFT_HISTORY_SIZE) {
            int prev_idx = i + AUDIO_FRAME_SIZE;
            feat->ola_sum[i] += feat->win_ana[prev_idx] * feat->win_syn[prev_idx];
        }
        if (feat->ola_sum[i] < 1e-4f) {
            feat->ola_sum[i] = 1.0f; // Guard
        }
    }
    
    // 2. Construct 22-band Bark filterbank
    float edges[BARK_NUM_BANDS + 1];
    edges[0] = 0.0f;
    for (int b = 0; b < BARK_NUM_BANDS; b++) {
        edges[b + 1] = BARK_FREQS[b];
    }
    
    float bin_hz = ((float)AUDIO_SAMPLE_RATE / 2.0f) / (float)(FFT_BINS - 1); // 62.5 Hz per bin
    
    for (int b = 0; b < BARK_NUM_BANDS; b++) {
        float f_low = edges[b];
        float f_center = edges[b + 1];
        float f_high = (b + 2 <= BARK_NUM_BANDS) ? edges[b + 2] : ((float)AUDIO_SAMPLE_RATE / 2.0f);
        
        float band_sum = 0.0f;
        for (int k = 0; k < FFT_BINS; k++) {
            float f = (float)k * bin_hz;
            float w = 0.0f;
            if (f >= f_low && f < f_center) {
                w = (f - f_low) / (f_center - f_low + 1e-7f);
            } else if (f >= f_center && f <= f_high) {
                w = (f_high - f) / (f_high - f_center + 1e-7f);
            }
            feat->filterbank[b][k] = w;
            band_sum += w;
        }
        // Normalize filterbank band
        if (band_sum > 1e-7f) {
            float inv_sum = 1.0f / band_sum;
            for (int k = 0; k < FFT_BINS; k++) {
                feat->filterbank[b][k] *= inv_sum;
            }
        }
    }
    
    // 3. Compute gain expansion weights matrix (transpose normalized)
    for (int k = 0; k < FFT_BINS; k++) {
        float bin_sum = 0.0f;
        for (int b = 0; b < BARK_NUM_BANDS; b++) {
            bin_sum += feat->filterbank[b][k];
        }
        float inv_bin_sum = (bin_sum > 1e-7f) ? (1.0f / bin_sum) : 0.0f;
        for (int b = 0; b < BARK_NUM_BANDS; b++) {
            feat->expand_weights[k][b] = feat->filterbank[b][k] * inv_bin_sum;
        }
    }
    
    feat->is_first_frame = true;
}

void features_esp32_reset(esp32_features_t *feat) {
    if (!feat) return;
    memset(feat->history, 0, sizeof(feat->history));
    memset(feat->overlap_tail, 0, sizeof(feat->overlap_tail));
    memset(feat->prev_log_energies, 0, sizeof(feat->prev_log_energies));
    feat->is_first_frame = true;
}

void features_esp32_extract(esp32_features_t *feat, const float *new_audio_160,
                            float *features_44, float *fft_real_129, float *fft_imag_129) {
    if (!feat || !new_audio_160 || !features_44 || !fft_real_129 || !fft_imag_129) return;
    
    float time_vec[FFT_SIZE];
    float imag_vec[FFT_SIZE];
    memset(imag_vec, 0, sizeof(imag_vec));
    
    // 1. Causal 256-sample window assembly: 96 past samples + 160 new samples
    for (int i = 0; i < FFT_HISTORY_SIZE; i++) {
        time_vec[i] = feat->history[i] * feat->win_ana[i];
    }
    for (int i = 0; i < AUDIO_FRAME_SIZE; i++) {
        int idx = FFT_HISTORY_SIZE + i;
        time_vec[idx] = new_audio_160[i] * feat->win_ana[idx];
    }
    
    // Update history with the tail 96 samples of the current frame for next time
    // new_audio_160 has 160 samples. Tail 96 starts at index 160 - 96 = 64
    memcpy(feat->history, &new_audio_160[AUDIO_FRAME_SIZE - FFT_HISTORY_SIZE], sizeof(feat->history));
    
    // 2. Perform 256-point Forward Real FFT
    fft_radix2(time_vec, imag_vec, feat, false);
    
    // Copy first 129 bins (positive frequencies)
    for (int k = 0; k < FFT_BINS; k++) {
        fft_real_129[k] = time_vec[k];
        fft_imag_129[k] = imag_vec[k];
    }
    
    // 3. Compute Power Spectrum |X|^2 and apply Bark filterbank
    float current_log_energies[BARK_NUM_BANDS];
    for (int b = 0; b < BARK_NUM_BANDS; b++) {
        float band_energy = 0.0f;
        for (int k = 0; k < FFT_BINS; k++) {
            float pwr = (fft_real_129[k] * fft_real_129[k]) + (fft_imag_129[k] * fft_imag_129[k]);
            band_energy += pwr * feat->filterbank[b][k];
        }
        current_log_energies[b] = logf(band_energy + 1e-6f);
    }
    
    // 4. Compute Features: 22 log energies + 22 deltas
    for (int b = 0; b < BARK_NUM_BANDS; b++) {
        features_44[b] = current_log_energies[b];
        if (feat->is_first_frame) {
            features_44[BARK_NUM_BANDS + b] = 0.0f; // Zero delta on first frame
        } else {
            features_44[BARK_NUM_BANDS + b] = current_log_energies[b] - feat->prev_log_energies[b];
        }
        feat->prev_log_energies[b] = current_log_energies[b];
    }
    
    feat->is_first_frame = false;
}

void features_esp32_synthesize(esp32_features_t *feat, const float *fft_real_129,
                               const float *fft_imag_129, const float *band_gains_22,
                               float *clean_out_160) {
    if (!feat || !fft_real_129 || !fft_imag_129 || !band_gains_22 || !clean_out_160) return;
    
    // 1. Expand 22 predicted band gains to 129 FFT bins
    float bin_gains[FFT_BINS];
    for (int k = 0; k < FFT_BINS; k++) {
        float g = 0.0f;
        for (int b = 0; b < BARK_NUM_BANDS; b++) {
            g += band_gains_22[b] * feat->expand_weights[k][b];
        }
        bin_gains[k] = g;
    }
    
    // 2. Multiply complex spectrum by gains and construct 256-point Hermitian symmetric spectrum
    float time_vec[FFT_SIZE];
    float imag_vec[FFT_SIZE];
    
    // DC and Nyquist are pure real
    time_vec[0] = fft_real_129[0] * bin_gains[0];
    imag_vec[0] = 0.0f;
    time_vec[128] = fft_real_129[128] * bin_gains[128];
    imag_vec[128] = 0.0f;
    
    for (int k = 1; k < 128; k++) {
        float r = fft_real_129[k] * bin_gains[k];
        float i = fft_imag_129[k] * bin_gains[k];
        
        time_vec[k] = r;
        imag_vec[k] = i;
        
        // Hermitian conjugate for negative frequencies: X[N - k] = conj(X[k])
        time_vec[FFT_SIZE - k] =  r;
        imag_vec[FFT_SIZE - k] = -i;
    }
    
    // 3. Perform 256-point Inverse FFT
    fft_radix2(time_vec, imag_vec, feat, true);
    
    // 4. Multiply by synthesis window
    for (int i = 0; i < FFT_SIZE; i++) {
        time_vec[i] *= feat->win_syn[i];
    }
    
    // 5. Overlap-Add reconstruction
    // First 96 samples overlap with the tail from the previous frame
    for (int i = 0; i < FFT_HISTORY_SIZE; i++) {
        float val = (time_vec[i] + feat->overlap_tail[i]) / feat->ola_sum[i];
        clean_out_160[i] = val;
    }
    // Remaining 64 samples of the 160-sample frame
    for (int i = FFT_HISTORY_SIZE; i < AUDIO_FRAME_SIZE; i++) {
        float val = time_vec[i] / feat->ola_sum[i];
        clean_out_160[i] = val;
    }
    
    // Save tail 96 samples (from index 160 to 255) for the next frame
    for (int i = 0; i < FFT_HISTORY_SIZE; i++) {
        feat->overlap_tail[i] = time_vec[AUDIO_FRAME_SIZE + i];
    }
}
