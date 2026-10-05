#include "apsa_filter.h"
#include <string.h>
#include <math.h>

static inline float sign_val(float val) {
    if (val > 1e-6f) return 1.0f;
    if (val < -1e-6f) return -1.0f;
    return 0.0f;
}

void apsa_init(apsa_filter_t *filter, float step_size, float reg) {
    if (!filter) return;
    memset(filter, 0, sizeof(apsa_filter_t));
    filter->step_size = (step_size > 0.0f) ? step_size : APSA_STEP_SIZE;
    filter->regularization = (reg > 0.0f) ? reg : APSA_REGULARIZATION;
    filter->freeze_adaptation = false;
    filter->buffer_idx = 0;
}

void apsa_reset(apsa_filter_t *filter) {
    if (!filter) return;
    memset(filter->weights, 0, sizeof(filter->weights));
    memset(filter->x_buffer, 0, sizeof(filter->x_buffer));
    memset(filter->d_past, 0, sizeof(filter->d_past));
    filter->buffer_idx = 0;
}

void apsa_set_adaptation_freeze(apsa_filter_t *filter, bool freeze) {
    if (filter) {
        filter->freeze_adaptation = freeze;
    }
}

float apsa_process_sample(apsa_filter_t *filter, float d_sample, float x_sample) {
    const int L = APSA_FILTER_TAPS;
    const int total_buf = L + APSA_PROJECTION_ORDER;
    
    // Insert new reference sample into ring buffer
    filter->buffer_idx = (filter->buffer_idx - 1 + total_buf) % total_buf;
    filter->x_buffer[filter->buffer_idx] = x_sample;
    
    // Shift past desired samples: d_past[0] = d(n), d_past[1] = d(n-1)
    filter->d_past[1] = filter->d_past[0];
    filter->d_past[0] = d_sample;
    
    // P = 2 Affine Projection vectors:
    // x_0 corresponds to x(n), x_1 corresponds to x(n-1)
    float y0 = 0.0f;
    float y1 = 0.0f;
    float r00 = filter->regularization;
    float r11 = filter->regularization;
    float r01 = 0.0f;
    
    int idx0 = filter->buffer_idx;
    int idx1 = (filter->buffer_idx + 1) % total_buf;
    
    for (int i = 0; i < L; i++) {
        float x0_i = filter->x_buffer[(idx0 + i) % total_buf];
        float x1_i = filter->x_buffer[(idx1 + i) % total_buf];
        float w_i  = filter->weights[i];
        
        y0 += w_i * x0_i;
        y1 += w_i * x1_i;
        
        r00 += x0_i * x0_i;
        r11 += x1_i * x1_i;
        r01 += x0_i * x1_i;
    }
    
    // Error vector: e = d - y
    float e0 = filter->d_past[0] - y0;
    float e1 = filter->d_past[1] - y1;
    
    // Adapt weights if not frozen by Voice Activity Detection
    if (!filter->freeze_adaptation) {
        // Analytical inverse of 2x2 matrix R = [r00 r01; r01 r11]
        float det = (r00 * r11) - (r01 * r01);
        if (det > 1e-10f) {
            float inv_det = 1.0f / det;
            float inv_r00 =  r11 * inv_det;
            float inv_r11 =  r00 * inv_det;
            float inv_r01 = -r01 * inv_det;
            
            // Sign of error vector (APSA core innovation against explosive transients)
            float s0 = sign_val(e0);
            float s1 = sign_val(e1);
            
            // v = R^(-1) * sgn(e)
            float v0 = inv_r00 * s0 + inv_r01 * s1;
            float v1 = inv_r01 * s0 + inv_r11 * s1;
            
            float mu = filter->step_size;
            
            // w(n+1) = w(n) + mu * (v0 * x0 + v1 * x1)
            for (int i = 0; i < L; i++) {
                float x0_i = filter->x_buffer[(idx0 + i) % total_buf];
                float x1_i = filter->x_buffer[(idx1 + i) % total_buf];
                filter->weights[i] += mu * (v0 * x0_i + v1 * x1_i);
            }
        }
    }
    
    return e0;
}

void apsa_process_block(apsa_filter_t *filter, const float *d_in, const float *x_in, 
                        float *e_out, int num_samples) {
    if (!filter || !d_in || !x_in || !e_out) return;
    for (int i = 0; i < num_samples; i++) {
        e_out[i] = apsa_process_sample(filter, d_in[i], x_in[i]);
    }
}
