#include "front_end.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float coeff_attack(void) {
    return 1.0f - expf(-1.0f / (IMPULSE_ATTACK_MS * 1e-3f * (float)AUDIO_SAMPLE_RATE));
}

static float coeff_release(void) {
    return 1.0f - expf(-1.0f / (IMPULSE_RELEASE_MS * 1e-3f * (float)AUDIO_SAMPLE_RATE));
}

void front_end_init(front_end_state_t *state) {
    if (!state) return;
    memset(state, 0, sizeof(*state));

    const float w0 = 2.0f * (float)M_PI * HPF_CUTOFF_HZ / (float)AUDIO_SAMPLE_RATE;
    const float c = cosf(w0);
    const float s = sinf(w0);
    const float q = 0.7071067811865476f;
    const float alpha = s / (2.0f * q);
    const float a0 = 1.0f + alpha;
    state->hp_b0 = ((1.0f + c) * 0.5f) / a0;
    state->hp_b1 = (-(1.0f + c)) / a0;
    state->hp_b2 = ((1.0f + c) * 0.5f) / a0;
    state->hp_a1 = (-2.0f * c) / a0;
    state->hp_a2 = (1.0f - alpha) / a0;
    state->energy_ema = 1e-8f;
    state->energy_ema_initialized = false;
    state->guard_gain = 1.0f;
    state->guard_target = 1.0f;
}

void front_end_reset(front_end_state_t *state) {
    if (!state) return;
    float b0 = state->hp_b0, b1 = state->hp_b1, b2 = state->hp_b2;
    float a1 = state->hp_a1, a2 = state->hp_a2;
    memset(state, 0, sizeof(*state));
    state->hp_b0 = b0; state->hp_b1 = b1; state->hp_b2 = b2;
    state->hp_a1 = a1; state->hp_a2 = a2;
    state->energy_ema = 1e-8f;
    state->energy_ema_initialized = false;
    state->guard_gain = 1.0f;
    state->guard_target = 1.0f;
}

static float process_dc_hpf(front_end_state_t *s, float x, int ch) {
    float y_dc = x - s->dc_prev_x[ch] + DC_BLOCK_COEFF * s->dc_prev_y[ch];
    s->dc_prev_x[ch] = x;
    s->dc_prev_y[ch] = y_dc;

    float y = s->hp_b0 * y_dc
            + s->hp_b1 * s->hp_x1[ch]
            + s->hp_b2 * s->hp_x2[ch]
            - s->hp_a1 * s->hp_y1[ch]
            - s->hp_a2 * s->hp_y2[ch];
    s->hp_x2[ch] = s->hp_x1[ch];
    s->hp_x1[ch] = y_dc;
    s->hp_y2[ch] = s->hp_y1[ch];
    s->hp_y1[ch] = y;
    return y;
}

void front_end_process_frame(front_end_state_t *state,
                             const float *mic1_in,
                             const float *mic2_in,
                             float *mic1_out,
                             float *mic2_out,
                             int n) {
    if (!state || !mic1_in || !mic2_in || !mic1_out || !mic2_out || n <= 0) return;
    float energy = 0.0f;
    float peak = 0.0f;
    for (int i = 0; i < n; ++i) {
        mic1_out[i] = process_dc_hpf(state, mic1_in[i], 0);
        mic2_out[i] = process_dc_hpf(state, mic2_in[i], 1);
        energy += mic1_out[i] * mic1_out[i];
        float p = fabsf(mic1_out[i]);
        if (p > peak) peak = p;
    }
    energy /= (float)n;
    const float threshold_ratio = powf(10.0f, IMPULSE_THRESHOLD_DB / 10.0f);
    bool triggered = false;
    if (!state->energy_ema_initialized) {
        state->energy_ema = fmaxf(energy, 1e-12f);
        state->energy_ema_initialized = true;
    } else {
        const float prev_ema = fmaxf(state->energy_ema, 1e-12f);
        const bool relative_trigger = energy > prev_ema * threshold_ratio;
        const bool peak_trigger = peak > IMPULSE_PEAK_THRESHOLD;
        triggered = relative_trigger || peak_trigger;
        state->energy_ema = (1.0f - IMPULSE_EMA_ALPHA) * prev_ema
                          + IMPULSE_EMA_ALPHA * fmaxf(energy, 1e-12f);
    }
    state->last_impulse_trigger = triggered;
    if (triggered) {
        state->guard_target = powf(10.0f, -IMPULSE_MAX_ATTENUATION_DB / 20.0f);
        state->impulse_frames++;
    } else {
        state->guard_target = 1.0f;
    }

    const float attack = coeff_attack();
    const float release = coeff_release();
    for (int i = 0; i < n; ++i) {
        const float coeff = (state->guard_target < state->guard_gain) ? attack : release;
        state->guard_gain += coeff * (state->guard_target - state->guard_gain);
        mic1_out[i] *= state->guard_gain;
    }
}
