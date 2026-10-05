#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <math.h>
#include <string.h>
#include "audio_config.h"
#include "apsa_filter.h"
#include "flann_filter.h"
#include "front_end.h"
#include "features_esp32.h"
#include "gru_inference.h"
#include "i2s_audio.h"
#include "wifi_streamer.h"

// ============================================================================
// SIH26052 redesigned production/diagnostic firmware
// Default path:
//   Mic1 -> front-end -> ML (mic2 disabled)
// Optional side-information path:
//   Mic1 + Mic2 features -> same ML model
// Legacy APSA/FLANN path remains available for controlled A/B experiments.
// ============================================================================

enum monitor_stage_t : uint8_t {
    MONITOR_RAW = 0,
    MONITOR_FRONTEND = 1,
    MONITOR_ML_MIC1 = 2,
    MONITOR_ML_DUAL = 3,
    MONITOR_LEGACY_APSA = 4,
    MONITOR_LEGACY_DSP = 5,
    MONITOR_LEGACY_FULL = 6
};

static volatile uint8_t g_monitor_stage = MONITOR_ML_MIC1;

static apsa_filter_t g_apsa;
static flann_filter_t g_flann;
static front_end_state_t g_front_end;
static esp32_features_t g_features;
static gru_state_t g_gru_mic1;
static gru_state_t g_gru_dual;
static gru_state_t g_gru_legacy;

static float g_mic_primary[AUDIO_FRAME_SIZE];
static float g_mic_ref[AUDIO_FRAME_SIZE];
static float g_front_primary[AUDIO_FRAME_SIZE];
static float g_front_ref[AUDIO_FRAME_SIZE];
static float g_apsa_res[AUDIO_FRAME_SIZE];
static float g_flann_res[AUDIO_FRAME_SIZE];
static float g_ml_mic1[AUDIO_FRAME_SIZE];
static float g_ml_dual[AUDIO_FRAME_SIZE];
static float g_selected[AUDIO_FRAME_SIZE];

static float g_features_89[MODEL_INPUT_FEATURES];
static float g_features_dual[MODEL_INPUT_FEATURES];
static float g_features_mic1[MODEL_INPUT_FEATURES];
static float g_fft_real[FFT_BINS];
static float g_fft_imag[FFT_BINS];
static float g_band_gains[GRU_OUTPUT_DIM];

static TaskHandle_t s_dsp_task_handle = NULL;
static uint32_t s_frame_counter = 0;
static float s_total_latency_us = 0.0f;

static volatile float s_report_avg_latency = 0.0f;
static volatile float s_report_atten_db = 0.0f;
static volatile uint32_t s_report_frame = 0;
static volatile bool s_report_ready = false;
static volatile bool s_report_ref_valid = false;
static volatile float s_report_input_rms = 0.0f;
static volatile float s_report_input_peak = 0.0f;
static volatile float s_report_output_rms = 0.0f;
static volatile float s_report_noise_floor = 0.0f;
static volatile uint8_t s_report_stage = MONITOR_ML_MIC1;

// Approximate 1.5 s minimum-statistics noise tracker at 10 ms/frame.
#define NOISE_TRACK_FRAMES 150
static float g_noise_rms_hist[NOISE_TRACK_FRAMES];
static int g_noise_rms_idx = 0;
static int g_noise_rms_count = 0;
static float g_dry_mix = 1.0f;
static bool g_wet_initialized = false;

static const char* stage_name(uint8_t stage) {
    switch (stage) {
        case MONITOR_RAW: return "RAW";
        case MONITOR_FRONTEND: return "FRONTEND";
        case MONITOR_ML_MIC1: return "ML-MIC1";
        case MONITOR_ML_DUAL: return "ML-DUAL";
        case MONITOR_LEGACY_APSA: return "LEGACY-APSA";
        case MONITOR_LEGACY_DSP: return "LEGACY-DSP";
        case MONITOR_LEGACY_FULL: return "LEGACY-FULL";
        default: return "UNKNOWN";
    }
}

static float rms_of(const float *x, int n) {
    if (!x || n <= 0) return 0.0f;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double v = x[i];
        sum += v * v;
    }
    return sqrtf((float)(sum / (double)n));
}

static float peak_of(const float *x, int n) {
    float peak = 0.0f;
    for (int i = 0; i < n; ++i) {
        float p = fabsf(x[i]);
        if (p > peak) peak = p;
    }
    return peak;
}

static float dbfs(float rms) { return 20.0f * log10f(rms + 1e-9f); }

static float min_noise_rms(void) {
    if (g_noise_rms_count <= 0) return 1.0f;
    float m = g_noise_rms_hist[0];
    for (int i = 1; i < g_noise_rms_count; ++i) if (g_noise_rms_hist[i] < m) m = g_noise_rms_hist[i];
    return m;
}

static void update_noise_floor(float rms) {
    g_noise_rms_hist[g_noise_rms_idx] = rms;
    g_noise_rms_idx = (g_noise_rms_idx + 1) % NOISE_TRACK_FRAMES;
    if (g_noise_rms_count < NOISE_TRACK_FRAMES) g_noise_rms_count++;
}

static void reset_processing_state(void) {
    apsa_reset(&g_apsa);
    flann_reset(&g_flann);
    front_end_reset(&g_front_end);
    features_esp32_reset(&g_features);
    gru_inference_reset(&g_gru_mic1);
    gru_inference_reset(&g_gru_dual);
    gru_inference_reset(&g_gru_legacy);
    memset(g_noise_rms_hist, 0, sizeof(g_noise_rms_hist));
    g_noise_rms_idx = 0;
    g_noise_rms_count = 0;
    g_dry_mix = 1.0f;
    g_wet_initialized = false;
}

static float smooth_mix(float current, float target) {
    // 100 ms crossfade = 10 frames. This avoids abrupt timbral jumps.
    const float step = (float)AUDIO_FRAME_SIZE / (AUDIO_SAMPLE_RATE * (DRY_WET_CROSSFADE_MS * 1e-3f));
    if (current < target) { current += step; if (current > target) current = target; }
    else { current -= step; if (current < target) current = target; }
    return current;
}

static void apply_dry_wet(const float *dry, const float *wet, float noise_floor_rms, float *out) {
    const float threshold = powf(10.0f, DRY_WET_NOISE_FLOOR_DBFS / 20.0f);
    // In quiet rooms, favor dry primary mic. In noise, favor learned wet output.
    const float target_wet = (noise_floor_rms <= threshold) ? 0.0f : 1.0f;
    if (!g_wet_initialized) {
        g_dry_mix = 1.0f - target_wet;
        g_wet_initialized = true;
    } else {
        const float target_dry = 1.0f - target_wet;
        float current_dry = g_dry_mix;
        float current_wet = 1.0f - current_dry;
        if (current_dry < target_dry) current_dry = smooth_mix(current_dry, target_dry);
        else current_dry = 1.0f - smooth_mix(current_wet, target_wet);
        g_dry_mix = current_dry;
    }
    const float wet_mix = 1.0f - g_dry_mix;
    for (int i = 0; i < AUDIO_FRAME_SIZE; ++i) out[i] = g_dry_mix * dry[i] + wet_mix * wet[i];
}

static bool mic2_is_healthy(const float *mic1, const float *mic2) {
    const float r1 = rms_of(mic1, AUDIO_FRAME_SIZE);
    const float r2 = rms_of(mic2, AUDIO_FRAME_SIZE);
    const float p2 = peak_of(mic2, AUDIO_FRAME_SIZE);
    if (r2 < 1e-5f || p2 > 1.15f) return false;
    double corr_num = 0.0, e1 = 1e-12, e2 = 1e-12;
    for (int i = 0; i < AUDIO_FRAME_SIZE; ++i) {
        corr_num += (double)mic1[i] * mic2[i];
        e1 += (double)mic1[i] * mic1[i];
        e2 += (double)mic2[i] * mic2[i];
    }
    const float corr = (float)(fabs(corr_num) / sqrt(e1 * e2));
    // If the two channels are effectively identical, treat reference as suspect.
    if (corr > 0.995f && fabsf(r1 - r2) < 0.02f) return false;
    return true;
}

static void print_help(void) {
    Serial.println("\n=== SIH26052 ANC V2 ===");
    Serial.println("raw       : raw Mic1");
    Serial.println("front     : front-end Mic1");
    Serial.println("ml1       : GRU using Mic1 only (default)");
    Serial.println("dual      : GRU using Mic1 + Mic2 side information");
    Serial.println("apsa      : legacy APSA residual (comparison only)");
    Serial.println("dsp       : legacy APSA+FLANN residual (comparison only)");
    Serial.println("legacy    : legacy APSA+FLANN+GRU path (comparison only)");
    Serial.println("reset     : reset streaming state");
    Serial.println("help      : this menu");
}

static void handle_serial_command(void) {
    static String cmd;
    while (Serial.available()) {
        const char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            cmd.trim();
            cmd.toLowerCase();
            if (cmd == "raw" || cmd == "0") g_monitor_stage = MONITOR_RAW;
            else if (cmd == "front" || cmd == "frontend" || cmd == "1") g_monitor_stage = MONITOR_FRONTEND;
            else if (cmd == "ml1" || cmd == "mic1" || cmd == "2") g_monitor_stage = MONITOR_ML_MIC1;
            else if (cmd == "dual" || cmd == "3") g_monitor_stage = MONITOR_ML_DUAL;
            else if (cmd == "apsa" || cmd == "4") g_monitor_stage = MONITOR_LEGACY_APSA;
            else if (cmd == "dsp" || cmd == "5") g_monitor_stage = MONITOR_LEGACY_DSP;
            else if (cmd == "legacy" || cmd == "full" || cmd == "6") g_monitor_stage = MONITOR_LEGACY_FULL;
            else if (cmd == "reset") reset_processing_state();
            else if (cmd == "help" || cmd == "?") print_help();
            if (cmd.length()) Serial.printf("[MONITOR] %s\n", stage_name(g_monitor_stage));
            cmd = "";
        } else if (cmd.length() < 32) cmd += c;
    }
}

void print_system_memory(void) {
    Serial.println("\n=== SIH26052 ANC V2 STARTUP ===");
    Serial.printf("CPU: %d MHz | SRAM free: %d KB | PSRAM: %.2f MB free\n",
                  getCpuFrequencyMhz(),
                  heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024,
                  heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / (1024.0f * 1024.0f));
    Serial.printf("Default model: %d inputs -> %d/%d GRU -> %d bands | G_MIN=%.2f\n",
                  MODEL_INPUT_FEATURES, GRU_HIDDEN1_DIM, GRU_HIDDEN2_DIM, GRU_OUTPUT_DIM, G_MIN);
}

void dsp_processing_task(void *pvParameters) {
    (void)pvParameters;
    while (1) {
        if (!i2s_audio_read_frame(g_mic_primary, g_mic_ref, AUDIO_FRAME_SIZE)) {
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        const int64_t t_start = esp_timer_get_time();

        front_end_process_frame(&g_front_end, g_mic_primary, g_mic_ref,
                               g_front_primary, g_front_ref, AUDIO_FRAME_SIZE);
        const bool ref_valid = mic2_is_healthy(g_front_primary, g_front_ref);

        const uint8_t stage = g_monitor_stage;
        if (stage == MONITOR_RAW) {
            memcpy(g_selected, g_mic_primary, sizeof(g_selected));
        } else if (stage == MONITOR_FRONTEND) {
            memcpy(g_selected, g_front_primary, sizeof(g_selected));
        } else if (stage == MONITOR_ML_MIC1) {
            features_esp32_extract(&g_features, g_front_primary, g_front_ref, false,
                                   g_features_mic1, g_fft_real, g_fft_imag);
            gru_inference_step(&g_gru_mic1, g_features_mic1, g_band_gains);
            features_esp32_synthesize(&g_features, g_fft_real, g_fft_imag, g_band_gains, g_ml_mic1);
            update_noise_floor(rms_of(g_front_primary, AUDIO_FRAME_SIZE));
            apply_dry_wet(g_front_primary, g_ml_mic1, min_noise_rms(), g_selected);
        } else if (stage == MONITOR_ML_DUAL) {
            features_esp32_extract(&g_features, g_front_primary, g_front_ref, ref_valid,
                                   g_features_dual, g_fft_real, g_fft_imag);
            gru_inference_step(&g_gru_dual, g_features_dual, g_band_gains);
            features_esp32_synthesize(&g_features, g_fft_real, g_fft_imag, g_band_gains, g_ml_dual);
            update_noise_floor(rms_of(g_front_primary, AUDIO_FRAME_SIZE));
            apply_dry_wet(g_front_primary, g_ml_dual, min_noise_rms(), g_selected);
        } else {
            // Legacy comparison path. Enable explicitly at compile time if the
            // old adaptive filters are needed; default deployment does not use it.
#if ENABLE_ADAPTIVE_PREFILTER
            const bool freeze_adaptation = !ref_valid;
            apsa_set_adaptation_freeze(&g_apsa, freeze_adaptation);
            flann_set_adaptation_freeze(&g_flann, freeze_adaptation);
            if (stage == MONITOR_LEGACY_APSA || stage == MONITOR_LEGACY_DSP || stage == MONITOR_LEGACY_FULL) {
                apsa_process_block(&g_apsa, g_front_primary, g_front_ref, g_apsa_res, AUDIO_FRAME_SIZE);
            }
            if (stage == MONITOR_LEGACY_DSP || stage == MONITOR_LEGACY_FULL) {
                flann_process_block(&g_flann, g_apsa_res, g_front_ref, g_flann_res, AUDIO_FRAME_SIZE);
            }
            if (stage == MONITOR_LEGACY_APSA) memcpy(g_selected, g_apsa_res, sizeof(g_selected));
            else if (stage == MONITOR_LEGACY_DSP) memcpy(g_selected, g_flann_res, sizeof(g_selected));
            else {
                features_esp32_extract(&g_features, g_flann_res, g_front_ref, ref_valid,
                                       g_features_89, g_fft_real, g_fft_imag);
                gru_inference_step(&g_gru_legacy, g_features_89, g_band_gains);
                features_esp32_synthesize(&g_features, g_fft_real, g_fft_imag, g_band_gains, g_selected);
            }
#else
            memcpy(g_selected, g_front_primary, sizeof(g_selected));
#endif
        }

        const float current_noise_floor = min_noise_rms();
        const int64_t t_end = esp_timer_get_time();

        i2s_audio_write_frame(g_selected, AUDIO_FRAME_SIZE);
        wifi_streamer_send_frame(g_selected, g_mic_primary, g_mic_ref, AUDIO_FRAME_SIZE);

        const float latency_us = (float)(t_end - t_start);
        s_total_latency_us += latency_us;
        ++s_frame_counter;
        if (s_frame_counter % 100 == 0) {
            const float in_rms = rms_of(g_front_primary, AUDIO_FRAME_SIZE);
            const float out_rms = rms_of(g_selected, AUDIO_FRAME_SIZE);
            const float in_peak = peak_of(g_mic_primary, AUDIO_FRAME_SIZE);
            s_report_avg_latency = s_total_latency_us / 100.0f;
            s_report_atten_db = 20.0f * log10f((in_rms + 1e-7f) / (out_rms + 1e-7f));
            s_report_frame = s_frame_counter;
            s_report_ref_valid = ref_valid;
            s_report_input_rms = in_rms;
            s_report_input_peak = in_peak;
            s_report_output_rms = out_rms;
            s_report_noise_floor = current_noise_floor;
            s_report_stage = stage;
            s_report_ready = true;
            s_total_latency_us = 0.0f;
            wifi_streamer_update_telemetry(s_report_avg_latency, s_report_atten_db,
                                           false, in_rms, rms_of(g_front_ref, AUDIO_FRAME_SIZE));
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void setup(void) {
    Serial.begin(115200);
    Serial.setTimeout(50);
    delay(1000);
    print_system_memory();
    print_help();

    apsa_init(&g_apsa, APSA_STEP_SIZE, APSA_REGULARIZATION);
    flann_init(&g_flann, FLANN_STEP_SIZE);
    front_end_init(&g_front_end);
    features_esp32_init(&g_features);
    gru_inference_init(&g_gru_mic1);
    gru_inference_init(&g_gru_dual);
    gru_inference_init(&g_gru_legacy);

    if (!i2s_audio_init()) {
        Serial.println("[FATAL] I2S microphone init failed.");
        while (1) delay(1000);
    }
    if (!i2s_speaker_init()) Serial.println("[WARN] DAC unavailable; WiFi audio remains available.");
    if (!wifi_streamer_init()) Serial.println("[WARN] WiFi streamer unavailable.");

    xTaskCreatePinnedToCore(dsp_processing_task, "dsp_ml_task", 16384,
                            NULL, 2, &s_dsp_task_handle, 1);
    Serial.println("[OK] SIH26052 V2 firmware started. Default output: ML-MIC1.");
}

void loop(void) {
    handle_serial_command();
    if (s_report_ready) {
        s_report_ready = false;
        Serial.printf("FRAME:%u | %s | latency=%.1fus | ref=%s | in=%.2f dBFS peak=%.3f | out=%.2f dBFS | noiseFloor=%.2f dBFS\n",
                      s_report_frame, stage_name(s_report_stage), s_report_avg_latency,
                      s_report_ref_valid ? "VALID" : "OFF/INVALID",
                      dbfs(s_report_input_rms), s_report_input_peak,
                      dbfs(s_report_output_rms), dbfs(s_report_noise_floor));
    }
    vTaskDelay(pdMS_TO_TICKS(20));
}
