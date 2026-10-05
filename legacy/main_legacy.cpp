/* #include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include "audio_config.h"
#include "apsa_filter.h"
#include "flann_filter.h"
#include "features_esp32.h"
#include "gru_inference.h"
#include "i2s_audio.h"
#include "wifi_streamer.h"

// Filter and Neural Network instances
static apsa_filter_t    g_apsa;
static flann_filter_t   g_flann;
static esp32_features_t g_features;
static gru_state_t      g_gru_state;

// Audio processing frame buffers (allocated in fast SRAM for low latency)
static float g_mic_primary[AUDIO_FRAME_SIZE];
static float g_mic_ref[AUDIO_FRAME_SIZE];
static float g_apsa_res[AUDIO_FRAME_SIZE];
static float g_flann_res[AUDIO_FRAME_SIZE];
static float g_clean_out[AUDIO_FRAME_SIZE];

// Feature extraction and neural network buffers
static float g_features_44[MODEL_INPUT_FEATURES];
static float g_fft_real[FFT_BINS];
static float g_fft_imag[FFT_BINS];
static float g_band_gains[BARK_NUM_BANDS];

// FreeRTOS Task handle
static TaskHandle_t s_dsp_task_handle = NULL;

// Telemetry counters
static uint32_t s_frame_counter = 0;
static float s_total_latency_us = 0.0f;

// Volatile telemetry for cross-core reading (DSP writes on Core 1, Core 0 prints)
static volatile float    s_report_avg_latency = 0.0f;
static volatile float    s_report_atten_db = 0.0f;
static volatile uint32_t s_report_frame = 0;
static volatile bool     s_report_vad_speech = false;
static volatile bool     s_report_ready = false;

void print_system_memory(void) {
    Serial.println("\n========================================================");
    Serial.println("  SIH DEFENCE ANC — ESP32-S3 (N16R8) STARTUP");
    Serial.println("  Complete Pipeline: APSA -> FLANN -> Tiny GRU -> DAC");
    Serial.println("========================================================");
    Serial.printf("CPU Frequency         : %d MHz\n", getCpuFrequencyMhz());
    Serial.printf("Internal SRAM Free    : %d KB\n", heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024);
    
    size_t psram_size = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    Serial.printf("Octal PSRAM Total     : %.2f MB\n", psram_size / (1024.0f * 1024.0f));
    Serial.printf("Octal PSRAM Free      : %.2f MB\n", psram_free / (1024.0f * 1024.0f));
    
    if (psram_size < 1024 * 1024) {
        Serial.println("[WARNING] Octal PSRAM not detected! Check platformio.ini memory_type.");
    } else {
        Serial.println("[OK] 8MB Octal PSRAM active and ready for Edge AI / DSP.");
    }
    Serial.println("========================================================\n");
}
 */

/* void dsp_processing_task(void *pvParameters) {
    while (1) {
        // 1. Acquire synchronized stereo frame from I2S DMA
        bool read_ok = i2s_audio_read_frame(g_mic_primary, g_mic_ref, AUDIO_FRAME_SIZE);
        if (!read_ok) {
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        
        int64_t t_start = esp_timer_get_time();
        
        // --- Double-Talk / Speech Leakage Detector (Bugs #14, #15) ---
        float p_pri = 0.0f, p_ref = 0.0f;
        for (int i = 0; i < AUDIO_FRAME_SIZE; i++) {
            p_pri += g_mic_primary[i] * g_mic_primary[i];
            p_ref += g_mic_ref[i] * g_mic_ref[i];
        }
        float ratio = p_pri / (p_ref + 1e-6f);
        bool speech_detected = (ratio > 1.8f) && (p_pri > 1e-4f);
        
        // Freeze APSA weight adaptation during soldier speech to prevent voice cancellation
        apsa_set_adaptation_freeze(&g_apsa, speech_detected);
        
        // 2. Stage 1: APSA Adaptive Filter (Linear Impulse-Robust Cancellation)
        apsa_process_block(&g_apsa, g_mic_primary, g_mic_ref, g_apsa_res, AUDIO_FRAME_SIZE);
        
        // 3. Stage 2: FLANN Filter (Chebyshev Polynomial Non-linear Expansion)
        flann_process_block(&g_flann, g_apsa_res, g_mic_ref, g_flann_res, AUDIO_FRAME_SIZE);
        
        // 4. Stage 3: Tiny ML Speech Enhancer (Bugs #3, #4, #5, #6)
        // Feature Extraction: 160-sample frame -> 44 causal features + 129 complex FFT bins
        features_esp32_extract(&g_features, g_flann_res, g_features_44, g_fft_real, g_fft_imag);
        
        // Causal GRU Inference: 44 features -> 22 spectral gains in [0.0, 1.0]
        gru_inference_step(&g_gru_state, g_features_44, g_band_gains);
        
        // Causal Overlap-Add Synthesis: Apply gains to FFT bins -> 160 clean audio samples
        features_esp32_synthesize(&g_features, g_fft_real, g_fft_imag, g_band_gains, g_clean_out);
        
        int64_t t_end = esp_timer_get_time();
        
        // 5. Stage 4: Output clean audio to I2S DAC / Headset (if present)
        i2s_audio_write_frame(g_clean_out, AUDIO_FRAME_SIZE);

        // 6. Stage 5: Real-time low-latency audio stream to phone / browser over WiFi
        wifi_streamer_send_frame(g_clean_out, g_mic_primary, g_mic_ref, AUDIO_FRAME_SIZE);
        
        int64_t latency_us = t_end - t_start;
        s_total_latency_us += latency_us;
        s_frame_counter++;
        
        // Prepare telemetry snapshot every 100 frames (~1 second)
        if (s_frame_counter % 100 == 0) {
            float p_in = 0.0f, p_out = 0.0f, p_ref = 0.0f;
            for (int i = 0; i < AUDIO_FRAME_SIZE; i++) {
                p_in  += g_mic_primary[i] * g_mic_primary[i];
                p_out += g_clean_out[i]   * g_clean_out[i];
                p_ref += g_mic_ref[i]     * g_mic_ref[i];
            }
            p_in  = sqrtf(p_in / AUDIO_FRAME_SIZE) + 1e-6f;
            p_out = sqrtf(p_out / AUDIO_FRAME_SIZE) + 1e-6f;
            p_ref = sqrtf(p_ref / AUDIO_FRAME_SIZE) + 1e-6f;
            
            s_report_avg_latency = s_total_latency_us / 100.0f;
            s_report_atten_db = 20.0f * log10f(p_in / p_out);
            s_report_frame = s_frame_counter;
            s_report_vad_speech = speech_detected;
            s_total_latency_us = 0.0f;
            s_report_ready = true;

            // Broadcast live telemetry to web dashboard
            wifi_streamer_update_telemetry(s_report_avg_latency, s_report_atten_db, speech_detected, p_in, p_ref);
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000); // Allow USB CDC to attach
    
    print_system_memory();
    
    // Initialize DSP filters
    Serial.println("[INIT] Initializing APSA filter (64 taps, P=2)...");
    apsa_init(&g_apsa, APSA_STEP_SIZE, APSA_REGULARIZATION);
    
    Serial.println("[INIT] Initializing FLANN filter (16 taps, order 3)...");
    flann_init(&g_flann, FLANN_STEP_SIZE);
    
    // Initialize Feature Extractor and Synthesis Engine
    Serial.println("[INIT] Initializing 256-pt Causal STFT & 22-Band Bark Filterbank...");
    features_esp32_init(&g_features);
    
    // Initialize Tiny GRU Inference Engine
    Serial.println("[INIT] Initializing TinySpeechEnhancer GRU Inference Engine...");
    gru_inference_init(&g_gru_state);
    
    // Initialize I2S dual-mic driver (RX on GPIO 4, 5, 6)
    Serial.println("[INIT] Starting I2S dual-microphone driver (RX)...");
    if (!i2s_audio_init()) {
        Serial.println("[FATAL] I2S Mic Driver failed to initialize. Check audio_config.h.");
        while (1) { delay(1000); }
    }
    
    // Initialize I2S DAC / speaker driver (TX on GPIO 15, 16, 17)
    Serial.println("[INIT] Starting I2S DAC output driver (TX)...");
    if (!i2s_speaker_init()) {
        Serial.println("[WARNING] I2S Speaker Driver failed (no hardware DAC). Proceeding with WiFi streaming.");
    }

    // Initialize Real-Time WiFi Audio Streamer on Core 0 (for Phone / Laptop web playback)
    Serial.println("[INIT] Starting Real-Time WiFi Audio Streamer...");
    if (!wifi_streamer_init()) {
        Serial.println("[WARNING] WiFi Streamer failed to start.");
    }
    
    // Spawn real-time DSP & ML task pinned to Core 1 with 16KB stack
    xTaskCreatePinnedToCore(
        dsp_processing_task,
        "dsp_ml_task",
        16384,
        NULL,
        configMAX_PRIORITIES - 1, // Maximum real-time audio priority
        &s_dsp_task_handle,
        1 // Pinned to Core 1
    );
    
    Serial.println("[OK] SIH Defence ANC system running (APSA -> FLANN -> Tiny GRU -> DAC).");
}

void loop() {
    // Print telemetry on Core 0 (non-real-time) to prevent DSP jitter
    if (s_report_ready) {
        s_report_ready = false;
        Serial.printf("FRAME:%u | End-to-End Latency: %.1f us (%.2f%% of 10ms deadline) | Attenuation: %.2f dB | VAD Speech: %s\n",
                      s_report_frame, s_report_avg_latency,
                      (s_report_avg_latency / 10000.0f) * 100.0f, s_report_atten_db,
                      s_report_vad_speech ? "ACTIVE (APSA Frozen)" : "Noise Only (Adapting)");
    }
    vTaskDelay(pdMS_TO_TICKS(100));
}
 */

#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <math.h>
#include "audio_config.h"
#include "apsa_filter.h"
#include "flann_filter.h"
#include "features_esp32.h"
#include "gru_inference.h"
#include "i2s_audio.h"
#include "wifi_streamer.h"

// ============================================================================
// SIH DEFENCE ANC — DIAGNOSTIC MAIN
//
// Phone browser channels:
//   PRIMARY   = always raw primary microphone
//   REFERENCE = always raw reference microphone
//   CLEAN     = selectable diagnostic stage (see Serial commands below)
//
// Serial commands:
//   raw       -> CLEAN = raw primary mic
//   apsa      -> CLEAN = APSA residual
//   dsp       -> CLEAN = APSA + FLANN residual
//   full      -> CLEAN = APSA + FLANN + GRU
//   reset     -> reset APSA/FLANN/STFT/GRU states
//   help      -> print commands
//
// This lets us identify exactly which stage is destroying the audio without
// reflashing the ESP32 between tests.
// ============================================================================

enum monitor_stage_t : uint8_t {
    MONITOR_RAW  = 0,
    MONITOR_APSA = 1,
    MONITOR_DSP  = 2,   // APSA + FLANN
    MONITOR_FULL = 3
};

// Start with RAW so the first boot is guaranteed to be an understandable
// microphone test. Change live with Serial commands.
static volatile uint8_t g_monitor_stage = MONITOR_RAW;

// Filter and Neural Network instances
static apsa_filter_t    g_apsa;
static flann_filter_t   g_flann;
static esp32_features_t g_features;
static gru_state_t      g_gru_state;

// Audio processing frame buffers
static float g_mic_primary[AUDIO_FRAME_SIZE];
static float g_mic_ref[AUDIO_FRAME_SIZE];
static float g_apsa_res[AUDIO_FRAME_SIZE];
static float g_flann_res[AUDIO_FRAME_SIZE];
static float g_clean_out[AUDIO_FRAME_SIZE];

// Feature extraction / neural network buffers
static float g_features_44[MODEL_INPUT_FEATURES];
static float g_fft_real[FFT_BINS];
static float g_fft_imag[FFT_BINS];
static float g_band_gains[BARK_NUM_BANDS];

// FreeRTOS task handle
static TaskHandle_t s_dsp_task_handle = NULL;

// Telemetry counters
static uint32_t s_frame_counter = 0;
static float s_total_latency_us = 0.0f;

// Volatile telemetry for cross-core reading
static volatile float    s_report_avg_latency = 0.0f;
static volatile float    s_report_atten_db = 0.0f;
static volatile uint32_t s_report_frame = 0;
static volatile bool     s_report_vad_speech = false;
static volatile bool     s_report_ready = false;
static volatile uint8_t  s_report_stage = MONITOR_RAW;
static volatile float    s_report_rms_raw = 0.0f;
static volatile float    s_report_rms_apsa = 0.0f;
static volatile float    s_report_rms_dsp = 0.0f;
static volatile float    s_report_rms_full = 0.0f;

static const char* stage_name(uint8_t stage) {
    switch (stage) {
        case MONITOR_RAW:  return "RAW";
        case MONITOR_APSA: return "APSA";
        case MONITOR_DSP:  return "APSA+FLANN";
        case MONITOR_FULL: return "FULL";
        default:           return "UNKNOWN";
    }
}

static float rms_of(const float *x, int n) {
    if (!x || n <= 0) return 0.0f;

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = (double)x[i];
        sum += v * v;
    }

    return sqrtf((float)(sum / (double)n));
}

static float attenuation_db(float input_rms, float output_rms) {
    // Positive means output energy is lower than raw input.
    return 20.0f * log10f((input_rms + 1e-7f) / (output_rms + 1e-7f));
}

static void reset_processing_state() {
    apsa_reset(&g_apsa);
    flann_reset(&g_flann);
    features_esp32_reset(&g_features);
    gru_inference_reset(&g_gru_state);

    memset(g_mic_primary, 0, sizeof(g_mic_primary));
    memset(g_mic_ref, 0, sizeof(g_mic_ref));
    memset(g_apsa_res, 0, sizeof(g_apsa_res));
    memset(g_flann_res, 0, sizeof(g_flann_res));
    memset(g_clean_out, 0, sizeof(g_clean_out));
    memset(g_features_44, 0, sizeof(g_features_44));
    memset(g_fft_real, 0, sizeof(g_fft_real));
    memset(g_fft_imag, 0, sizeof(g_fft_imag));
    memset(g_band_gains, 0, sizeof(g_band_gains));

    Serial.println("[RESET] APSA + FLANN + STFT + GRU state reset.");
}

static void print_help() {
    Serial.println();
    Serial.println("================ ANC AUDIO DIAGNOSTICS ================");
    Serial.println("Phone browser channel mapping:");
    Serial.println("  PRIMARY   = raw primary microphone");
    Serial.println("  REFERENCE = raw reference microphone");
    Serial.println("  CLEAN     = selected diagnostic stage");
    Serial.println();
    Serial.println("Serial commands:");
    Serial.println("  raw       -> CLEAN = raw primary mic");
    Serial.println("  apsa      -> CLEAN = APSA residual");
    Serial.println("  dsp       -> CLEAN = APSA + FLANN residual");
    Serial.println("  full      -> CLEAN = APSA + FLANN + GRU");
    Serial.println("  reset     -> reset all adaptive / streaming states");
    Serial.println("  help      -> print this menu");
    Serial.println("========================================================");
    Serial.println();
}

static void handle_serial_command() {
    static String cmd;

    while (Serial.available()) {
        char c = (char)Serial.read();
        Serial.println(c);

        // Accept either Enter style: \n or \r
        if (c == '\n' || c == '\r') {
            if (cmd.length() == 0) {
                continue;
            }

            cmd.trim();
            cmd.toLowerCase();

            Serial.printf("[RX] '%s'\n", cmd.c_str());

            if (cmd == "raw" || cmd == "0") {
                g_monitor_stage = MONITOR_RAW;
                Serial.println("[MONITOR] CLEAN -> RAW PRIMARY");
            }
            else if (cmd == "apsa" || cmd == "1") {
                g_monitor_stage = MONITOR_APSA;
                Serial.println("[MONITOR] CLEAN -> APSA residual");
            }
            else if (cmd == "dsp" || cmd == "flann" || cmd == "2") {
                g_monitor_stage = MONITOR_DSP;
                Serial.println("[MONITOR] CLEAN -> APSA + FLANN residual");
            }
            else if (cmd == "full" || cmd == "gru" || cmd == "3") {
                g_monitor_stage = MONITOR_FULL;
                Serial.println("[MONITOR] CLEAN -> FULL APSA + FLANN + GRU");
            }
            else if (cmd == "reset") {
                reset_processing_state();
            }
            else if (cmd == "help" || cmd == "?") {
                print_help();
            }
            else {
                Serial.printf("[MONITOR] Unknown command: '%s'\n", cmd.c_str());
            }

            cmd = "";
        }
        else {
            // Prevent accidental runaway input
            if (cmd.length() < 32) {
                cmd += c;
            }
        }
    }
}

void print_system_memory() {
    Serial.println("\n========================================================");
    Serial.println("  SIH DEFENCE ANC — ESP32-S3 (N16R8) DIAGNOSTIC BUILD");
    Serial.println("  RAW / APSA / APSA+FLANN / FULL-GRU monitor");
    Serial.println("========================================================");
    Serial.printf("CPU Frequency         : %d MHz\n", getCpuFrequencyMhz());
    Serial.printf("Internal SRAM Free    : %d KB\n", heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024);

    size_t psram_size = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t psram_free = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    Serial.printf("Octal PSRAM Total     : %.2f MB\n", psram_size / (1024.0f * 1024.0f));
    Serial.printf("Octal PSRAM Free      : %.2f MB\n", psram_free / (1024.0f * 1024.0f));

    if (psram_size < 1024 * 1024) {
        Serial.println("[WARNING] Octal PSRAM not detected! Check platformio.ini memory_type.");
    } else {
        Serial.println("[OK] 8MB Octal PSRAM active and ready for Edge AI / DSP.");
    }

    Serial.println("========================================================\n");
}

/**
 * Real-time DSP & Edge AI Processing Task pinned to Core 1.
 * Core 0 handles network, serial commands, and telemetry reporting.
 */
void dsp_processing_task(void *pvParameters) {
    (void)pvParameters;

    while (1) {
        // ---------------------------------------------------------------------
        // 1. Acquire synchronized stereo frame from the two INMP441 mics
        // ---------------------------------------------------------------------
        bool read_ok = i2s_audio_read_frame(
            g_mic_primary,
            g_mic_ref,
            AUDIO_FRAME_SIZE
        );

        if (!read_ok) {
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }

        int64_t t_start = esp_timer_get_time();

        // ---------------------------------------------------------------------
        // 2. Simple double-talk / speech detector used by adaptive filters
        // ---------------------------------------------------------------------
        float p_pri = 0.0f;
        float p_ref = 0.0f;

        for (int i = 0; i < AUDIO_FRAME_SIZE; ++i) {
            p_pri += g_mic_primary[i] * g_mic_primary[i];
            p_ref += g_mic_ref[i] * g_mic_ref[i];
        }

        float ratio = p_pri / (p_ref + 1e-6f);
        bool speech_detected = (ratio > 1.8f) && (p_pri > 1e-4f);

        // Keep the existing production behavior for the adaptive filters.
        apsa_set_adaptation_freeze(&g_apsa, speech_detected);

        // ---------------------------------------------------------------------
        // 3. Stage 1 — APSA
        // ---------------------------------------------------------------------
        apsa_process_block(
            &g_apsa,
            g_mic_primary,
            g_mic_ref,
            g_apsa_res,
            AUDIO_FRAME_SIZE
        );

        // ---------------------------------------------------------------------
        // 4. Stage 2 — FLANN
        // ---------------------------------------------------------------------
        flann_process_block(
            &g_flann,
            g_apsa_res,
            g_mic_ref,
            g_flann_res,
            AUDIO_FRAME_SIZE
        );

        // ---------------------------------------------------------------------
        // 5. Stage 3 — Tiny GRU enhancement
        // ---------------------------------------------------------------------
        features_esp32_extract(
            &g_features,
            g_flann_res,
            g_features_44,
            g_fft_real,
            g_fft_imag
        );

        gru_inference_step(
            &g_gru_state,
            g_features_44,
            g_band_gains
        );

        features_esp32_synthesize(
            &g_features,
            g_fft_real,
            g_fft_imag,
            g_band_gains,
            g_clean_out
        );

        int64_t t_end = esp_timer_get_time();

        // ---------------------------------------------------------------------
        // 6. Select what the phone's CLEAN channel actually hears
        // ---------------------------------------------------------------------
        const uint8_t selected_stage = g_monitor_stage;
        const float *selected_output = g_clean_out;

        switch (selected_stage) {
            case MONITOR_RAW:
                selected_output = g_mic_primary;
                break;

            case MONITOR_APSA:
                selected_output = g_apsa_res;
                break;

            case MONITOR_DSP:
                selected_output = g_flann_res;
                break;

            case MONITOR_FULL:
            default:
                selected_output = g_clean_out;
                break;
        }

        // Optional hardware DAC mirrors the currently selected stage too.
        i2s_audio_write_frame(selected_output, AUDIO_FRAME_SIZE);

        // Phone:
        //   CLEAN    = selected diagnostic stage
        //   PRIMARY  = raw primary mic
        //   REFERENCE= raw reference mic
        wifi_streamer_send_frame(
            selected_output,
            g_mic_primary,
            g_mic_ref,
            AUDIO_FRAME_SIZE
        );

        // ---------------------------------------------------------------------
        // 7. Telemetry
        // ---------------------------------------------------------------------
        s_total_latency_us += (float)(t_end - t_start);
        s_frame_counter++;

        if (s_frame_counter % 100 == 0) {
            const float rms_raw  = rms_of(g_mic_primary, AUDIO_FRAME_SIZE);
            const float rms_apsa = rms_of(g_apsa_res, AUDIO_FRAME_SIZE);
            const float rms_dsp  = rms_of(g_flann_res, AUDIO_FRAME_SIZE);
            const float rms_full = rms_of(g_clean_out, AUDIO_FRAME_SIZE);
            const float rms_sel  = rms_of(selected_output, AUDIO_FRAME_SIZE);

            s_report_avg_latency = s_total_latency_us / 100.0f;
            s_report_atten_db = attenuation_db(rms_raw, rms_sel);
            s_report_frame = s_frame_counter;
            s_report_vad_speech = speech_detected;
            s_report_stage = selected_stage;
            s_report_rms_raw = rms_raw;
            s_report_rms_apsa = rms_apsa;
            s_report_rms_dsp = rms_dsp;
            s_report_rms_full = rms_full;

            s_total_latency_us = 0.0f;
            s_report_ready = true;

            // Keep the web telemetry field meaningful for the selected output.
            wifi_streamer_update_telemetry(
                s_report_avg_latency,
                s_report_atten_db,
                speech_detected,
                rms_raw,
                rms_of(g_mic_ref, AUDIO_FRAME_SIZE)
            );
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void setup() {
    Serial.begin(115200);
    Serial.setTimeout(50);
    delay(1000); // Allow USB CDC to attach

    print_system_memory();
    print_help();

    Serial.println("[INIT] Initializing APSA filter (64 taps, P=2)...");
    apsa_init(&g_apsa, APSA_STEP_SIZE, APSA_REGULARIZATION);

    Serial.println("[INIT] Initializing FLANN filter (16 taps, order 3)...");
    flann_init(&g_flann, FLANN_STEP_SIZE);

    Serial.println("[INIT] Initializing 256-pt causal STFT + 22-band Bark filterbank...");
    features_esp32_init(&g_features);

    Serial.println("[INIT] Initializing TinySpeechEnhancer GRU...");
    gru_inference_init(&g_gru_state);

    Serial.println("[INIT] Starting I2S dual-microphone driver (RX on GPIO 4/5/6)...");
    if (!i2s_audio_init()) {
        Serial.println("[FATAL] I2S Mic Driver failed to initialize. Check audio_config.h.");
        while (1) {
            delay(1000);
        }
    }

    Serial.println("[INIT] Starting I2S DAC output driver (TX on GPIO 15/16/17)...");
    if (!i2s_speaker_init()) {
        Serial.println("[WARNING] I2S Speaker Driver failed (no hardware DAC). WiFi streaming will continue.");
    }

    Serial.println("[INIT] Starting Real-Time WiFi Audio Streamer...");
    if (!wifi_streamer_init()) {
        Serial.println("[WARNING] WiFi Streamer failed to start.");
    }

    xTaskCreatePinnedToCore(
        dsp_processing_task,
        "dsp_ml_task",
        16384,
        NULL,
        2,
        &s_dsp_task_handle,
        1
    );

    Serial.println("[OK] Diagnostic firmware running.");
    Serial.println("[OK] Open the phone dashboard and use the CLEAN channel.");
    Serial.println("[OK] Use Serial commands to change CLEAN stage live.");
}

void loop() {
    // Handle stage selection without touching the real-time DSP task.
    handle_serial_command();

    // Print one diagnostic snapshot per second on Core 0.
    if (s_report_ready) {
        s_report_ready = false;

        const float raw_rms  = s_report_rms_raw;
        const float apsa_rms = s_report_rms_apsa;
        const float dsp_rms  = s_report_rms_dsp;
        const float full_rms = s_report_rms_full;

        Serial.printf(
            "FRAME:%u | Stage:%s | Latency:%.1fus | Selected Atten:%+.2fdB | VAD:%s\n",
            s_report_frame,
            stage_name(s_report_stage),
            s_report_avg_latency,
            s_report_atten_db,
            s_report_vad_speech ? "SPEECH/FROZEN" : "NOISE/ADAPTING"
        );

        Serial.printf(
            "    RMS raw=%.5f | APSA=%.5f (%+.2fdB) | APSA+FLANN=%.5f (%+.2fdB) | FULL=%.5f (%+.2fdB)\n",
            raw_rms,
            apsa_rms,
            attenuation_db(raw_rms, apsa_rms),
            dsp_rms,
            attenuation_db(raw_rms, dsp_rms),
            full_rms,
            attenuation_db(raw_rms, full_rms)
        );
    }

    vTaskDelay(pdMS_TO_TICKS(20));
}
