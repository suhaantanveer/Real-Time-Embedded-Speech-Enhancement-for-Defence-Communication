#ifndef WIFI_STREAMER_H
#define WIFI_STREAMER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Audio stream channel selection
typedef enum {
    STREAM_MODE_CLEAN = 0,    // Enhanced clean speech (APSA + FLANN + GRU)
    STREAM_MODE_PRIMARY = 1,  // Raw primary mic (speech + environmental noise)
    STREAM_MODE_REF = 2,      // Reference mic (environmental noise)
    STREAM_MODE_ALL = 3       // Diagnostic: PRIMARY + REF + CLEAN in one packet
} stream_source_t;

/**
 * Initializes the WiFi subsystem (AP or Station) and starts the
 * real-time HTTP + WebSocket audio streaming server on Core 0.
 * Zero files stored on disk — audio is streamed as processed in real-time.
 */
bool wifi_streamer_init(void);

/**
 * Non-blocking push of 160 processed audio samples into the real-time stream queue.
 * Called directly by the DSP task on Core 1 every 10ms.
 * Takes < 5 microseconds; if the network queue is full, oldest frames drop
 * to ensure zero latency build-up and zero DSP stutter.
 */
void wifi_streamer_send_frame(const float *clean, const float *primary, const float *ref, int num_samples);

/**
 * Updates real-time telemetry broadcast to the web dashboard (latency, attenuation, VAD, mic levels).
 */
void wifi_streamer_update_telemetry(float latency_us, float atten_db, bool vad_speech, float p_in, float p_ref);

/**
 * Returns true if at least one client (phone, laptop) is actively streaming.
 */
bool wifi_streamer_has_clients(void);

#ifdef __cplusplus
}
#endif

#endif // WIFI_STREAMER_H
