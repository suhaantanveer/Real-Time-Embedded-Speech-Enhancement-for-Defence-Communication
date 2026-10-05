#include "i2s_audio.h"
#include <esp_heap_caps.h>
#include <driver/i2s.h>
#include <esp_log.h>
#include <string.h>

static const char *TAG = "I2S_AUDIO";
static const i2s_port_t I2S_PORT = I2S_NUM_0;

bool i2s_audio_init(void) {
    ESP_LOGI(TAG, "Initializing I2S dual-microphone driver (16 kHz, Stereo)...");
    
    // INMP441 sends 24-bit data MSB-aligned in 32-bit frame slots
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = AUDIO_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT, // Stereo: Left=Mic1, Right=Mic2
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = AUDIO_FRAME_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_MIC_BCLK_PIN,
        .ws_io_num = I2S_MIC_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_MIC_DIN_PIN
    };
    
    esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed installing I2S driver: %d", err);
        return false;
    }
    
    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed setting I2S pins: %d", err);
        return false;
    }
    
    i2s_zero_dma_buffer(I2S_PORT);
    ESP_LOGI(TAG, "I2S dual-mic driver successfully initialized.");
    return true;
}

bool i2s_audio_read_frame(float *primary_out, float *ref_out, int num_samples) {
    if (!primary_out || !ref_out) return false;
    
    // Each stereo sample consists of 2 x 32-bit words (Left and Right)
    static int32_t raw_i2s_buf[AUDIO_FRAME_SIZE * 2];
    size_t bytes_to_read = num_samples * 2 * sizeof(int32_t);
    size_t bytes_read = 0;
    
    esp_err_t res = i2s_read(I2S_PORT, raw_i2s_buf, bytes_to_read, &bytes_read, portMAX_DELAY);
    if (res != ESP_OK || bytes_read != bytes_to_read) {
        ESP_LOGW(TAG, "I2S read error or partial read: %d (got %d of %d)", res, bytes_read, bytes_to_read);
        return false;
    }
    
    // INMP441 24-bit data in 32-bit slot:
    // Bits [31:8] contain 24-bit signed audio. Shift right by 8 and apply preamp gain.
    const float norm = (1.0f / 8388608.0f) * MIC_PREAMP_GAIN;
    
    for (int i = 0; i < num_samples; i++) {
        int32_t sample_left = raw_i2s_buf[i * 2 + 0] >> 8;
        int32_t sample_right = raw_i2s_buf[i * 2 + 1] >> 8;
        
        float sl = (float)sample_left * norm;
        float sr = (float)sample_right * norm;
        // Soft clipping via tanhf to eliminate harsh digital clipping distortion
        if (sl > 1.0f || sl < -1.0f) sl = tanhf(sl);
        if (sr > 1.0f || sr < -1.0f) sr = tanhf(sr);

#if I2S_SWAP_MIC_CHANNELS
        primary_out[i] = sr;
        ref_out[i]     = sl;
#else
        primary_out[i] = sl;
        ref_out[i]     = sr;
#endif
    }
    
    return true;
}

static const i2s_port_t I2S_SPK_PORT = I2S_NUM_1;

bool i2s_speaker_init(void) {
    ESP_LOGI(TAG, "Initializing I2S DAC / Speaker output driver (16 kHz, Mono/Stereo TX)...");
    
    i2s_config_t spk_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = AUDIO_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT, // Duplicate mono to both ears
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = AUDIO_FRAME_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = true,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t spk_pin_config = {
        .bck_io_num = I2S_SPK_BCLK_PIN,
        .ws_io_num = I2S_SPK_WS_PIN,
        .data_out_num = I2S_SPK_DOUT_PIN,
        .data_in_num = I2S_PIN_NO_CHANGE
    };
    
    esp_err_t err = i2s_driver_install(I2S_SPK_PORT, &spk_config, 0, NULL);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed installing I2S speaker driver: %d", err);
        return false;
    }
    
    err = i2s_set_pin(I2S_SPK_PORT, &spk_pin_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed setting I2S speaker pins: %d", err);
        return false;
    }
    
    i2s_zero_dma_buffer(I2S_SPK_PORT);
    ESP_LOGI(TAG, "I2S speaker output driver successfully initialized on GPIO %d, %d, %d.",
             I2S_SPK_BCLK_PIN, I2S_SPK_WS_PIN, I2S_SPK_DOUT_PIN);
    return true;
}

bool i2s_audio_write_frame(const float *clean_in, int num_samples) {
    if (!clean_in) return false;
    
    // Convert float [-1.0f, +1.0f] to 16-bit signed PCM interleaved stereo
    static int16_t spk_buf[AUDIO_FRAME_SIZE * 2];
    
    for (int i = 0; i < num_samples; i++) {
        float sample = clean_in[i];
        if (sample > 0.999f) sample = 0.999f;
        if (sample < -0.999f) sample = -0.999f;
        int16_t pcm = (int16_t)(sample * 32767.0f);
        spk_buf[i * 2 + 0] = pcm; // Left ear
        spk_buf[i * 2 + 1] = pcm; // Right ear
    }
    
    size_t bytes_to_write = num_samples * 2 * sizeof(int16_t);
    size_t bytes_written = 0;
    
    esp_err_t res = i2s_write(I2S_SPK_PORT, spk_buf, bytes_to_write, &bytes_written, portMAX_DELAY);
    return (res == ESP_OK && bytes_written == bytes_to_write);
}

audio_ring_buffer_t* audio_ring_buffer_create(int num_frames) {
    audio_ring_buffer_t *rb = (audio_ring_buffer_t*)heap_caps_malloc(sizeof(audio_ring_buffer_t), MALLOC_CAP_SPIRAM);
    if (!rb) {
        rb = (audio_ring_buffer_t*)malloc(sizeof(audio_ring_buffer_t)); // Fallback to internal SRAM
    }
    if (!rb) return NULL;
    
    int total_samples = num_frames * AUDIO_FRAME_SIZE;
    rb->capacity = total_samples;
    rb->head = 0;
    rb->tail = 0;
    rb->count = 0;
    
    // Allocate audio arrays in Octal PSRAM
    rb->primary_buf = (float*)heap_caps_malloc(total_samples * sizeof(float), MALLOC_CAP_SPIRAM);
    rb->ref_buf = (float*)heap_caps_malloc(total_samples * sizeof(float), MALLOC_CAP_SPIRAM);
    
    if (!rb->primary_buf || !rb->ref_buf) {
        ESP_LOGE(TAG, "Failed allocating PSRAM ring buffer!");
        audio_ring_buffer_destroy(rb);
        return NULL;
    }
    
    ESP_LOGI(TAG, "Allocated %d audio samples ring buffer in Octal PSRAM.", total_samples);
    return rb;
}

void audio_ring_buffer_destroy(audio_ring_buffer_t *rb) {
    if (!rb) return;
    if (rb->primary_buf) free(rb->primary_buf);
    if (rb->ref_buf) free(rb->ref_buf);
    free(rb);
}
