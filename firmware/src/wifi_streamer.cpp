#include "wifi_streamer.h"
#include "wifi_config.h"
#include "audio_config.h"

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiServer.h>
#include <WiFiClient.h>
#include <mbedtls/sha1.h>
#include <mbedtls/base64.h>

#define MAX_STREAM_CLIENTS 2
#define BATCH_FRAMES       4
#define BATCH_SAMPLES      (AUDIO_FRAME_SIZE * BATCH_FRAMES) // 160 * 4 = 640 samples = 40ms
#define AUDIO_QUEUE_LEN    6                                 // 6 batches = 240ms max buffer
#define ALL_STREAM_SAMPLES (BATCH_SAMPLES * 3)                 // primary + ref + clean
#define ALL_STREAM_BYTES   (ALL_STREAM_SAMPLES * sizeof(int16_t))


typedef struct {
    int16_t clean[BATCH_SAMPLES];
    int16_t primary[BATCH_SAMPLES];
    int16_t ref[BATCH_SAMPLES];
} audio_stream_packet_t;

static QueueHandle_t s_audio_queue = NULL;
static volatile uint32_t s_queue_drop_count = 0;
static volatile uint32_t s_queue_send_count = 0;
static volatile uint32_t s_queue_max_depth = 0;
static volatile uint32_t s_frame_calls = 0;
static volatile uint32_t s_batch_produced = 0;
static volatile uint32_t s_max_frame_gap_us = 0;
static volatile int64_t s_last_frame_call_us = 0;
static WiFiServer    s_server(WIFI_STREAM_PORT);
static WiFiClient    s_clients[MAX_STREAM_CLIENTS];
static bool          s_is_websocket[MAX_STREAM_CLIENTS];
static stream_source_t s_client_source[MAX_STREAM_CLIENTS];
static uint8_t s_all_stream_packet[ALL_STREAM_BYTES];


// Telemetry cache
static volatile float    s_cur_latency_us = 0.0f;
static volatile float    s_cur_atten_db = 0.0f;
static volatile bool     s_cur_vad_speech = false;
static volatile float    s_cur_p_in = 0.0f;
static volatile float    s_cur_p_ref = 0.0f;

// Modern Responsive HTML5 Web Dashboard (stored in Flash PROGMEM)
static const char INDEX_HTML[] PROGMEM = R"rawliteral(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
<title>SIH Defence ANC — Real-Time Tactical Audio</title>
<style>
  :root {
    --bg: #0b0f19;
    --card: #151d30;
    --card-border: #233152;
    --accent: #00f0ff;
    --green: #00ff88;
    --amber: #ffaa00;
    --red: #ff3366;
    --text: #f0f4fc;
    --text-dim: #7a8ba9;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  body { background: var(--bg); color: var(--text); padding: 16px; min-height: 100vh; display: flex; flex-direction: column; align-items: center; }
  .container { width: 100%; max-width: 540px; display: flex; flex-direction: column; gap: 14px; }
  
  /* Header */
  .header { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: var(--card); border-radius: 12px; border: 1px solid var(--card-border); }
  .title { font-size: 1.1rem; font-weight: 800; letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px; }
  .badge { background: rgba(0,255,136,0.15); color: var(--green); padding: 4px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; border: 1px solid rgba(0,255,136,0.4); text-transform: uppercase; }
  .badge.live { animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* Hero Audio Unlock Banner */
  .unmute-banner { background: linear-gradient(135deg, #1f2d4d, #0f172a); border: 2px dashed var(--accent); border-radius: 14px; padding: 20px 16px; text-align: center; cursor: pointer; transition: all 0.2s ease; user-select: none; }
  .unmute-banner:active { transform: scale(0.98); background: #1f2d4d; }
  .unmute-icon { font-size: 2.2rem; margin-bottom: 6px; }
  .unmute-title { font-size: 1.15rem; font-weight: 800; color: #fff; margin-bottom: 4px; }
  .unmute-sub { font-size: 0.8rem; color: var(--text-dim); }
  .unmute-banner.playing { border: 2px solid var(--green); background: rgba(0,255,136,0.08); }
  .unmute-banner.playing .unmute-title { color: var(--green); }

  /* Waveform Visualizer + Live Mic Meters */
  .viz-card { background: var(--card); border: 1px solid var(--card-border); border-radius: 12px; padding: 12px; }
  canvas { width: 100%; height: 95px; background: #070a12; border-radius: 8px; display: block; }
  .mic-levels { display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.75rem; color: var(--text-dim); font-weight: 700; }
  .mic-levels span { color: #fff; font-family: monospace; font-size: 0.85rem; }

  /* Audio Source Switcher */
  .card-title { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); font-weight: 700; margin-bottom: 8px; }
  .source-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
  .source-btn { background: #0c1220; border: 1px solid var(--card-border); color: var(--text-dim); padding: 12px 6px; border-radius: 10px; font-weight: 700; font-size: 0.8rem; cursor: pointer; text-align: center; transition: all 0.15s ease; }
  .source-btn.active { background: rgba(0,240,255,0.15); color: var(--accent); border-color: var(--accent); box-shadow: 0 0 15px rgba(0,240,255,0.25); }
  .source-btn.active.clean { background: rgba(0,255,136,0.15); color: var(--green); border-color: var(--green); box-shadow: 0 0 15px rgba(0,255,136,0.25); }
  .source-btn.active.raw { background: rgba(255,170,0,0.15); color: var(--amber); border-color: var(--amber); box-shadow: 0 0 15px rgba(255,170,0,0.25); }
  .source-btn.active.ref { background: rgba(255,51,102,0.15); color: var(--red); border-color: var(--red); box-shadow: 0 0 15px rgba(255,51,102,0.25); }

  /* Telemetry Grid */
  .telemetry-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
  .tele-card { background: var(--card); border: 1px solid var(--card-border); border-radius: 10px; padding: 10px; text-align: center; }
  .tele-val { font-size: 1.25rem; font-weight: 800; color: #fff; margin-top: 2px; }
  .tele-val.green { color: var(--green); }
  .tele-val.amber { color: var(--amber); }

  /* VAD Banner */
  .vad-card { background: var(--card); border: 1px solid var(--card-border); border-radius: 10px; padding: 10px 14px; display: flex; justify-content: space-between; align-items: center; }
  .vad-status { font-weight: 800; font-size: 0.85rem; }
  .vad-status.speech { color: var(--amber); }
  .vad-status.noise { color: var(--green); }

  /* Volume slider */
  .vol-row { display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: var(--card); border: 1px solid var(--card-border); border-radius: 10px; font-size: 0.8rem; color: var(--text-dim); }
  .vol-row input { flex: 1; accent-color: var(--accent); }
</style>
</head>
<body>

<div class="container">
  <!-- Header -->
  <div class="header">
    <div class="title">
      <span>🛡️</span> SIH DEFENCE ANC
    </div>
    <div id="statusBadge" class="badge live">CONNECTING...</div>
  </div>

  <!-- Hero Audio Unlock Banner -->
  <div id="unmuteBanner" class="unmute-banner">
    <div id="unmuteIcon" class="unmute-icon">🔊</div>
    <div id="unmuteTitle" class="unmute-title">TAP ANYWHERE TO LISTEN</div>
    <div id="unmuteSub" class="unmute-sub">Real-Time 16 kHz Audio Stream (No Audio Files Stored)</div>
  </div>

  <!-- Real-Time Waveform Visualizer + Live Mic Meters -->
  <div class="viz-card">
    <div class="card-title">LIVE OSCILLOSCOPE</div>
    <canvas id="scopeCanvas"></canvas>
    <div class="mic-levels">
      <div>🎤 Mic 1 (Primary): <span id="lvlMic1">0.0000</span></div>
      <div>🔊 Mic 2 (Ref): <span id="lvlMic2">0.0000</span></div>
    </div>
  </div>

  <!-- Source Channel Switcher -->
  <div>
    <div class="card-title">AUDIO MONITORING CHANNEL</div>
    <div class="source-grid">
      <button id="btnClean" class="source-btn active clean" onclick="setSource('clean')">✨ ENHANCED<br><small>(Clean Voice)</small></button>
      <button id="btnPrimary" class="source-btn" onclick="setSource('primary')">🎤 RAW INPUT<br><small>(Voice + Noise)</small></button>
      <button id="btnRef" class="source-btn" onclick="setSource('ref')">🔊 REF NOISE<br><small>(Noise Only)</small></button>
    </div>
  </div>

  <!-- Real-Time Telemetry -->
  <div class="telemetry-grid">
    <div class="tele-card">
      <div class="card-title">LATENCY</div>
      <div id="telLatency" class="tele-val green">&lt; 1 ms</div>
    </div>
    <div class="tele-card">
      <div class="card-title">ATTENUATION</div>
      <div id="telAtten" class="tele-val green">+10.2 dB</div>
    </div>
    <div class="tele-card">
      <div class="card-title">SAMPLING</div>
      <div class="tele-val">16 kHz</div>
    </div>
  </div>

  <!-- VAD State -->
  <div class="vad-card">
    <span class="card-title" style="margin:0;">VAD / DOUBLE-TALK:</span>
    <span id="vadStatus" class="vad-status noise">NOISE ONLY (ADAPTING)</span>
  </div>

  <!-- Volume Slider (Up to 400% boost for phone speakers) -->
  <div class="vol-row">
    <span>🔈 VOLUME</span>
    <input type="range" id="volSlider" min="0" max="4" step="0.1" value="1.0" oninput="setVolume(this.value)">
    <span id="volVal">100%</span>
  </div>
</div>

<script>
let audioCtx = null;
let gainNode = null;
let nextPlayTime = 0;
let ws = null;
let currentSource = 'clean';
let isAudioActive = false;

// Oscilloscope setup
const canvas = document.getElementById('scopeCanvas');
const ctx = canvas.getContext('2d');
let lastWaveform = new Float32Array(320);

function resizeCanvas() {
  canvas.width = canvas.clientWidth * window.devicePixelRatio;
  canvas.height = canvas.clientHeight * window.devicePixelRatio;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawScope() {
  requestAnimationFrame(drawScope);
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = '#070a12';
  ctx.fillRect(0, 0, w, h);

  // Center reference line
  ctx.strokeStyle = '#151d30';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  // Waveform line with visual gain
  ctx.strokeStyle = currentSource === 'clean' ? '#00ff88' : (currentSource === 'primary' ? '#ffaa00' : '#ff3366');
  ctx.lineWidth = 2 * window.devicePixelRatio;
  ctx.beginPath();

  const sliceWidth = w / lastWaveform.length;
  let x = 0;
  for (let i = 0; i < lastWaveform.length; i++) {
    const v = lastWaveform[i];
    const y = (0.5 - v * 2.0) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
    x += sliceWidth;
  }
  ctx.stroke();
}
drawScope();

// Initialize Web Audio Context
function initAudio() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass({ sampleRate: 16000 });
    gainNode = audioCtx.createGain();
    gainNode.gain.value = parseFloat(document.getElementById('volSlider').value);
    gainNode.connect(audioCtx.destination);
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  isAudioActive = true;
  updateBanner();
}

function updateBanner() {
  const banner = document.getElementById('unmuteBanner');
  const title = document.getElementById('unmuteTitle');
  const sub = document.getElementById('unmuteSub');
  const icon = document.getElementById('unmuteIcon');
  if (isAudioActive && audioCtx && audioCtx.state === 'running') {
    banner.classList.add('playing');
    icon.textContent = '🟢';
    title.textContent = 'LIVE STREAMING ACTIVE';
    sub.textContent = 'Real-time audio outputting to device speakers / headphones';
  } else {
    banner.classList.remove('playing');
    icon.textContent = '🔊';
    title.textContent = 'TAP ANYWHERE TO LISTEN';
    sub.textContent = 'Browser blocked autoplay — tap once to start real-time audio';
  }
}

// User gesture unlock for entire document
['click', 'touchstart', 'touchend', 'keydown'].forEach(evt => {
  document.addEventListener(evt, () => {
    initAudio();
  }, { passive: true });
});

// Play 16-bit PCM Audio Chunk (640 samples = 40ms, crackle-free)
function playPcmChunk(int16Buf) {
  if (!audioCtx || audioCtx.state !== 'running') return;
  
  const numSamples = int16Buf.length;
  const float32 = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    float32[i] = int16Buf[i] / 32768.0;
  }
  lastWaveform = float32.slice(0, 320);

  const audioBuf = audioCtx.createBuffer(1, numSamples, 16000);
  audioBuf.copyToChannel(float32, 0);

  const src = audioCtx.createBufferSource();
  src.buffer = audioBuf;
  src.connect(gainNode);

  const now = audioCtx.currentTime;
  if (nextPlayTime < now) {
    // 50ms lead time to eliminate under-run crackle
    nextPlayTime = now + 0.050;
  }
  src.start(nextPlayTime);
  nextPlayTime += audioBuf.duration;
}

// WebSocket connection with automatic reconnection
function connectWebSocket() {
  const wsUrl = `ws://${window.location.host}/ws`;
  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    document.getElementById('statusBadge').textContent = 'CONNECTED';
    document.getElementById('statusBadge').style.borderColor = 'rgba(0,255,136,0.6)';
    document.getElementById('statusBadge').style.color = '#00ff88';
    setSource(currentSource);
  };

  ws.onclose = () => {
    document.getElementById('statusBadge').textContent = 'RECONNECTING...';
    document.getElementById('statusBadge').style.borderColor = 'rgba(255,170,0,0.6)';
    document.getElementById('statusBadge').style.color = '#ffaa00';
    setTimeout(connectWebSocket, 500);
  };

  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      // Telemetry JSON
      try {
        const tel = JSON.parse(event.data);
        if (tel.lat !== undefined) {
          document.getElementById('telLatency').textContent = tel.lat < 1000 ? `${tel.lat} μs` : `${(tel.lat/1000).toFixed(1)} ms`;
        }
        if (tel.att !== undefined) {
          document.getElementById('telAtten').textContent = `${tel.att > 0 ? '+' : ''}${tel.att.toFixed(1)} dB`;
        }
        if (tel.vad !== undefined) {
          const vElem = document.getElementById('vadStatus');
          if (tel.vad) {
            vElem.textContent = 'SPEECH DETECTED (APSA FROZEN)';
            vElem.className = 'vad-status speech';
          } else {
            vElem.textContent = 'NOISE ONLY (ADAPTING)';
            vElem.className = 'vad-status noise';
          }
        }
        if (tel.pin !== undefined) {
          document.getElementById('lvlMic1').textContent = tel.pin.toFixed(4);
        }
        if (tel.pref !== undefined) {
          document.getElementById('lvlMic2').textContent = tel.pref.toFixed(4);
        }
      } catch (e) {}
    } else {
      // Binary Int16 PCM Audio
      playPcmChunk(new Int16Array(event.data));
    }
  };
}

function setSource(src) {
  currentSource = src;
  ['btnClean', 'btnPrimary', 'btnRef'].forEach(id => {
    document.getElementById(id).className = 'source-btn';
  });
  if (src === 'clean') document.getElementById('btnClean').className = 'source-btn active clean';
  if (src === 'primary') document.getElementById('btnPrimary').className = 'source-btn active raw';
  if (src === 'ref') document.getElementById('btnRef').className = 'source-btn active ref';

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(src);
  }
}

function setVolume(val) {
  if (gainNode) gainNode.gain.value = parseFloat(val);
  document.getElementById('volVal').textContent = `${Math.round(val * 100)}%`;
}

// Start WebSocket on load
window.addEventListener('load', () => {
  connectWebSocket();
  initAudio();
});
</script>
</body>
</html>
)rawliteral";

// WebSocket Key Handshake Helper
static String compute_websocket_accept(const String &key) {
    String combined = key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    unsigned char sha1_res[20];
    mbedtls_sha1((const unsigned char*)combined.c_str(), combined.length(), sha1_res);
    
    unsigned char base64_res[36];
    size_t olen = 0;
    mbedtls_base64_encode(base64_res, sizeof(base64_res), &olen, sha1_res, 20);
    base64_res[olen] = '\0';
    return String((char*)base64_res);
}

// Send binary WebSocket frame in a single atomic TCP write (RFC 6455)
static bool ws_send_binary(WiFiClient &client, const uint8_t *data, size_t len) {
    if (!client || !client.connected()) return false;
    static uint8_t packet[4 + ALL_STREAM_BYTES];
    size_t header_len = 0;
    packet[0] = 0x82; // FIN + Binary Frame opcode
    if (len < 126) {
        packet[1] = (uint8_t)len;
        header_len = 2;
    } else {
        packet[1] = 126;
        packet[2] = (uint8_t)((len >> 8) & 0xFF);
        packet[3] = (uint8_t)(len & 0xFF);
        header_len = 4;
    }
    memcpy(packet + header_len, data, len);
    size_t total_len = header_len + len;
    return (client.write(packet, total_len) == total_len);
}

// Send text WebSocket frame (RFC 6455)
static bool ws_send_text(WiFiClient &client, const char *str, size_t len) {
    if (!client || !client.connected()) return false;
    uint8_t packet[4 + 256];
    if (len > 250) len = 250;
    size_t header_len = 0;
    packet[0] = 0x81; // FIN + Text Frame opcode
    if (len < 126) {
        packet[1] = (uint8_t)len;
        header_len = 2;
    } else {
        packet[1] = 126;
        packet[2] = (uint8_t)((len >> 8) & 0xFF);
        packet[3] = (uint8_t)(len & 0xFF);
        header_len = 4;
    }
    memcpy(packet + header_len, str, len);
    size_t total_len = header_len + len;
    return (client.write(packet, total_len) == total_len);
}

// Process incoming WebSocket client message (e.g. channel switch "clean", "primary", "ref")
static void ws_handle_client_input(int client_idx) {
    WiFiClient &client = s_clients[client_idx];
    if (client.available() < 2) return;

    uint8_t b0 = client.read();
    uint8_t b1 = client.read();
    bool is_masked = (b1 & 0x80) != 0;
    uint64_t payload_len = (b1 & 0x7F);

    if (payload_len == 126) {
        if (client.available() < 2) return;
        payload_len = (client.read() << 8) | client.read();
    }

    uint8_t mask[4] = {0};
    if (is_masked) {
        if (client.available() < 4) return;
        for (int i = 0; i < 4; i++) mask[i] = client.read();
    }

    if (payload_len > 128) return; // Discard oversized message
    char msg[128];
    for (size_t i = 0; i < payload_len; i++) {
        uint8_t b = client.read();
        msg[i] = is_masked ? (b ^ mask[i % 4]) : b;
    }
    msg[payload_len] = '\0';

    if (strcmp(msg, "clean") == 0) {
        s_client_source[client_idx] = STREAM_MODE_CLEAN;
    } else if (strcmp(msg, "primary") == 0) {
        s_client_source[client_idx] = STREAM_MODE_PRIMARY;
    } else if (strcmp(msg, "ref") == 0) {
        s_client_source[client_idx] = STREAM_MODE_REF;
    } else if (strcmp(msg, "sync_all") == 0) {
        s_client_source[client_idx] = STREAM_MODE_ALL;
    }
}

// Background FreeRTOS task running on Core 0 (Network & HTTP/WebSocket management)
static void wifi_server_task(void *pvParameters) {
    uint32_t last_telemetry_ms = 0;

    while (1) {
        // 1. Accept new incoming clients
        if (s_server.hasClient()) {
            WiFiClient new_client = s_server.available();
            new_client.setNoDelay(true); // Disable Nagle algorithm for real-time streaming
            bool slot_found = false;
            for (int i = 0; i < MAX_STREAM_CLIENTS; i++) {
                if (!s_clients[i] || !s_clients[i].connected()) {
                    s_clients[i] = new_client;
                    s_is_websocket[i] = false;
                    s_client_source[i] = STREAM_MODE_CLEAN;
                    slot_found = true;
                    break;
                }
            }
            if (!slot_found) {
                new_client.stop(); // Reject if slots full
            }
        }

        // 2. Handle HTTP / WebSocket protocol for each connected client
        for (int i = 0; i < MAX_STREAM_CLIENTS; i++) {
            if (!s_clients[i] || !s_clients[i].connected()) continue;

            // If not yet upgraded to WebSocket, check for HTTP request
            if (!s_is_websocket[i]) {
                if (s_clients[i].available()) {
                    String req = s_clients[i].readStringUntil('\r');
                    s_clients[i].readStringUntil('\n'); // discard newline

                    // Check if WebSocket upgrade request
                    if (req.indexOf("GET /ws") >= 0) {
                        String ws_key = "";
                        while (s_clients[i].connected()) {
                            String header = s_clients[i].readStringUntil('\n');
                            if (header == "\r" || header.length() == 0) break;
                            if (header.startsWith("Sec-WebSocket-Key: ")) {
                                ws_key = header.substring(19);
                                ws_key.trim();
                            }
                        }
                        if (ws_key.length() > 0) {
                            String accept_key = compute_websocket_accept(ws_key);
                            s_clients[i].print("HTTP/1.1 101 Switching Protocols\r\n");
                            s_clients[i].print("Upgrade: websocket\r\n");
                            s_clients[i].print("Connection: Upgrade\r\n");
                            s_clients[i].print("Sec-WebSocket-Accept: " + accept_key + "\r\n\r\n");
                            s_is_websocket[i] = true;
                        }
                    } else if (req.indexOf("GET /") >= 0) {
                        // Serve embedded HTML5 dashboard
                        s_clients[i].print("HTTP/1.1 200 OK\r\n");
                        s_clients[i].print("Content-Type: text/html\r\n");
                        s_clients[i].print("Connection: close\r\n\r\n");
                        s_clients[i].print(FPSTR(INDEX_HTML));
                        s_clients[i].stop();
                    } else {
                        s_clients[i].print("HTTP/1.1 404 Not Found\r\n\r\n");
                        s_clients[i].stop();
                    }
                }
            } else {
                // Client is in WebSocket mode: check for incoming control commands
                if (s_clients[i].available()) {
                    ws_handle_client_input(i);
                }
            }
        }

        // 3. Dequeue real-time 40ms audio batches from Core 1 and stream to WebSocket clients
        static audio_stream_packet_t pkt;
        while (xQueueReceive(s_audio_queue, &pkt, 0) == pdTRUE) {
            for (int i = 0; i < MAX_STREAM_CLIENTS; i++) {
                if (s_clients[i] && s_clients[i].connected() && s_is_websocket[i]) {
                    const int16_t *buf = pkt.clean;
                    size_t send_len = BATCH_SAMPLES * sizeof(int16_t);

                    if (s_client_source[i] == STREAM_MODE_PRIMARY) {
                        buf = pkt.primary;
                    } else if (s_client_source[i] == STREAM_MODE_REF) {
                        buf = pkt.ref;
                    } else if (s_client_source[i] == STREAM_MODE_ALL) {
                        // Single synchronized packet:
                        // [PRIMARY 640][REF 640][CLEAN 640] int16 samples
                        memcpy(s_all_stream_packet,
                               pkt.primary,
                               BATCH_SAMPLES * sizeof(int16_t));
                        memcpy(s_all_stream_packet + BATCH_SAMPLES * sizeof(int16_t),
                               pkt.ref,
                               BATCH_SAMPLES * sizeof(int16_t));
                        memcpy(s_all_stream_packet + 2 * BATCH_SAMPLES * sizeof(int16_t),
                               pkt.clean,
                               BATCH_SAMPLES * sizeof(int16_t));

                        bool ok = ws_send_binary(
                            s_clients[i],
                            s_all_stream_packet,
                            ALL_STREAM_BYTES
                        );
                        if (!ok) {
                            s_clients[i].stop();
                            s_is_websocket[i] = false;
                        }
                        continue;
                    }

                    bool ok = ws_send_binary(
                        s_clients[i],
                        (const uint8_t*)buf,
                        send_len
                    );
                    if (!ok) {
                        s_clients[i].stop();
                        s_is_websocket[i] = false;
                    }
                }
            }
        }

        // 4. Send periodic telemetry JSON every 500ms
        uint32_t now = millis();

        if (now - last_telemetry_ms >= 500) {
            last_telemetry_ms = now;

            char tele_json[200];

            snprintf(
                tele_json,
                sizeof(tele_json),
                "{\"lat\":%d,\"att\":%.1f,\"vad\":%s,"
                "\"pin\":%.4f,\"pref\":%.4f,"
                "\"qdrop\":%lu,\"qsend\":%lu,\"qmax\":%lu,"
                "\"frames\":%lu,\"batches\":%lu,\"maxgap\":%lu}",
                (int)s_cur_latency_us,
                (double)s_cur_atten_db,
                s_cur_vad_speech ? "true" : "false",
                (double)s_cur_p_in,
                (double)s_cur_p_ref,
                (unsigned long)s_queue_drop_count,
                (unsigned long)s_queue_send_count,
                (unsigned long)s_queue_max_depth,
                (unsigned long)s_frame_calls,
                (unsigned long)s_batch_produced,
                (unsigned long)s_max_frame_gap_us
            );

            for (int i = 0; i < MAX_STREAM_CLIENTS; i++) {
                if (s_clients[i] &&
                    s_clients[i].connected() &&
                    s_is_websocket[i]) {

                    ws_send_text(
                        s_clients[i],
                        tele_json,
                        strlen(tele_json)
                    );
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(5)); // Low CPU idle delay on Core 0
    }
}

bool wifi_streamer_init(void) {
    // 1. Create real-time audio FIFO queue
    s_audio_queue = xQueueCreate(AUDIO_QUEUE_LEN, sizeof(audio_stream_packet_t));
    if (!s_audio_queue) {
        Serial.println("[ERROR] Failed to allocate audio streaming queue!");
        return false;
    }

    // 2. Configure WiFi mode
#if WIFI_MODE_STATION
    Serial.printf("[WIFI] Connecting to Station SSID: %s ...\n", WIFI_STA_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_STA_SSID, WIFI_STA_PASSWORD);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WIFI] Connected! Web Audio Player: http://%s/\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WARNING] Station WiFi failed. Falling back to Access Point mode...");
        WiFi.mode(WIFI_AP);
        IPAddress local_ip(192, 168, 4, 1);
        IPAddress gateway(192, 168, 4, 1);
        IPAddress subnet(255, 255, 255, 0);
        WiFi.softAPConfig(local_ip, gateway, subnet);
        WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD);
        Serial.printf("[WIFI] Access Point '%s' ready. Open: http://%s/\n", WIFI_AP_SSID, WiFi.softAPIP().toString().c_str());
    }
#else
    Serial.printf("[WIFI] Starting Access Point: '%s' (Pass: '%s') ...\n", WIFI_AP_SSID, WIFI_AP_PASSWORD);
    WiFi.mode(WIFI_AP);
    IPAddress local_ip(192, 168, 4, 1);
    IPAddress gateway(192, 168, 4, 1);
    IPAddress subnet(255, 255, 255, 0);
    WiFi.softAPConfig(local_ip, gateway, subnet);
    WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD);
    Serial.printf("[WIFI] Access Point started!\n");
    Serial.printf("========================================================\n");
    Serial.printf("  CONNECT YOUR PHONE TO WIFI:  %s\n", WIFI_AP_SSID);
    Serial.printf("  WIFI PASSWORD:              %s\n", WIFI_AP_PASSWORD);
    Serial.printf("  OPEN PHONE BROWSER TO:      http://%s/\n", WiFi.softAPIP().toString().c_str());
    Serial.printf("========================================================\n");
#endif

    // 3. Start Port 80 Web & WebSocket Server
    s_server.begin();

    // 4. Launch streaming task on Core 0 (leaving Core 1 100% dedicated to real-time DSP)
    xTaskCreatePinnedToCore(
        wifi_server_task,
        "wifi_stream_task",
        8192,
        NULL,
        3, // Standard priority
        NULL,
        0  // Pinned to Core 0
    );

    return true;
}

void wifi_streamer_send_frame(const float *clean, const float *primary, const float *ref, int num_samples) {
    int64_t now_us = esp_timer_get_time();

    s_frame_calls++;

    if (s_last_frame_call_us != 0) {
        uint32_t gap_us = (uint32_t)(now_us - s_last_frame_call_us);
        if (gap_us > s_max_frame_gap_us) {
            s_max_frame_gap_us = gap_us;
        }
    }

    s_last_frame_call_us = now_us;
    if (!s_audio_queue || !wifi_streamer_has_clients()) return;

    static audio_stream_packet_t s_accum_pkt;
    static int s_accum_count = 0;

    int offset = s_accum_count * AUDIO_FRAME_SIZE;
    for (int i = 0; i < num_samples && i < AUDIO_FRAME_SIZE; i++) {
        float c = clean[i];
        if (c > 1.0f) c = 1.0f; else if (c < -1.0f) c = -1.0f;
        s_accum_pkt.clean[offset + i] = (int16_t)(c * 32767.0f);

        float p = primary[i];
        if (p > 1.0f) p = 1.0f; else if (p < -1.0f) p = -1.0f;
        s_accum_pkt.primary[offset + i] = (int16_t)(p * 32767.0f);

        float r = ref[i];
        if (r > 1.0f) r = 1.0f; else if (r < -1.0f) r = -1.0f;
        s_accum_pkt.ref[offset + i] = (int16_t)(r * 32767.0f);
    }
    s_accum_count++;

    if (s_accum_count >= BATCH_FRAMES) {
        s_accum_count = 0;
        s_batch_produced++;
        if (xQueueSend(s_audio_queue, &s_accum_pkt, 0) != pdTRUE) {
            s_queue_drop_count++;

            audio_stream_packet_t dummy;
            xQueueReceive(s_audio_queue, &dummy, 0);

            xQueueSend(s_audio_queue, &s_accum_pkt, 0);
        } else {
            s_queue_send_count++;
        }

        UBaseType_t depth = uxQueueMessagesWaiting(s_audio_queue);
        if ((uint32_t)depth > s_queue_max_depth) {
            s_queue_max_depth = depth;
        }
    }
}





void wifi_streamer_update_telemetry(float latency_us, float atten_db, bool vad_speech, float p_in, float p_ref) {
    s_cur_latency_us = latency_us;
    s_cur_atten_db = atten_db;
    s_cur_vad_speech = vad_speech;
    s_cur_p_in = p_in;
    s_cur_p_ref = p_ref;
}

bool wifi_streamer_has_clients(void) {
    for (int i = 0; i < MAX_STREAM_CLIENTS; i++) {
        if (s_clients[i] && s_clients[i].connected() && s_is_websocket[i]) {
            return true;
        }
    }
    return false;
}
