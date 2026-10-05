#ifndef WIFI_CONFIG_H
#define WIFI_CONFIG_H

// ==============================================================================
// WiFi Audio Streaming Configuration
// Edit this file with your WiFi credentials before flashing.
// ==============================================================================

// --- MODE SELECTION ---
// 0 = Access Point (AP) — ESP32 creates its own hotspot, phone connects directly.
//     Use this if you don't want to share your home WiFi password.
// 1 = Station (STA) — ESP32 joins your existing WiFi network.
//     Phone must be on the same network.
#define WIFI_MODE_STATION   0

// --- ACCESS POINT SETTINGS (used when WIFI_MODE_STATION = 0) ---
#define WIFI_AP_SSID        "SIH-ANC"
#define WIFI_AP_PASSWORD    "12345678"   // min 8 chars; leave "" for open AP

// --- STATION SETTINGS (used when WIFI_MODE_STATION = 1) ---
#define WIFI_STA_SSID       "YourWiFiSSID"
#define WIFI_STA_PASSWORD   "YourWiFiPassword"

// --- STREAMING ---
#define WIFI_STREAM_PORT    80           // HTTP + WebSocket on port 80

#endif // WIFI_CONFIG_H
