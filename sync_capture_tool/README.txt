SYNCHRONIZED CAPTURE TOOL

1. From the repo root, run:
   python ml/enable_sync_capture.py

2. Flash:
   pio run -t upload

3. Connect the laptop to the ESP32 AP:
   SSID: SIH-ANC
   Password: 12345678

4. Run:
   python ml/capture_live_all.py --ip 192.168.4.1 --seconds 15

The new WebSocket message "sync_all" requests one binary packet containing:
  PRIMARY 640 samples (1280 bytes)
  REF     640 samples (1280 bytes)
  CLEAN   640 samples (1280 bytes)
Total = 3840 bytes.

Existing dashboard source modes continue to work unchanged.
The patcher makes .bak_sync backups before modifying firmware files.
