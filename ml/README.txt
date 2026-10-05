SYNCHRONIZED PRIMARY + REF + CLEAN CAPTURE
===========================================

Why this patch exists
---------------------
The normal streamer lets a client request only one source at a time. That makes
separate primary/ref/clean WAV files difficult to compare because they are not
the exact same 40 ms packet timeline.

This patch adds an "all" diagnostic mode. One WebSocket connection receives,
for every 40 ms packet:

    PRIMARY[640 int16] + REF[640 int16] + CLEAN[640 int16]

So the three output WAVs have exactly the same packet boundaries and timestamps.

Files
-----
- capture_live_all.py
- wifi_streamer_all.patch

Apply
-----
1. Open firmware/src/wifi_streamer.cpp.
2. Apply wifi_streamer_all.patch to that file.
3. Build and flash:
       pio run -t upload
4. Connect the laptop to SIH-ANC.
5. Run:
       python ml/capture_live_all.py --ip 192.168.4.1 --seconds 15

Expected result
---------------
~375 packets for 15 s, with ~240000 samples in each WAV if the stream runs
continuously with no drops.

The script writes:
    captures/sync_<timestamp>_primary.wav
    captures/sync_<timestamp>_ref.wav
    captures/sync_<timestamp>_clean.wav
    captures/sync_<timestamp>_stats.json

Notes
-----
- This uses one WebSocket client, so it does not consume a second client slot.
- Existing clean/primary/ref modes are unchanged.
- The extra packet size is 3840 bytes every 40 ms (~768 kbps payload), which is
  still tiny relative to 2.4 GHz Wi-Fi link capacity; the purpose is diagnostic
  synchronization, not a new production streaming format.
