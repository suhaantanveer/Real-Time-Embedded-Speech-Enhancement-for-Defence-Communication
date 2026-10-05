from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
H = ROOT / "firmware" / "include" / "wifi_streamer.h"
CPP = ROOT / "firmware" / "src" / "wifi_streamer.cpp"

def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

def backup(path):
    bak = path.with_suffix(path.suffix + ".bak_sync")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"[BACKUP] {bak}")
    return bak

if not H.exists():
    fail(f"Not found: {H}")
if not CPP.exists():
    fail(f"Not found: {CPP}")

h = H.read_text(encoding="utf-8")
cpp = CPP.read_text(encoding="utf-8")

changed_h = False
changed_cpp = False

# 1) Add a dedicated STREAM_MODE_ALL enum value.
old_enum = """    STREAM_MODE_REF = 2       // Reference mic (environmental noise)
} stream_source_t;"""
new_enum = """    STREAM_MODE_REF = 2,      // Reference mic (environmental noise)
    STREAM_MODE_ALL = 3       // Diagnostic: PRIMARY + REF + CLEAN in one packet
} stream_source_t;"""

if "STREAM_MODE_ALL = 3" not in h:
    if old_enum not in h:
        fail("Could not find the expected stream_source_t enum in wifi_streamer.h")
    h = h.replace(old_enum, new_enum, 1)
    changed_h = True

# 2) Add constants for the combined diagnostic packet.
anchor = """#define AUDIO_QUEUE_LEN    6                                 // 6 batches = 240ms max buffer"""
insert = anchor + """
#define ALL_STREAM_SAMPLES (BATCH_SAMPLES * 3)                 // primary + ref + clean
#define ALL_STREAM_BYTES   (ALL_STREAM_SAMPLES * sizeof(int16_t))
"""
if "ALL_STREAM_BYTES" not in cpp:
    if anchor not in cpp:
        fail("Could not find AUDIO_QUEUE_LEN anchor in wifi_streamer.cpp")
    cpp = cpp.replace(anchor, insert, 1)
    changed_cpp = True

# 3) Add a static buffer for the combined diagnostic packet.
anchor = """static stream_source_t s_client_source[MAX_STREAM_CLIENTS];"""
insert = anchor + """
static uint8_t s_all_stream_packet[ALL_STREAM_BYTES];
"""
if "s_all_stream_packet" not in cpp:
    if anchor not in cpp:
        fail("Could not find s_client_source declaration in wifi_streamer.cpp")
    cpp = cpp.replace(anchor, insert, 1)
    changed_cpp = True

# 4) Let the WebSocket client request "sync_all".
old = """    if (strcmp(msg, "clean") == 0) {
        s_client_source[client_idx] = STREAM_MODE_CLEAN;
    } else if (strcmp(msg, "primary") == 0) {
        s_client_source[client_idx] = STREAM_MODE_PRIMARY;
    } else if (strcmp(msg, "ref") == 0) {
        s_client_source[client_idx] = STREAM_MODE_REF;
    }
"""
new = """    if (strcmp(msg, "clean") == 0) {
        s_client_source[client_idx] = STREAM_MODE_CLEAN;
    } else if (strcmp(msg, "primary") == 0) {
        s_client_source[client_idx] = STREAM_MODE_PRIMARY;
    } else if (strcmp(msg, "ref") == 0) {
        s_client_source[client_idx] = STREAM_MODE_REF;
    } else if (strcmp(msg, "sync_all") == 0) {
        s_client_source[client_idx] = STREAM_MODE_ALL;
    }
"""
if 'strcmp(msg, "sync_all")' not in cpp:
    if old not in cpp:
        fail("Could not find the WebSocket source-selection block in wifi_streamer.cpp")
    cpp = cpp.replace(old, new, 1)
    changed_cpp = True

# 5) Replace only the audio send selection block with ALL support.
old = """                    const int16_t *buf = pkt.clean;
                    if (s_client_source[i] == STREAM_MODE_PRIMARY) buf = pkt.primary;
                    else if (s_client_source[i] == STREAM_MODE_REF) buf = pkt.ref;

                    bool ok = ws_send_binary(s_clients[i], (const uint8_t*)buf, BATCH_SAMPLES * sizeof(int16_t));
"""
new = """                    const int16_t *buf = pkt.clean;
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
"""
if 's_client_source[i] == STREAM_MODE_ALL' not in cpp:
    if old not in cpp:
        fail("Could not find the WebSocket binary-send block in wifi_streamer.cpp")
    cpp = cpp.replace(old, new, 1)
    changed_cpp = True

if not changed_h and not changed_cpp:
    print("[OK] Sync-all firmware support is already present. Nothing changed.")
    sys.exit(0)

backup(H)
backup(CPP)

H.write_text(h, encoding="utf-8", newline="\n")
CPP.write_text(cpp, encoding="utf-8", newline="\n")

print("[DONE] Firmware modified for synchronized capture.")
print("       Existing clean/primary/ref modes remain unchanged.")
print("       New diagnostic mode is requested by WebSocket message: sync_all")
print("")
print("Next:")
print("  pio run -t upload")
