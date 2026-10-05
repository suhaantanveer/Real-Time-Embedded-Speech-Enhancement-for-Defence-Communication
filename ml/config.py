"""Single-source Python configuration mirrored by firmware/include/audio_config.h."""
from math import sqrt

SAMPLE_RATE = 16000
FRAME_SIZE = 160
N_FFT = 256
HOP_LENGTH = 160
WIN_LENGTH = 256
NUM_BANDS = 22

HPF_CUTOFF_HZ = 100.0
DC_BLOCK_COEFF = 0.995
IMPULSE_THRESHOLD_DB = 20.0
IMPULSE_PEAK_THRESHOLD = 0.90
IMPULSE_EMA_ALPHA = 0.10
IMPULSE_MAX_ATTENUATION_DB = 12.0
IMPULSE_ATTACK_MS = 1.0
IMPULSE_RELEASE_MS = 40.0

# Dataset level augmentation. The lower ceiling leaves useful headroom so that
# global post-mix renormalization is rare and never acts as a compressor.
TARGET_LEVEL_MIN_DBFS = -45.0
TARGET_LEVEL_MAX_DBFS = -15.0
NOISE_ONLY_LEVEL_MIN_DBFS = -45.0
NOISE_ONLY_LEVEL_MAX_DBFS = -15.0
MIC2_NOISE_OFFSET_MIN_DB = -3.0
MIC2_NOISE_OFFSET_MAX_DB = 10.0
MIC2_DROPOUT_RATE = 0.30
MIC2_FAULT_RATE = 0.05

G_MIN = 0.1
N_FEATURES = 89
HIDDEN1 = 64
HIDDEN2 = 48

# Per-band mic-2 calibration offsets in dB. Filled after a real noise-only calibration.
MIC2_CALIBRATION_DB = [0.0] * NUM_BANDS

BARK_FREQS = [
    100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270,
    1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6350, 7700, 8000,
]


def rbj_highpass_coefficients(cutoff_hz: float = HPF_CUTOFF_HZ,
                              sample_rate: int = SAMPLE_RATE) -> tuple[float, ...]:
    """2nd-order Butterworth high-pass coefficients using the RBJ biquad form."""
    q = 1.0 / sqrt(2.0)
    w0 = 2.0 * 3.141592653589793 * cutoff_hz / sample_rate
    c = __import__("math").cos(w0)
    s = __import__("math").sin(w0)
    alpha = s / (2.0 * q)
    b0 = (1.0 + c) / 2.0
    b1 = -(1.0 + c)
    b2 = (1.0 + c) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * c
    a2 = 1.0 - alpha
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
