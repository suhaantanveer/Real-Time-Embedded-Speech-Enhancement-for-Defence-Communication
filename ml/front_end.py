"""Stateful causal microphone front-end mirrored by the ESP32 implementation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import (
    DC_BLOCK_COEFF,
    FRAME_SIZE,
    HPF_CUTOFF_HZ,
    IMPULSE_ATTACK_MS,
    IMPULSE_EMA_ALPHA,
    IMPULSE_MAX_ATTENUATION_DB,
    IMPULSE_PEAK_THRESHOLD,
    IMPULSE_RELEASE_MS,
    IMPULSE_THRESHOLD_DB,
    SAMPLE_RATE,
    rbj_highpass_coefficients,
)


@dataclass
class BiquadState:
    x1: float = 0.0
    x2: float = 0.0
    y1: float = 0.0
    y2: float = 0.0


class MicFrontEnd:
    """Causal DC block + 100 Hz HPF + optional mic-1 impulse guard."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.hp_b = rbj_highpass_coefficients(HPF_CUTOFF_HZ, sample_rate)
        self.dc_prev_x = [0.0, 0.0]
        self.dc_prev_y = [0.0, 0.0]
        self.hp_state = [BiquadState(), BiquadState()]
        self.energy_ema = 1e-8
        self.energy_ema_initialized = False
        self.guard_gain = 1.0
        self.trigger_count = 0
        self._attack = 1.0 - math.exp(-1.0 / (max(1e-6, IMPULSE_ATTACK_MS) * 1e-3 * sample_rate))
        self._release = 1.0 - math.exp(-1.0 / (max(1e-6, IMPULSE_RELEASE_MS) * 1e-3 * sample_rate))
        self.guard_target = 1.0

    def reset(self) -> None:
        self.dc_prev_x = [0.0, 0.0]
        self.dc_prev_y = [0.0, 0.0]
        self.hp_state = [BiquadState(), BiquadState()]
        self.energy_ema = 1e-8
        self.energy_ema_initialized = False
        self.guard_gain = 1.0
        self.trigger_count = 0
        self.guard_target = 1.0

    def _dc_hpf_sample(self, x: float, ch: int) -> float:
        y_dc = x - self.dc_prev_x[ch] + DC_BLOCK_COEFF * self.dc_prev_y[ch]
        self.dc_prev_x[ch] = x
        self.dc_prev_y[ch] = y_dc

        b0, b1, b2, a1, a2 = self.hp_b
        st = self.hp_state[ch]
        y = b0 * y_dc + b1 * st.x1 + b2 * st.x2 - a1 * st.y1 - a2 * st.y2
        st.x2 = st.x1
        st.x1 = y_dc
        st.y2 = st.y1
        st.y1 = y
        return y

    def process_frame(self, mic1: np.ndarray, mic2: np.ndarray,
                      apply_guard: bool = True) -> tuple[np.ndarray, np.ndarray, bool]:
        mic1 = np.asarray(mic1, dtype=np.float32)
        mic2 = np.asarray(mic2, dtype=np.float32)
        if mic1.shape != mic2.shape:
            raise ValueError("mic1 and mic2 must have equal shape")
        if mic1.ndim != 1:
            raise ValueError("inputs must be 1-D")

        y1 = np.empty_like(mic1)
        y2 = np.empty_like(mic2)
        for i, (x1, x2) in enumerate(zip(mic1, mic2)):
            y1[i] = self._dc_hpf_sample(float(x1), 0)
            y2[i] = self._dc_hpf_sample(float(x2), 1)

        if not apply_guard:
            return y1, y2, False

        energy = float(np.mean(y1 * y1)) if len(y1) else 0.0
        peak = float(np.max(np.abs(y1))) if len(y1) else 0.0
        prev_ema = self.energy_ema
        ratio = 10.0 ** (IMPULSE_THRESHOLD_DB / 10.0)

        # Compare against the previous frame's baseline. Do not let the current
        # frame raise its own reference before the trigger decision.
        if not self.energy_ema_initialized:
            triggered = False
            self.energy_ema = max(energy, 1e-12)
            self.energy_ema_initialized = True
        else:
            relative_trigger = energy > max(prev_ema, 1e-12) * ratio
            peak_trigger = peak > IMPULSE_PEAK_THRESHOLD
            triggered = bool(relative_trigger or peak_trigger)
            self.energy_ema = ((1.0 - IMPULSE_EMA_ALPHA) * max(prev_ema, 1e-12)
                               + IMPULSE_EMA_ALPHA * max(energy, 1e-12))

        if triggered:
            self.trigger_count += 1
            self.guard_target = 10.0 ** (-IMPULSE_MAX_ATTENUATION_DB / 20.0)
        else:
            self.guard_target = 1.0

        for i in range(len(y1)):
            coeff = self._attack if self.guard_target < self.guard_gain else self._release
            self.guard_gain += coeff * (self.guard_target - self.guard_gain)
            y1[i] *= self.guard_gain

        return y1, y2, triggered

    def process_audio(self, mic1: np.ndarray, mic2: np.ndarray,
                      apply_guard: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
        """Process arbitrary-length audio using vectorized linear filtering.

        The DC blocker and HPF are vectorized with scipy.signal.lfilter; the
        impulse guard still evaluates frame-by-frame, but its gain envelope is
        generated vectorially. This avoids tens of thousands of Python sample
        loops per 3-second training example while preserving the reference
        recurrence.
        """
        from scipy.signal import lfilter

        mic1 = np.asarray(mic1, dtype=np.float32)
        mic2 = np.asarray(mic2, dtype=np.float32)
        n = min(len(mic1), len(mic2))
        mic1 = mic1[:n]
        mic2 = mic2[:n]

        if n == 0:
            return mic1.copy(), mic2.copy(), 0

        # Exact causal DC blocker: y[n] = x[n] - x[n-1] + a*y[n-1].
        dc_b = np.array([1.0, -1.0], dtype=np.float64)
        dc_a = np.array([1.0, -DC_BLOCK_COEFF], dtype=np.float64)
        y1_dc = lfilter(dc_b, dc_a, mic1.astype(np.float64))
        y2_dc = lfilter(dc_b, dc_a, mic2.astype(np.float64))

        b0, b1, b2, a1, a2 = self.hp_b
        hp_b = np.array([b0, b1, b2], dtype=np.float64)
        hp_a = np.array([1.0, a1, a2], dtype=np.float64)
        y1 = lfilter(hp_b, hp_a, y1_dc).astype(np.float32)
        y2 = lfilter(hp_b, hp_a, y2_dc).astype(np.float32)

        if not apply_guard:
            return y1, y2, 0

        self.energy_ema = 1e-8
        self.energy_ema_initialized = False
        self.guard_gain = 1.0
        self.trigger_count = 0
        self.guard_target = 1.0

        ratio = 10.0 ** (IMPULSE_THRESHOLD_DB / 10.0)
        frame_size = FRAME_SIZE
        out = y1.copy()

        for start in range(0, n, frame_size):
            end = min(start + frame_size, n)
            frame = y1[start:end]
            energy = float(np.mean(frame * frame)) if len(frame) else 0.0
            peak = float(np.max(np.abs(frame))) if len(frame) else 0.0
            prev_ema = self.energy_ema

            if not self.energy_ema_initialized:
                triggered = False
                self.energy_ema = max(energy, 1e-12)
                self.energy_ema_initialized = True
            else:
                relative_trigger = energy > max(prev_ema, 1e-12) * ratio
                peak_trigger = peak > IMPULSE_PEAK_THRESHOLD
                triggered = bool(relative_trigger or peak_trigger)
                self.energy_ema = ((1.0 - IMPULSE_EMA_ALPHA) * max(prev_ema, 1e-12)
                                   + IMPULSE_EMA_ALPHA * max(energy, 1e-12))

            if triggered:
                self.trigger_count += 1
                self.guard_target = 10.0 ** (-IMPULSE_MAX_ATTENUATION_DB / 20.0)
            else:
                self.guard_target = 1.0

            alpha = self._attack if self.guard_target < self.guard_gain else self._release
            k = np.arange(1, end - start + 1, dtype=np.float64)
            decay = np.power(1.0 - alpha, k)
            gains = self.guard_target + (self.guard_gain - self.guard_target) * decay
            out[start:end] *= gains.astype(np.float32)
            self.guard_gain = float(gains[-1]) if len(gains) else self.guard_gain

        return out, y2, self.trigger_count



def front_end_single(audio: np.ndarray) -> np.ndarray:
    """Convenience helper when only one microphone is available."""
    zeros = np.zeros_like(audio, dtype=np.float32)
    fe = MicFrontEnd()
    out, _, _ = fe.process_audio(np.asarray(audio, dtype=np.float32), zeros)
    return out


def front_end_filters_only(audio: np.ndarray) -> np.ndarray:
    """Apply only the causal DC block + HPF, with no impulse guard."""
    zeros = np.zeros_like(audio, dtype=np.float32)
    fe = MicFrontEnd()
    out, _, _ = fe.process_audio(np.asarray(audio, dtype=np.float32), zeros, apply_guard=False)
    return out
