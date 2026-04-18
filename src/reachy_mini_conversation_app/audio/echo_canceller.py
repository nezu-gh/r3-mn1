"""Acoustic echo cancellation for barge-in.

Subtracts the robot's own TTS playback from the mic signal so Silero VAD
can detect real human speech even while the robot is talking. Pure-numpy
NLMS adaptive filter — no external deps, runs on aarch64 RPi.

Adapted from algal/sparky's `echo_canceller.py`
(https://github.com/algal/sparky). See that repo for the original design.
"""

from __future__ import annotations

import logging
import threading

import numpy as np


logger = logging.getLogger(__name__)


class AcousticEchoCanceller:
    """Thread-safe NLMS AEC.

    The speaker reference is buffered and consumed frame-by-frame as the
    microphone signal is processed. When no speaker reference is available
    (robot is silent), mic audio passes through unchanged.
    """

    def __init__(
        self,
        frame_size: int = 160,
        filter_length: int = 3200,
        sample_rate: int = 16000,
        mu: float = 0.3,
    ) -> None:
        """Initialise the AEC.

        Args:
            frame_size: Samples per AEC frame. 160 = 10 ms at 16 kHz.
            filter_length: NLMS filter taps. 3200 ≈ 200 ms echo tail at 16 kHz.
            sample_rate: Must match mic and speaker rate (16 kHz).
            mu: NLMS step size in (0, 1]. Lower = more stable, higher = faster
                adaptation. 0.3 is a reasonable starting point.
        """
        self._frame_size = frame_size
        self._filter_length = filter_length
        self._sample_rate = sample_rate
        self._mu = mu

        self._w = np.zeros(filter_length, dtype=np.float64)
        self._x_hist = np.zeros(filter_length, dtype=np.float64)

        self._speaker_buf = bytearray()
        self._lock = threading.Lock()

        self._frames_processed = 0
        self._frames_with_ref = 0

    def feed_speaker_pcm(self, pcm_int16: bytes) -> None:
        """Queue 16 kHz int16 mono PCM from the speaker path as the reference."""
        with self._lock:
            self._speaker_buf.extend(pcm_int16)

    def process_mic_chunk(self, mic_pcm: bytes) -> bytes:
        """Filter a mic chunk of 16 kHz int16 mono PCM. Length-preserving."""
        frame_bytes = self._frame_size * 2
        if len(mic_pcm) < frame_bytes:
            return mic_pcm

        result = bytearray()
        offset = 0
        while offset + frame_bytes <= len(mic_pcm):
            result.extend(self._process_frame(mic_pcm[offset : offset + frame_bytes]))
            offset += frame_bytes
        if offset < len(mic_pcm):
            result.extend(mic_pcm[offset:])
        return bytes(result)

    def _process_frame(self, mic_frame: bytes) -> bytes:
        frame_bytes = self._frame_size * 2
        self._frames_processed += 1

        with self._lock:
            if len(self._speaker_buf) >= frame_bytes:
                speaker_data: bytes | None = bytes(self._speaker_buf[:frame_bytes])
                del self._speaker_buf[:frame_bytes]
                has_ref = True
            else:
                speaker_data = None
                has_ref = False

        mic = np.frombuffer(mic_frame, dtype=np.int16).astype(np.float64)

        if not has_ref:
            return mic_frame

        self._frames_with_ref += 1
        speaker = np.frombuffer(speaker_data, dtype=np.int16).astype(np.float64)

        output = np.zeros(self._frame_size, dtype=np.float64)
        for i in range(self._frame_size):
            self._x_hist = np.roll(self._x_hist, 1)
            self._x_hist[0] = speaker[i]
            echo_est = np.dot(self._w, self._x_hist)
            error = mic[i] - echo_est
            output[i] = error
            norm = np.dot(self._x_hist, self._x_hist) + 1e-6
            self._w += (self._mu * error / norm) * self._x_hist

        return np.clip(output, -32768, 32767).astype(np.int16).tobytes()

    def clear_buffer(self) -> None:
        """Drop queued reference audio (e.g. after barge-in drained the speaker).

        Keeps the learned filter weights — the speaker→mic echo path is a
        property of the hardware and doesn't change between turns, so
        resetting weights would force a re-learn on every new bot response.
        """
        with self._lock:
            self._speaker_buf.clear()

    def clear(self) -> None:
        """Full reset: buffer + filter weights (use sparingly, e.g. on reconfig)."""
        with self._lock:
            self._speaker_buf.clear()
        self._w[:] = 0.0
        self._x_hist[:] = 0.0

    @property
    def stats(self) -> dict:
        """Counts and filter energy — useful for debugging convergence."""
        return {
            "frames_processed": self._frames_processed,
            "frames_with_ref": self._frames_with_ref,
            "speaker_buf_bytes": len(self._speaker_buf),
            "filter_energy": float(np.dot(self._w, self._w)),
        }
