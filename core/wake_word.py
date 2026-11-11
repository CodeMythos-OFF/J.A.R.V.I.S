"""Local wake-word detection for J.A.R.V.I.S.

Audio is processed on-device while asleep. Only frames after detection are
forwarded to the active Gemini Live session.
"""
from __future__ import annotations

import numpy as np


class WakeWordDetector:
    """Small adapter around openWakeWord's pretrained Hey JARVIS model."""

    def __init__(self, threshold: float = 0.55):
        from openwakeword.model import Model

        self.threshold = float(threshold)
        self.model = Model(
            wakeword_models=["hey_jarvis"],
            inference_framework="onnx",
        )
        self._pending = np.empty(0, dtype=np.int16)

    def process(self, samples) -> bool:
        """Consume mono 16 kHz int16 samples and return True on a wake hit."""
        audio = np.asarray(samples, dtype=np.int16).reshape(-1)
        if audio.size == 0:
            return False

        self._pending = np.concatenate((self._pending, audio))
        hit = False
        frame_size = 1280  # openWakeWord expects 80 ms at 16 kHz
        while self._pending.size >= frame_size:
            frame = self._pending[:frame_size]
            self._pending = self._pending[frame_size:]
            scores = self.model.predict(frame)
            if scores and max(float(v) for v in scores.values()) >= self.threshold:
                hit = True
                self._pending = np.empty(0, dtype=np.int16)
                break
        return hit
