from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DisturbanceConfig:
    enabled: bool
    frequency_hz: tuple[float, float]
    amplitude: np.ndarray
    gaussian_std: float


class HarmonicDisturbance:
    def __init__(self, config: DisturbanceConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.freq_hz = np.zeros(3, dtype=np.float32)
        self.phase = np.zeros(3, dtype=np.float32)
        self.reset()

    def set_seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        lo, hi = self.config.frequency_hz
        self.freq_hz = self.rng.uniform(lo, hi, size=3).astype(np.float32)
        self.phase = self.rng.uniform(0.0, 2.0 * np.pi, size=3).astype(np.float32)

    def sample(self, t_sec: float) -> np.ndarray:
        if not self.config.enabled:
            return np.zeros(3, dtype=np.float32)
        harmonic = self.config.amplitude * np.sin(2.0 * np.pi * self.freq_hz * t_sec + self.phase)
        noise = self.rng.normal(0.0, self.config.gaussian_std, size=3).astype(np.float32)
        return harmonic.astype(np.float32) + noise
