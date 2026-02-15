from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDGains:
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    integral_limit: np.ndarray | None = None


class VectorPID:
    def __init__(self, gains: PIDGains, dt: float) -> None:
        self.gains = gains
        self.dt = dt
        self.integral = np.zeros(3, dtype=np.float32)
        self.prev_error = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0

    def step(self, error: np.ndarray) -> np.ndarray:
        self.integral += error * self.dt
        if self.gains.integral_limit is not None:
            self.integral = np.clip(
                self.integral, -self.gains.integral_limit, self.gains.integral_limit
            )
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error.copy()
        return (
            self.gains.kp * error
            + self.gains.ki * self.integral
            + self.gains.kd * derivative
        )
