from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DigitalTwinConfig:
    dt: float
    inertia: np.ndarray
    damping: np.ndarray
    stiffness: np.ndarray
    torque_limit: np.ndarray
    target_angles: np.ndarray
    lever_arm_m: float
    imu_noise_std: float


@dataclass
class DigitalTwinState:
    theta: np.ndarray
    theta_dot: np.ndarray
    theta_ddot: np.ndarray
    imu_accel: np.ndarray


class SimpleGimbalTwin:
    def __init__(self, config: DigitalTwinConfig, seed: int | None = None) -> None:
        self.base_config = config
        self.rng = np.random.default_rng(seed)
        self.inertia = config.inertia.copy()
        self.damping = config.damping.copy()
        self.stiffness = config.stiffness.copy()
        self.target_angles = config.target_angles.copy()
        self.torque_limit = config.torque_limit.copy()
        self.state = DigitalTwinState(
            theta=np.zeros(3, dtype=np.float32),
            theta_dot=np.zeros(3, dtype=np.float32),
            theta_ddot=np.zeros(3, dtype=np.float32),
            imu_accel=np.zeros(3, dtype=np.float32),
        )

    def set_seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(
        self,
        payload_mass_scale: float,
        mount_stiffness_scale: float,
        damping_scale: float,
        initial_theta: np.ndarray,
        initial_theta_dot: np.ndarray,
    ) -> DigitalTwinState:
        self.inertia = self.base_config.inertia * payload_mass_scale
        self.stiffness = self.base_config.stiffness * mount_stiffness_scale
        self.damping = self.base_config.damping * damping_scale
        self.state.theta = initial_theta.astype(np.float32)
        self.state.theta_dot = initial_theta_dot.astype(np.float32)
        self.state.theta_ddot[:] = 0.0
        self.state.imu_accel[:] = 0.0
        return self.copy_state()

    def step(self, torque_cmd: np.ndarray, disturbance_torque: np.ndarray) -> DigitalTwinState:
        torque_cmd = np.clip(torque_cmd, -self.torque_limit, self.torque_limit)
        restoring = self.stiffness * (self.state.theta - self.target_angles)
        net_torque = (
            torque_cmd
            + disturbance_torque
            - self.damping * self.state.theta_dot
            - restoring
        )
        self.state.theta_ddot = (net_torque / self.inertia).astype(np.float32)
        self.state.theta_dot = self.state.theta_dot + self.state.theta_ddot * self.base_config.dt
        self.state.theta = self.state.theta + self.state.theta_dot * self.base_config.dt
        self.state.imu_accel = (
            self.base_config.lever_arm_m * self.state.theta_ddot
            + self.rng.normal(0.0, self.base_config.imu_noise_std, size=3).astype(np.float32)
        )
        return self.copy_state()

    def copy_state(self) -> DigitalTwinState:
        return DigitalTwinState(
            theta=self.state.theta.copy(),
            theta_dot=self.state.theta_dot.copy(),
            theta_ddot=self.state.theta_ddot.copy(),
            imu_accel=self.state.imu_accel.copy(),
        )
