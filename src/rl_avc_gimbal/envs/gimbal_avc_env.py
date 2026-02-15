from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_avc_gimbal.config import deep_merge
from rl_avc_gimbal.controllers import PIDGains, VectorPID
from rl_avc_gimbal.envs.disturbance import DisturbanceConfig, HarmonicDisturbance
from rl_avc_gimbal.sim import DigitalTwinConfig, SimpleGimbalTwin

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "dt": 0.005,
    "episode_seconds": 20,
    "history_length": 8,
    "termination_max_angle_rad": 1.2,
    "imu_noise_std": 0.01,
    "torque_limit_nm": [0.6, 0.6, 0.4],
    "inertia_kgm2": [0.012, 0.012, 0.008],
    "damping": [0.09, 0.09, 0.07],
    "stiffness": [0.8, 0.8, 0.5],
    "target_angles_rad": [0.0, 0.0, 0.0],
    "lever_arm_m": 0.08,
    "weights": {"accel": 1.0, "position": 0.25, "chatter": 0.01},
    "disturbance": {
        "enabled": True,
        "frequency_hz": [100.0, 400.0],
        "amplitude": [0.2, 0.2, 0.15],
        "gaussian_std": 0.02,
    },
    "domain_randomization": {
        "payload_mass_kg": [0.25, 0.75],
        "mount_stiffness_scale": [0.8, 1.2],
        "damping_scale": [0.9, 1.1],
    },
    "hybrid_control": {
        "enabled": False,
        "pid_kp": [4.0, 4.0, 3.0],
        "pid_ki": [0.2, 0.2, 0.2],
        "pid_kd": [0.1, 0.1, 0.08],
        "integral_limit": [0.4, 0.4, 0.3],
        "rl_delta_scale": [0.25, 0.25, 0.25],
    },
}


def _vec3(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"Expected vector length 3, got shape {arr.shape}.")
    return arr


class GimbalAVCEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = deep_merge(DEFAULT_CONFIG, config or {})

        self.dt = float(self.config["dt"])
        self.history_length = int(self.config["history_length"])
        self.episode_steps = int(float(self.config["episode_seconds"]) / self.dt)
        self.max_angle = float(self.config["termination_max_angle_rad"])
        self.torque_limit = _vec3(self.config["torque_limit_nm"])
        self.target_angles = _vec3(self.config["target_angles_rad"])

        twin_config = DigitalTwinConfig(
            dt=self.dt,
            inertia=_vec3(self.config["inertia_kgm2"]),
            damping=_vec3(self.config["damping"]),
            stiffness=_vec3(self.config["stiffness"]),
            torque_limit=self.torque_limit,
            target_angles=self.target_angles,
            lever_arm_m=float(self.config["lever_arm_m"]),
            imu_noise_std=float(self.config["imu_noise_std"]),
        )
        self.twin = SimpleGimbalTwin(twin_config, seed=int(self.config["seed"]))

        disturbance_cfg = self.config["disturbance"]
        self.disturbance = HarmonicDisturbance(
            DisturbanceConfig(
                enabled=bool(disturbance_cfg["enabled"]),
                frequency_hz=tuple(disturbance_cfg["frequency_hz"]),
                amplitude=_vec3(disturbance_cfg["amplitude"]),
                gaussian_std=float(disturbance_cfg["gaussian_std"]),
            ),
            seed=int(self.config["seed"]),
        )

        self.weights = self.config["weights"]
        self.randomization = self.config["domain_randomization"]
        self.hybrid_cfg = self.config["hybrid_control"]
        self.pid: VectorPID | None = None
        self.rl_delta_scale = _vec3(self.hybrid_cfg["rl_delta_scale"])
        if bool(self.hybrid_cfg["enabled"]):
            gains = PIDGains(
                kp=_vec3(self.hybrid_cfg["pid_kp"]),
                ki=_vec3(self.hybrid_cfg["pid_ki"]),
                kd=_vec3(self.hybrid_cfg["pid_kd"]),
                integral_limit=_vec3(self.hybrid_cfg["integral_limit"]),
            )
            self.pid = VectorPID(gains=gains, dt=self.dt)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        obs_dim = self.history_length * 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.history: deque[np.ndarray] = deque(maxlen=self.history_length)
        self.t_sec = 0.0
        self.step_count = 0
        self.prev_torque = np.zeros(3, dtype=np.float32)
        self.last_seed = int(self.config["seed"])
        self.rng = np.random.default_rng(self.last_seed)

    def _sample_domain(self) -> tuple[float, float, float]:
        payload_min, payload_max = self.randomization["payload_mass_kg"]
        mass = float(self.rng.uniform(payload_min, payload_max))
        ref_mass = (payload_min + payload_max) / 2.0
        payload_scale = max(0.2, mass / ref_mass)
        stiff_lo, stiff_hi = self.randomization["mount_stiffness_scale"]
        damp_lo, damp_hi = self.randomization["damping_scale"]
        stiffness_scale = float(self.rng.uniform(stiff_lo, stiff_hi))
        damping_scale = float(self.rng.uniform(damp_lo, damp_hi))
        return payload_scale, stiffness_scale, damping_scale

    def _build_frame(self) -> np.ndarray:
        state = self.twin.state
        return np.concatenate([state.theta, state.theta_dot, state.imu_accel]).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(list(self.history), axis=0).astype(np.float32)

    def _resolve_torque(self, action: np.ndarray) -> np.ndarray:
        if self.pid is None:
            return action * self.torque_limit
        error = self.target_angles - self.twin.state.theta
        pid_torque = self.pid.step(error)
        delta_torque = action * self.torque_limit * self.rl_delta_scale
        return np.clip(pid_torque + delta_torque, -self.torque_limit, self.torque_limit)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.last_seed = seed
        self.rng = np.random.default_rng(self.last_seed)
        self.twin.set_seed(self.last_seed)
        self.disturbance.set_seed(self.last_seed + 1)
        self.disturbance.reset()
        if self.pid is not None:
            self.pid.reset()

        payload_scale, stiffness_scale, damping_scale = self._sample_domain()
        init_theta = self.rng.normal(0.0, 0.03, size=3).astype(np.float32)
        init_theta_dot = self.rng.normal(0.0, 0.10, size=3).astype(np.float32)
        self.twin.reset(
            payload_mass_scale=payload_scale,
            mount_stiffness_scale=stiffness_scale,
            damping_scale=damping_scale,
            initial_theta=init_theta,
            initial_theta_dot=init_theta_dot,
        )

        self.history.clear()
        frame = self._build_frame()
        for _ in range(self.history_length):
            self.history.append(frame.copy())

        self.t_sec = 0.0
        self.step_count = 0
        self.prev_torque[:] = 0.0
        info = {
            "payload_scale": payload_scale,
            "stiffness_scale": stiffness_scale,
            "damping_scale": damping_scale,
        }
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        torque_cmd = self._resolve_torque(action)
        disturbance_torque = self.disturbance.sample(self.t_sec)
        state = self.twin.step(torque_cmd, disturbance_torque)

        self.history.append(self._build_frame())
        self.t_sec += self.dt
        self.step_count += 1

        accel_term = float(np.mean(state.imu_accel ** 2))
        pos_term = float(np.mean((state.theta - self.target_angles) ** 2))
        chatter_term = float(np.mean((torque_cmd - self.prev_torque) ** 2))
        reward = -(
            float(self.weights["accel"]) * accel_term
            + float(self.weights["position"]) * pos_term
            + float(self.weights["chatter"]) * chatter_term
        )

        self.prev_torque = torque_cmd.copy()
        terminated = bool(np.any(np.abs(state.theta) > self.max_angle))
        truncated = self.step_count >= self.episode_steps
        info = {
            "accel_term": accel_term,
            "pos_term": pos_term,
            "chatter_term": chatter_term,
            "disturbance_torque": disturbance_torque,
            "torque_cmd": torque_cmd,
            "theta": state.theta.copy(),
            "theta_dot": state.theta_dot.copy(),
            "imu_accel": state.imu_accel.copy(),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
