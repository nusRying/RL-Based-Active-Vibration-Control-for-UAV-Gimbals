from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimObservation:
    theta: np.ndarray
    theta_dot: np.ndarray
    imu_accel: np.ndarray


class RosSimBridgeStub:
    """
    Replace this stub with a ROS2/Gazebo or Isaac bridge.

    Expected behavior:
    - read_observation(): returns gimbal angles, rates, and payload IMU acceleration.
    - write_torque_command(): sends 3-axis torque/current command to low-level controller.
    """

    def read_observation(self) -> SimObservation:
        raise NotImplementedError("Implement bridge read path for your simulator.")

    def write_torque_command(self, torque_nm: np.ndarray) -> None:
        raise NotImplementedError("Implement bridge write path for your simulator.")
