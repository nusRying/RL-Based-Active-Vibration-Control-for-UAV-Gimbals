from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort


@dataclass
class Telemetry:
    theta: np.ndarray
    theta_dot: np.ndarray
    imu_accel: np.ndarray


class MavlinkPolicyNode:
    """
    Runtime stub for policy inference loop.

    Replace read_telemetry/send_torque with MAVLink or direct serial integration.
    """

    def __init__(self, model_path: str, history_length: int = 8, control_rate_hz: int = 200) -> None:
        self.history_length = history_length
        self.control_rate_hz = control_rate_hz
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.history: deque[np.ndarray] = deque(maxlen=history_length)

    def _build_obs(self, telemetry: Telemetry) -> np.ndarray:
        frame = np.concatenate([telemetry.theta, telemetry.theta_dot, telemetry.imu_accel]).astype(
            np.float32
        )
        if not self.history:
            for _ in range(self.history_length):
                self.history.append(frame.copy())
        else:
            self.history.append(frame)
        obs = np.concatenate(list(self.history), axis=0).astype(np.float32)
        return obs[None, :]

    def read_telemetry(self) -> Telemetry:
        raise NotImplementedError("Connect this to Pixhawk/MAVLink telemetry stream.")

    def send_torque(self, action: np.ndarray) -> None:
        raise NotImplementedError("Send torque/current command to motor controller.")

    def run_forever(self) -> None:
        dt = 1.0 / self.control_rate_hz
        while True:
            t0 = time.perf_counter()
            telemetry = self.read_telemetry()
            obs = self._build_obs(telemetry)
            action = self.session.run(None, {self.input_name: obs})[0][0]
            self.send_torque(action.astype(np.float32))
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ONNX runtime MAVLink policy loop (stub).")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--rate-hz", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    node = MavlinkPolicyNode(
        model_path=args.model_path,
        history_length=args.history_length,
        control_rate_hz=args.rate_hz,
    )
    node.run_forever()


if __name__ == "__main__":
    main()
