from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_KEYS = (
    "dt",
    "time_sec",
    "reward",
    "theta",
    "theta_dot",
    "imu_accel",
    "torque_cmd",
    "disturbance_torque",
)


@dataclass
class RolloutSummary:
    path: str
    steps: int
    dt: float
    accel_rms: float
    pos_rms: float
    torque_rms: float


def _load_rollout(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    rollout = {key: np.asarray(data[key]) for key in REQUIRED_KEYS}
    _validate_shapes(path, rollout)
    return rollout


def _validate_shapes(path: Path, rollout: dict[str, np.ndarray]) -> None:
    t = rollout["time_sec"]
    if t.ndim != 1:
        raise ValueError(f"{path}: expected time_sec shape [T], got {t.shape}")
    steps = t.shape[0]
    reward = rollout["reward"]
    if reward.shape != (steps,):
        raise ValueError(f"{path}: expected reward shape [{steps}], got {reward.shape}")
    for key in ("theta", "theta_dot", "imu_accel", "torque_cmd", "disturbance_torque"):
        arr = rollout[key]
        if arr.shape != (steps, 3):
            raise ValueError(f"{path}: expected {key} shape [{steps}, 3], got {arr.shape}")
    dt_arr = np.asarray(rollout["dt"]).squeeze()
    if dt_arr.ndim != 0:
        raise ValueError(f"{path}: expected dt scalar, got shape {rollout['dt'].shape}")


def rollout_summary(path: str | Path) -> RolloutSummary:
    p = Path(path)
    r = _load_rollout(p)
    steps = int(r["time_sec"].shape[0])
    dt = float(np.asarray(r["dt"]).squeeze())
    accel_rms = float(np.sqrt(np.mean(np.square(r["imu_accel"]))))
    pos_rms = float(np.sqrt(np.mean(np.square(r["theta"]))))
    torque_rms = float(np.sqrt(np.mean(np.square(r["torque_cmd"]))))
    return RolloutSummary(
        path=str(p),
        steps=steps,
        dt=dt,
        accel_rms=accel_rms,
        pos_rms=pos_rms,
        torque_rms=torque_rms,
    )


def merge_rollouts(
    inputs: list[str | Path],
    output: str | Path,
    dt_tolerance: float = 1e-9,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("No rollout files provided.")

    paths = [Path(p) for p in inputs]
    rollouts = [_load_rollout(p) for p in paths]
    dt_values = [float(np.asarray(r["dt"]).squeeze()) for r in rollouts]
    ref_dt = dt_values[0]
    for idx, dt in enumerate(dt_values):
        if abs(dt - ref_dt) > dt_tolerance:
            raise ValueError(
                f"dt mismatch: {paths[idx]} has {dt}, expected {ref_dt} +/- {dt_tolerance}"
            )

    merged: dict[str, np.ndarray] = {}
    for key in REQUIRED_KEYS:
        if key == "dt":
            merged[key] = np.array(ref_dt, dtype=np.float32)
            continue
        merged[key] = np.concatenate([r[key] for r in rollouts], axis=0)

    episode_id = []
    step_in_episode = []
    for ep_idx, r in enumerate(rollouts):
        steps = int(r["time_sec"].shape[0])
        episode_id.append(np.full((steps,), ep_idx, dtype=np.int32))
        step_in_episode.append(np.arange(steps, dtype=np.int32))
    merged["episode_id"] = np.concatenate(episode_id, axis=0)
    merged["step_in_episode"] = np.concatenate(step_in_episode, axis=0)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **merged)

    summaries = [rollout_summary(p) for p in paths]
    total_steps = int(sum(s.steps for s in summaries))
    report = {
        "output": str(out),
        "rollouts": len(paths),
        "dt": ref_dt,
        "total_steps": total_steps,
        "duration_sec": float(total_steps * ref_dt),
        "input_summaries": [s.__dict__ for s in summaries],
    }
    return report
