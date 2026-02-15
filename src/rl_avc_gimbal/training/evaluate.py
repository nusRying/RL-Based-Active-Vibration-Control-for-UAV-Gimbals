from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from stable_baselines3 import DDPG, SAC

from rl_avc_gimbal.config import load_yaml
from rl_avc_gimbal.envs import GimbalAVCEnv


def _model_cls(algo: str):
    algo = algo.lower()
    if algo == "sac":
        return SAC
    if algo == "ddpg":
        return DDPG
    raise ValueError(f"Unsupported algo '{algo}'.")


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(mean(values)), float(pstdev(values) if len(values) > 1 else 0.0)


def evaluate(
    algo: str,
    model_path: str | Path,
    env_config: str | Path,
    episodes: int,
    seed: int = 42,
    save_json: str | Path | None = None,
    record_first_episode: str | Path | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    env = GimbalAVCEnv(load_yaml(env_config))
    model = _model_cls(algo).load(str(model_path))

    episode_metrics: list[dict[str, Any]] = []
    terminations = 0
    truncations = 0

    rollout: dict[str, list[Any]] | None = None
    if record_first_episode is not None:
        rollout = {
            "time_sec": [],
            "reward": [],
            "theta": [],
            "theta_dot": [],
            "imu_accel": [],
            "torque_cmd": [],
            "disturbance_torque": [],
        }

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_accel_sq: list[float] = []
        ep_pos_sq: list[float] = []
        ep_chatter_sq: list[float] = []
        peak_abs_angle = 0.0
        peak_abs_torque = 0.0
        steps = 0
        ep_terminated = False
        ep_truncated = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_accel_sq.append(float(info["accel_term"]))
            ep_pos_sq.append(float(info["pos_term"]))
            ep_chatter_sq.append(float(info["chatter_term"]))
            peak_abs_angle = max(peak_abs_angle, float(np.max(np.abs(info["theta"]))))
            peak_abs_torque = max(peak_abs_torque, float(np.max(np.abs(info["torque_cmd"]))))

            if rollout is not None and ep == 0:
                rollout["time_sec"].append(float(steps * env.dt))
                rollout["reward"].append(float(reward))
                rollout["theta"].append(np.asarray(info["theta"], dtype=np.float32))
                rollout["theta_dot"].append(np.asarray(info["theta_dot"], dtype=np.float32))
                rollout["imu_accel"].append(np.asarray(info["imu_accel"], dtype=np.float32))
                rollout["torque_cmd"].append(np.asarray(info["torque_cmd"], dtype=np.float32))
                rollout["disturbance_torque"].append(
                    np.asarray(info["disturbance_torque"], dtype=np.float32)
                )

            steps += 1
            done = terminated or truncated
            ep_terminated = bool(terminated)
            ep_truncated = bool(truncated)

        terminations += int(ep_terminated)
        truncations += int(ep_truncated)
        episode_metrics.append(
            {
                "episode": ep,
                "return": ep_return,
                "accel_rms": float(np.sqrt(max(np.mean(ep_accel_sq), 0.0))),
                "pos_rms": float(np.sqrt(max(np.mean(ep_pos_sq), 0.0))),
                "chatter_rms": float(np.sqrt(max(np.mean(ep_chatter_sq), 0.0))),
                "peak_abs_angle_rad": peak_abs_angle,
                "peak_abs_torque_nm": peak_abs_torque,
                "steps": steps,
                "terminated": ep_terminated,
                "truncated": ep_truncated,
            }
        )

    returns = [float(ep["return"]) for ep in episode_metrics]
    accel_rms = [float(ep["accel_rms"]) for ep in episode_metrics]
    pos_rms = [float(ep["pos_rms"]) for ep in episode_metrics]
    chatter_rms = [float(ep["chatter_rms"]) for ep in episode_metrics]
    peak_abs_angle = [float(ep["peak_abs_angle_rad"]) for ep in episode_metrics]
    peak_abs_torque = [float(ep["peak_abs_torque_nm"]) for ep in episode_metrics]

    return_mean, return_std = _mean_std(returns)
    accel_mean, accel_std = _mean_std(accel_rms)
    pos_mean, pos_std = _mean_std(pos_rms)
    chatter_mean, chatter_std = _mean_std(chatter_rms)
    peak_angle_mean, peak_angle_std = _mean_std(peak_abs_angle)
    peak_torque_mean, peak_torque_std = _mean_std(peak_abs_torque)

    summary = {
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "return_mean": return_mean,
        "return_std": return_std,
        "accel_rms_mean": accel_mean,
        "accel_rms_std": accel_std,
        "pos_rms_mean": pos_mean,
        "pos_rms_std": pos_std,
        "chatter_rms_mean": chatter_mean,
        "chatter_rms_std": chatter_std,
        "peak_abs_angle_mean_rad": peak_angle_mean,
        "peak_abs_angle_std_rad": peak_angle_std,
        "peak_abs_torque_mean_nm": peak_torque_mean,
        "peak_abs_torque_std_nm": peak_torque_std,
        "terminated_episodes": terminations,
        "truncated_episodes": truncations,
        "termination_rate": float(terminations / max(episodes, 1)),
    }
    result = {"summary": summary, "episodes": episode_metrics}
    env.close()

    print(f"Episodes: {summary['episodes']}")
    print(f"Return mean/std: {summary['return_mean']:.3f} / {summary['return_std']:.3f}")
    print(
        f"Accel RMS mean/std: {summary['accel_rms_mean']:.5f} / {summary['accel_rms_std']:.5f}"
    )
    print(f"Pos RMS mean/std: {summary['pos_rms_mean']:.5f} / {summary['pos_rms_std']:.5f}")
    print(
        f"Chatter RMS mean/std: {summary['chatter_rms_mean']:.5f} / {summary['chatter_rms_std']:.5f}"
    )
    print(
        "Peak |angle| mean/std (rad): "
        f"{summary['peak_abs_angle_mean_rad']:.5f} / {summary['peak_abs_angle_std_rad']:.5f}"
    )
    print(
        "Peak |torque| mean/std (Nm): "
        f"{summary['peak_abs_torque_mean_nm']:.5f} / {summary['peak_abs_torque_std_nm']:.5f}"
    )
    print(f"Terminated early: {terminations}, Truncated by horizon: {truncations}")

    if save_json is not None:
        out_path = Path(save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote evaluation JSON: {out_path}")

    if rollout is not None and record_first_episode is not None:
        rollout_path = Path(record_first_episode)
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(rollout_path),
            dt=np.array(env.dt, dtype=np.float32),
            time_sec=np.asarray(rollout["time_sec"], dtype=np.float32),
            reward=np.asarray(rollout["reward"], dtype=np.float32),
            theta=np.asarray(rollout["theta"], dtype=np.float32),
            theta_dot=np.asarray(rollout["theta_dot"], dtype=np.float32),
            imu_accel=np.asarray(rollout["imu_accel"], dtype=np.float32),
            torque_cmd=np.asarray(rollout["torque_cmd"], dtype=np.float32),
            disturbance_torque=np.asarray(rollout["disturbance_torque"], dtype=np.float32),
        )
        print(f"Wrote rollout NPZ: {rollout_path}")

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC/DDPG model.")
    parser.add_argument("--algo", choices=["sac", "ddpg"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--record-first-episode", default=None)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action sampling when supported by the model.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate(
        algo=args.algo,
        model_path=args.model_path,
        env_config=args.env_config,
        episodes=args.episodes,
        seed=args.seed,
        save_json=args.save_json,
        record_first_episode=args.record_first_episode,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
