from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rl_avc_gimbal.config import ensure_dir, load_yaml
from rl_avc_gimbal.envs import GimbalAVCEnv
from rl_avc_gimbal.runtime_compat import prepare_runtime_compat

prepare_runtime_compat()
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def _as_train_freq(value: Any) -> tuple[int, str] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), str(value[1])
    return None


def _make_vec_env(env_cfg: dict[str, Any], n_envs: int, base_seed: int):
    def make_env(rank: int):
        def _init():
            env = GimbalAVCEnv(env_cfg)
            env.reset(seed=base_seed + rank)
            return env

        return _init

    return VecMonitor(DummyVecEnv([make_env(i) for i in range(n_envs)]))


def _build_model(algo_name: str, env, cfg: dict[str, Any], tensorboard_log: str):
    algo = algo_name.lower()
    train_freq = _as_train_freq(cfg.get("train_freq"))
    policy_kwargs = cfg.get("policy_kwargs", {"net_arch": [256, 256]})
    verbose = int(cfg.get("verbose", 1))
    device = str(cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 42))
    common_kwargs = dict(
        policy=str(cfg.get("policy", "MlpPolicy")),
        env=env,
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        buffer_size=int(cfg.get("buffer_size", 1_000_000)),
        learning_starts=int(cfg.get("learning_starts", 10_000)),
        batch_size=int(cfg.get("batch_size", 256)),
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.005)),
        train_freq=train_freq or (1, "step"),
        gradient_steps=int(cfg.get("gradient_steps", 1)),
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        device=device,
        seed=seed,
    )
    if algo == "sac":
        return SAC(**common_kwargs)
    if algo == "ddpg":
        n_actions = env.action_space.shape[-1]
        sigma = float(cfg.get("action_noise_sigma", 0.1))
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions)
        )
        common_kwargs["action_noise"] = action_noise
        return DDPG(**common_kwargs)
    raise ValueError(f"Unsupported algo '{algo_name}'. Use 'sac' or 'ddpg'.")


def train(
    algo_name: str,
    env_cfg_path: str | Path,
    algo_cfg_path: str | Path,
    run_dir: str | Path,
) -> None:
    env_cfg = load_yaml(env_cfg_path)
    algo_cfg = load_yaml(algo_cfg_path)
    run_path = ensure_dir(run_dir)
    ensure_dir(run_path / "checkpoints")
    ensure_dir(run_path / "best_model")
    ensure_dir(run_path / "eval")

    with (run_path / "resolved_env_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(env_cfg, f, sort_keys=False)
    with (run_path / "resolved_algo_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(algo_cfg, f, sort_keys=False)

    seed = int(algo_cfg.get("seed", env_cfg.get("seed", 42)))
    n_envs = int(algo_cfg.get("n_envs", 1))
    total_timesteps = int(algo_cfg.get("total_timesteps", 100_000))
    checkpoint_freq = int(algo_cfg.get("checkpoint_freq", 50_000))
    eval_freq = int(algo_cfg.get("eval_freq", 25_000))
    log_interval = int(algo_cfg.get("log_interval", 10))

    vec_env = _make_vec_env(env_cfg, n_envs=n_envs, base_seed=seed)
    eval_env = Monitor(GimbalAVCEnv(env_cfg))
    model = _build_model(algo_name, vec_env, algo_cfg, tensorboard_log=str(run_path / "tb"))

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(1, checkpoint_freq // n_envs),
                save_path=str(run_path / "checkpoints"),
                name_prefix=algo_name,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_path / "best_model"),
                log_path=str(run_path / "eval"),
                eval_freq=max(1, eval_freq // n_envs),
                deterministic=True,
                render=False,
            ),
        ]
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=log_interval,
        progress_bar=True,
    )
    model.save(str(run_path / "final_model"))
    vec_env.close()
    eval_env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAC/DDPG for gimbal AVC.")
    parser.add_argument("--algo", choices=["sac", "ddpg"], required=True)
    parser.add_argument("--env-config", required=True, help="Path to env YAML config.")
    parser.add_argument("--algo-config", required=True, help="Path to algo YAML config.")
    parser.add_argument("--run-dir", required=True, help="Output directory for artifacts.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(
        algo_name=args.algo,
        env_cfg_path=args.env_config,
        algo_cfg_path=args.algo_config,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
