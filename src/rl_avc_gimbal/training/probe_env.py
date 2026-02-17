from __future__ import annotations

import argparse
from statistics import mean, pstdev

import numpy as np

from rl_avc_gimbal.config import load_yaml
from rl_avc_gimbal.envs import GimbalAVCEnv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe environment episode completion with random/zero actions."
    )
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy",
        choices=["random", "zero"],
        default="random",
        help="Action source for probing.",
    )
    return parser


def _sample_action(env: GimbalAVCEnv, policy: str) -> np.ndarray:
    if policy == "zero":
        return np.zeros(env.action_space.shape, dtype=np.float32)
    return env.action_space.sample().astype(np.float32)


def main() -> None:
    args = build_arg_parser().parse_args()
    env = GimbalAVCEnv(load_yaml(args.env_config))

    lengths: list[int] = []
    returns: list[float] = []
    terminations = 0
    truncations = 0
    done_reasons: dict[str, int] = {}

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        _ = obs
        ep_return = 0.0
        steps = 0
        done = False
        reason = "unknown"
        while not done:
            action = _sample_action(env, args.policy)
            _, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            reason = str(info.get("done_reason", "unknown"))
            if terminated:
                terminations += 1
            if truncated:
                truncations += 1

        lengths.append(steps)
        returns.append(ep_return)
        done_reasons[reason] = done_reasons.get(reason, 0) + 1
        print(
            f"episode={ep:03d} steps={steps:5d} return={ep_return:10.3f} "
            f"terminated={terminated} truncated={truncated} reason={reason}"
        )

    env.close()
    length_mean = mean(lengths) if lengths else 0.0
    length_std = pstdev(lengths) if len(lengths) > 1 else 0.0
    return_mean = mean(returns) if returns else 0.0
    return_std = pstdev(returns) if len(returns) > 1 else 0.0
    print("")
    print(f"Episodes: {args.episodes}")
    print(f"Length mean/std: {length_mean:.2f} / {length_std:.2f}")
    print(f"Return mean/std: {return_mean:.3f} / {return_std:.3f}")
    print(f"Terminations: {terminations}, Truncations: {truncations}")
    print(f"Done reasons: {done_reasons}")


if __name__ == "__main__":
    main()
