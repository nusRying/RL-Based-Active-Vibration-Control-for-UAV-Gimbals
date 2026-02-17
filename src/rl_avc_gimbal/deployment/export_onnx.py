from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rl_avc_gimbal.config import ensure_dir, load_yaml
from rl_avc_gimbal.envs import GimbalAVCEnv
from rl_avc_gimbal.runtime_compat import prepare_runtime_compat

prepare_runtime_compat()
from stable_baselines3 import DDPG, SAC


class ActorWrapper(torch.nn.Module):
    def __init__(self, model, algo: str) -> None:
        super().__init__()
        self.model = model
        self.algo = algo.lower()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.algo == "sac":
            return self.model.policy.actor(obs, deterministic=True)
        return self.model.policy.actor(obs)


def _model_cls(algo: str):
    algo = algo.lower()
    if algo == "sac":
        return SAC
    if algo == "ddpg":
        return DDPG
    raise ValueError(f"Unsupported algo '{algo}'.")


def export_onnx(
    algo: str,
    model_path: str | Path,
    env_config: str | Path,
    output_path: str | Path,
    opset: int = 17,
) -> Path:
    env = GimbalAVCEnv(load_yaml(env_config))
    model = _model_cls(algo).load(str(model_path))
    wrapper = ActorWrapper(model=model, algo=algo).eval()

    obs_dim = env.observation_space.shape[0]
    dummy_input = torch.zeros((1, obs_dim), dtype=torch.float32)

    output = Path(output_path)
    ensure_dir(output.parent)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=opset,
        )
    env.close()
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trained RL policy to ONNX.")
    parser.add_argument("--algo", choices=["sac", "ddpg"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out = export_onnx(
        algo=args.algo,
        model_path=args.model_path,
        env_config=args.env_config,
        output_path=args.output,
        opset=args.opset,
    )
    print(f"Exported ONNX policy: {out}")


if __name__ == "__main__":
    main()
