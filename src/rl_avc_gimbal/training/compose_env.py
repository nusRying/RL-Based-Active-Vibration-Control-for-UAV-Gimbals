from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from rl_avc_gimbal.config import deep_merge, load_yaml


def _expect_vec3(cfg: dict[str, Any], key: str) -> None:
    if key not in cfg:
        return
    value = cfg[key]
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"'{key}' must be a list of length 3.")


def _validate_merged_env(cfg: dict[str, Any]) -> None:
    for key in [
        "torque_limit_nm",
        "inertia_kgm2",
        "damping",
        "stiffness",
        "target_angles_rad",
    ]:
        _expect_vec3(cfg, key)
    if "dt" in cfg and float(cfg["dt"]) <= 0.0:
        raise ValueError("'dt' must be > 0.")
    if "episode_seconds" in cfg and float(cfg["episode_seconds"]) <= 0.0:
        raise ValueError("'episode_seconds' must be > 0.")
    if "history_length" in cfg and int(cfg["history_length"]) < 1:
        raise ValueError("'history_length' must be >= 1.")


def compose_env_config(
    base_config_path: str | Path,
    profile_path: str | Path,
    output_path: str | Path,
) -> Path:
    base = load_yaml(base_config_path)
    profile = load_yaml(profile_path)
    merged = deep_merge(base, profile)
    _validate_merged_env(merged)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge base env config with hardware profile overrides."
    )
    parser.add_argument("--base", required=True, help="Base env config (e.g. configs/env.yaml).")
    parser.add_argument(
        "--profile",
        required=True,
        help="Override profile (e.g. configs/env_hw_profile.yaml).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output merged env config path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out = compose_env_config(
        base_config_path=args.base,
        profile_path=args.profile,
        output_path=args.output,
    )
    print(f"Wrote merged env config: {out}")


if __name__ == "__main__":
    main()
