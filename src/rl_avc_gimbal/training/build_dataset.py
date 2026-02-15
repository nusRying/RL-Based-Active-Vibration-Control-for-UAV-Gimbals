from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from rl_avc_gimbal.training.dataset_tools import merge_rollouts


def _collect_inputs(explicit: list[str], pattern: str | None) -> list[str]:
    inputs = list(explicit)
    if pattern:
        inputs.extend(sorted(glob.glob(pattern)))
    unique = []
    seen = set()
    for item in inputs:
        p = str(Path(item))
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build merged dataset NPZ from rollout NPZ files."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input rollout NPZ (repeat flag for multiple files).",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Optional glob pattern for rollout NPZ files (e.g. runs/*/rollout_ep0.npz).",
    )
    parser.add_argument("--output", required=True, help="Output merged dataset NPZ path.")
    parser.add_argument(
        "--save-report-json",
        default=None,
        help="Optional report JSON output path.",
    )
    parser.add_argument(
        "--dt-tolerance",
        type=float,
        default=1e-9,
        help="Allowed dt mismatch tolerance between input rollouts.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    inputs = _collect_inputs(args.input, args.glob)
    if not inputs:
        raise ValueError("No inputs provided. Use --input and/or --glob.")

    report = merge_rollouts(
        inputs=inputs,
        output=args.output,
        dt_tolerance=args.dt_tolerance,
    )
    print(f"Output dataset: {report['output']}")
    print(f"Rollouts: {report['rollouts']}")
    print(f"Total steps: {report['total_steps']}")
    print(f"Duration (s): {report['duration_sec']:.3f}")
    print(f"dt: {report['dt']:.6f}")

    if args.save_report_json is not None:
        out = Path(args.save_report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report JSON: {out}")


if __name__ == "__main__":
    main()
