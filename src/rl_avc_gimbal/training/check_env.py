from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _version_of_pkg(dist_name: str) -> str:
    try:
        return str(importlib_metadata.version(dist_name))
    except Exception as exc:
        return f"ERROR: {exc}"


def _parse_major(version_str: str) -> int | None:
    if version_str.startswith("ERROR:"):
        return None
    token = version_str.split(".")[0]
    return int(token) if token.isdigit() else None


def _check_import(module_name: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-s", "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return "ok"
    stderr = (proc.stderr or "").strip()
    if stderr:
        first_line = stderr.splitlines()[-1]
        return f"ERROR: {first_line}"
    return "ERROR: import failed"


def diagnose() -> dict[str, Any]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": _version_of_pkg("numpy"),
        "torch": _version_of_pkg("torch"),
        "stable_baselines3": _version_of_pkg("stable-baselines3"),
        "tensorboard": _version_of_pkg("tensorboard"),
        "tensorflow": _version_of_pkg("tensorflow"),
        "matplotlib": _version_of_pkg("matplotlib"),
        "gymnasium": _version_of_pkg("gymnasium"),
    }
    import_checks = {
        "stable_baselines3_import": _check_import("stable_baselines3"),
        "matplotlib_import": _check_import("matplotlib"),
    }

    issues: list[str] = []
    suggestions: list[str] = []

    np_major = _parse_major(versions["numpy"])
    tf_ver = versions["tensorflow"]
    if np_major is not None and np_major >= 2 and not tf_ver.startswith("ERROR:"):
        issues.append(
            "Detected NumPy >= 2 with TensorFlow present. This commonly breaks TensorBoard import paths in SB3 stacks."
        )
        suggestions.append("Pin NumPy to <2 for this environment.")
        suggestions.append("If TensorFlow is not required, uninstall tensorflow/tensorflow-intel.")

    if versions["stable_baselines3"].startswith("ERROR:") or import_checks[
        "stable_baselines3_import"
    ].startswith("ERROR:"):
        issues.append(
            f"stable_baselines3 import failed ({import_checks['stable_baselines3_import']})."
        )
        suggestions.append("Install or repair stable-baselines3 and its dependencies in comp_vision.")

    if versions["matplotlib"].startswith("ERROR:") or import_checks["matplotlib_import"].startswith(
        "ERROR:"
    ):
        issues.append(f"matplotlib import failed ({import_checks['matplotlib_import']}).")
        suggestions.append("Repair matplotlib installation in comp_vision.")

    return {
        "versions": versions,
        "import_checks": import_checks,
        "issues": issues,
        "suggestions": suggestions,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose RL project Python environment.")
    parser.add_argument("--save-json", default=None, help="Optional path to save diagnosis JSON.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = diagnose()
    print("Environment Versions:")
    for key, value in report["versions"].items():
        print(f"  {key}: {value}")
    print("")
    print("Import Checks:")
    for key, value in report["import_checks"].items():
        print(f"  {key}: {value}")
    print("")
    if report["issues"]:
        print("Issues:")
        for item in report["issues"]:
            print(f"  - {item}")
    else:
        print("Issues: none detected")
    if report["suggestions"]:
        print("")
        print("Suggestions:")
        for item in report["suggestions"]:
            print(f"  - {item}")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("")
        print(f"Wrote report: {out}")


if __name__ == "__main__":
    main()
