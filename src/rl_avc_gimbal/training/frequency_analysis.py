from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

AXES = ("roll", "pitch", "yaw")
EPS = 1e-12


def _transfer_function(
    disturbance: np.ndarray,
    response: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = disturbance.shape[0]
    u = disturbance.astype(np.float64) - float(np.mean(disturbance))
    y = response.astype(np.float64) - float(np.mean(response))

    u_fft = np.fft.rfft(u)
    y_fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=dt)

    suu = u_fft * np.conjugate(u_fft)
    syu = y_fft * np.conjugate(u_fft)
    h = syu / (suu + EPS)

    mag_db = 20.0 * np.log10(np.abs(h) + EPS)
    phase_deg = np.unwrap(np.angle(h)) * 180.0 / np.pi
    return freqs.astype(np.float64), mag_db.astype(np.float64), phase_deg.astype(np.float64)


def _band_summary(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    band_min_hz: float,
    band_max_hz: float,
) -> dict[str, float]:
    mask = (freqs >= band_min_hz) & (freqs <= band_max_hz)
    if not np.any(mask):
        return {
            "band_mean_mag_db": float("nan"),
            "band_peak_mag_db": float("nan"),
            "band_peak_freq_hz": float("nan"),
        }

    f_band = freqs[mask]
    m_band = mag_db[mask]
    peak_idx = int(np.argmax(m_band))
    return {
        "band_mean_mag_db": float(np.mean(m_band)),
        "band_peak_mag_db": float(m_band[peak_idx]),
        "band_peak_freq_hz": float(f_band[peak_idx]),
    }


def analyze_rollout(
    rollout_npz: str | Path,
    band_min_hz: float = 100.0,
    band_max_hz: float = 400.0,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    path = Path(rollout_npz)
    data = np.load(path)

    required = {"dt", "disturbance_torque", "imu_accel", "torque_cmd"}
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing keys in rollout NPZ: {missing}")

    dt = float(np.asarray(data["dt"]).squeeze())
    disturbance = np.asarray(data["disturbance_torque"], dtype=np.float64)
    imu_accel = np.asarray(data["imu_accel"], dtype=np.float64)
    torque_cmd = np.asarray(data["torque_cmd"], dtype=np.float64)

    if disturbance.ndim != 2 or disturbance.shape[1] != 3:
        raise ValueError("Expected disturbance_torque shape [T, 3].")
    if imu_accel.shape != disturbance.shape:
        raise ValueError("Expected imu_accel shape to match disturbance_torque.")
    if torque_cmd.shape != disturbance.shape:
        raise ValueError("Expected torque_cmd shape to match disturbance_torque.")

    traces: dict[str, dict[str, np.ndarray]] = {}
    axis_summaries: dict[str, dict[str, float]] = {}

    for axis_idx, axis_name in enumerate(AXES):
        freqs, mag_db, phase_deg = _transfer_function(
            disturbance=disturbance[:, axis_idx],
            response=imu_accel[:, axis_idx],
            dt=dt,
        )
        summary = _band_summary(
            freqs=freqs,
            mag_db=mag_db,
            band_min_hz=band_min_hz,
            band_max_hz=band_max_hz,
        )
        summary["torque_rms_nm"] = float(np.sqrt(np.mean(torque_cmd[:, axis_idx] ** 2)))

        traces[axis_name] = {
            "freq_hz": freqs,
            "mag_db": mag_db,
            "phase_deg": phase_deg,
        }
        axis_summaries[axis_name] = summary

    output = {
        "rollout": str(path),
        "dt": dt,
        "sample_count": int(disturbance.shape[0]),
        "band_min_hz": float(band_min_hz),
        "band_max_hz": float(band_max_hz),
        "axes": axis_summaries,
    }
    return output, traces


def _write_csv(path: str | Path, traces: dict[str, dict[str, np.ndarray]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["axis", "freq_hz", "mag_db", "phase_deg"])
        for axis_name in AXES:
            axis_trace = traces[axis_name]
            for f, m, p in zip(
                axis_trace["freq_hz"],
                axis_trace["mag_db"],
                axis_trace["phase_deg"],
            ):
                writer.writerow([axis_name, f"{f:.6f}", f"{m:.6f}", f"{p:.6f}"])
    return out_path


def _write_json(path: str | Path, summary: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze rollout frequency response from saved evaluation NPZ."
    )
    parser.add_argument("--rollout-npz", required=True)
    parser.add_argument("--band-min-hz", type=float, default=100.0)
    parser.add_argument("--band-max-hz", type=float, default=400.0)
    parser.add_argument("--save-csv", default=None)
    parser.add_argument("--save-json", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary, traces = analyze_rollout(
        rollout_npz=args.rollout_npz,
        band_min_hz=args.band_min_hz,
        band_max_hz=args.band_max_hz,
    )

    print(f"Rollout: {summary['rollout']}")
    print(f"Samples: {summary['sample_count']} | dt: {summary['dt']:.6f} s")
    print(
        "Band: "
        f"{summary['band_min_hz']:.1f}-{summary['band_max_hz']:.1f} Hz"
    )

    for axis_name in AXES:
        axis = summary["axes"][axis_name]
        print(
            f"{axis_name}: mean={axis['band_mean_mag_db']:.3f} dB, "
            f"peak={axis['band_peak_mag_db']:.3f} dB @ {axis['band_peak_freq_hz']:.2f} Hz, "
            f"torque_rms={axis['torque_rms_nm']:.4f} Nm"
        )

    if args.save_csv is not None:
        csv_path = _write_csv(args.save_csv, traces)
        print(f"Wrote frequency CSV: {csv_path}")

    if args.save_json is not None:
        json_path = _write_json(args.save_json, summary)
        print(f"Wrote frequency JSON: {json_path}")


if __name__ == "__main__":
    main()
