# Optional Dataset Workflow (Not Required for Baseline RL)

This project trains with online RL, so no external dataset is required for baseline
SAC/DDPG training.

This document defines the optional dataset workflow for:
- offline analysis,
- controller comparisons,
- future offline RL experiments.

## 1) What Data to Collect

Use rollout files produced by evaluation:
- `time_sec`
- `reward`
- `theta`
- `theta_dot`
- `imu_accel`
- `torque_cmd`
- `disturbance_torque`
- `dt`

These keys are already emitted when running:

```powershell
python scripts/eval.py `
  --algo sac `
  --model-path runs/sac_baseline/final_model `
  --env-config configs/env.yaml `
  --episodes 20 `
  --record-first-episode runs/sac_baseline/rollout_ep0.npz
```

## 2) Merge Multiple Rollouts into One Dataset

Use the dataset builder:

```powershell
python scripts/build_dataset.py `
  --glob "runs/*/rollout_ep0.npz" `
  --output datasets/avc_merged.npz `
  --save-report-json datasets/avc_merged_report.json
```

You can also pass explicit files:

```powershell
python scripts/build_dataset.py `
  --input runs/sac_a/rollout_ep0.npz `
  --input runs/sac_b/rollout_ep0.npz `
  --output datasets/avc_two_runs.npz
```

## 3) Output Schema

Merged dataset keys:
- `dt` scalar
- `time_sec` shape `[N]`
- `reward` shape `[N]`
- `theta` shape `[N, 3]`
- `theta_dot` shape `[N, 3]`
- `imu_accel` shape `[N, 3]`
- `torque_cmd` shape `[N, 3]`
- `disturbance_torque` shape `[N, 3]`
- `episode_id` shape `[N]`
- `step_in_episode` shape `[N]`

`N` is the total number of concatenated steps across all input rollouts.

## 4) Validation Rules Enforced

`build_dataset` rejects inputs when:
- required keys are missing,
- shape is inconsistent (`[T,3]` expected for vector signals),
- `dt` mismatches exceed tolerance (`--dt-tolerance`, default `1e-9`).

## 5) Recommended Logging Plan for Real Hardware

When extending from simulation to real bench/flight:
- Keep timestamped channels aligned at fixed control rate (target 200 Hz).
- Preserve axis ordering (`roll`, `pitch`, `yaw`) in every stream.
- Record actuator saturation flags and supply voltage/current if available.
- Record motor RPM or ESC telemetry if available for better disturbance references.

Minimum practical channels:
- gimbal angles/rates,
- payload IMU acceleration,
- control command (torque/current),
- synchronized timestamp.

## 6) Why This Dataset is Useful

- Repeatable A/B policy comparisons on identical logged trajectories.
- Frequency analysis across many conditions instead of one rollout.
- Foundation for future offline RL or imitation-style warm-start experiments.
