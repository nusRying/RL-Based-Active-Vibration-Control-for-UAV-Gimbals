# Baseline Run Record: `runs/sac_hw_short`

This file records the first completed SAC hardware-profile baseline run.

## Run Context

- Algorithm: `SAC`
- Timesteps: `300000`
- Env config: `configs/env_hw_merged.yaml`
- Eval episodes: `20` (deterministic)
- Artifacts:
  - `runs/sac_hw_short/eval_summary.json`
  - `runs/sac_hw_short/rollout_ep0.npz`
  - `runs/sac_hw_short/frequency_summary.json`

## Key Evaluation Metrics

From `runs/sac_hw_short/eval_summary.json`:
- `return_mean = -2570.839`
- `return_std = 1620.164`
- `accel_rms_mean = 0.76562`
- `accel_rms_std = 0.23747`
- `pos_rms_mean = 0.00610`
- `chatter_rms_mean = 0.07524`
- `peak_abs_angle_mean_rad = 0.04439`
- `peak_abs_torque_mean_nm = 0.44250`
- `termination_rate = 0.0`
- `truncated_episodes = 20`

Interpretation:
- Stability gate passes (`termination_rate == 0`).
- Episodes end by horizon (not failure), which is expected for current setup.
- Torque is close to limit (`0.45 Nm` axis limit), so next tuning should watch chatter/torque tradeoff.

## Frequency Analysis Caveat

From `runs/sac_hw_short/frequency_summary.json`:
- `dt ~ 0.005`, so sample rate is `200 Hz` and Nyquist is `100 Hz`.
- Reported `100-400 Hz` band effectively collapses to the single `100 Hz` edge.

Action:
- For valid `100-400 Hz` analysis, increase sampling rate (reduce `dt`), or
- keep `dt=0.005` and analyze a valid band like `10-90 Hz`.

## Recommended Next Iteration

1. Keep this run as baseline `B0`.
2. Re-run frequency analysis at `10-90 Hz` for current sampling rate:

```powershell
python scripts/analyze_frequency.py `
  --rollout-npz runs/sac_hw_short/rollout_ep0.npz `
  --band-min-hz 10 `
  --band-max-hz 90 `
  --save-json runs/sac_hw_short/frequency_summary_10_90.json `
  --save-csv runs/sac_hw_short/frequency_trace_10_90.csv
```

3. Apply next single tuning change from `docs/tuning_playbook.md` based on those corrected band metrics.
