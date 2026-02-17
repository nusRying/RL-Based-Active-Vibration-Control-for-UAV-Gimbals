# Hardware Kickoff Checklist

This checklist is the starting point to move from default simulation numbers to
your real gimbal setup.

Before running commands in this checklist:

```powershell
conda activate comp_vision
```

## 1) Create Your Hardware Profile

1. Copy `configs/env_hw_profile.example.yaml` to `configs/env_hw_profile.yaml`.
2. Fill measured/estimated values from bench data.
3. Keep units consistent:
- torque in Nm
- inertia in kg*m^2
- angle in rad
- rate in rad/s

## 2) Measure Core Parameters

Minimum measurements before training:
- Max continuous torque per axis (`torque_limit_nm`).
- Payload mass and center-of-mass offset.
- Closed-loop control rate (set `dt = 1/rate_hz`).
- IMU noise floor at idle for `imu_noise_std`.

Sampling rule for spectral targets:
- If you want analysis/control coverage up to `f_max`, ensure sample rate
  `f_s >= 2*f_max` (Nyquist minimum), and preferably `f_s >= 2.5*f_max`.
- For `100-400 Hz` vibration studies, use at least `f_s=800 Hz` (`dt<=0.00125`).
- Recommended practical setting: `f_s=1000 Hz` (`dt=0.001`) when feasible.

Recommended identification:
- Inertia per axis using step response or CAD + mass properties.
- Damping/stiffness via ring-down fit.

Ring-down model per axis:
\[
I\ddot{\theta} + c\dot{\theta} + k\theta = 0
\]
Estimate damping ratio \(\zeta\) and natural frequency \(\omega_n\), then:
\[
k = I\omega_n^2,\quad c = 2\zeta\sqrt{kI}
\]

## 3) Compose Final Training Config

Merge base config with hardware overrides:

```powershell
python scripts/compose_env.py `
  --base configs/env.yaml `
  --profile configs/env_hw_profile.yaml `
  --output configs/env_hw_merged.yaml
```

If this command fails, fix profile shape/type errors first.

## 4) Safety-First Initial Run

Before long training:
1. Keep `hybrid_control.enabled: true`.
2. Keep `rl_delta_scale` conservative (0.15-0.25).
3. Run short training budget first.

Example short run:

```powershell
python scripts/train.py `
  --algo sac `
  --env-config configs/env_hw_merged.yaml `
  --algo-config configs/sac.yaml `
  --run-dir runs/sac_hw_short
```

## 5) Immediate Validation Outputs

After short training, generate:

```powershell
python scripts/eval.py `
  --algo sac `
  --model-path runs/sac_hw_short/final_model `
  --env-config configs/env_hw_merged.yaml `
  --episodes 20 `
  --save-json runs/sac_hw_short/eval_summary.json `
  --record-first-episode runs/sac_hw_short/rollout_ep0.npz

python scripts/analyze_frequency.py `
  --rollout-npz runs/sac_hw_short/rollout_ep0.npz `
  --band-min-hz 100 `
  --band-max-hz 400 `
  --save-json runs/sac_hw_short/frequency_summary.json `
  --save-csv runs/sac_hw_short/frequency_trace.csv
```

## 6) Acceptance Gates

Do not proceed to aggressive tuning until:
- `termination_rate == 0` over 20 episodes.
- Peak torque remains inside safe actuator envelope.
- Frequency-band attenuation improves vs baseline.
- Pointing error does not regress beyond mission threshold.

## 7) Next Step After Kickoff

Use `docs/tuning_playbook.md` for controlled one-change-per-run tuning.
