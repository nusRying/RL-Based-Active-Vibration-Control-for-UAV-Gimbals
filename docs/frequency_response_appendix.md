# Frequency-Response Appendix for RL AVC

This appendix explains how to interpret the controller in the frequency domain,
how each reward term shapes that response, and how to extract practical Bode-like
insights from this repository's rollout logs.

## 1) Why Frequency Domain for AVC

Active vibration control is fundamentally a spectral problem. Camera jitter is often
concentrated in narrow harmonic bands (prop/motor related) plus broadband residuals.
A time-domain reward can train a useful policy, but frequency-domain diagnostics tell
you *why* the policy works or fails.

Useful outcomes from frequency analysis:
- Identify resonance peaks that remain uncontrolled.
- Verify attenuation in the target disturbance band (default 100-400 Hz).
- Quantify the control-effort tradeoff for attenuation improvements.
- Compare SAC vs DDPG policies beyond aggregate episode return.

## 2) Signals in This Project

From `src/rl_avc_gimbal/training/evaluate.py` with `--record-first-episode`,
you get rollout signals in `.npz`:
- `disturbance_torque[t, axis]`: injected disturbance input `u(t)`.
- `imu_accel[t, axis]`: measured payload acceleration response `y(t)`.
- `torque_cmd[t, axis]`: control effort signal.
- `theta[t, axis]`, `theta_dot[t, axis]`: state traces.
- `dt`: sample interval.

Interpretation:
- Input for transfer estimate: disturbance torque.
- Output for transfer estimate: IMU acceleration proxy.

## 3) Transfer Function Estimate Used Here

For each axis, with sampled input `u[n]` and output `y[n]`:

1. Remove mean:
\[
\tilde{u}[n] = u[n] - \bar{u},\quad \tilde{y}[n] = y[n] - \bar{y}
\]

2. FFT:
\[
U[k] = \mathcal{F}\{\tilde{u}[n]\},\quad Y[k] = \mathcal{F}\{\tilde{y}[n]\}
\]

3. Spectra:
\[
S_{uu}[k] = U[k]U^*[k],\quad S_{yu}[k] = Y[k]U^*[k]
\]

4. FRF estimate:
\[
\hat{H}[k] = \frac{S_{yu}[k]}{S_{uu}[k] + \epsilon}
\]

5. Bode-like quantities:
\[
|\hat{H}[k]|_{dB} = 20\log_{10}(|\hat{H}[k]| + \epsilon)
\]
\[
\angle\hat{H}[k]_{deg} = \text{unwrap}(\arg(\hat{H}[k]))\cdot\frac{180}{\pi}
\]

`src/rl_avc_gimbal/training/frequency_analysis.py` implements this exactly.

## 3.1 Sampling Constraints (Critical)

Frequency analysis is limited by sampling rate:
\[
f_s=\frac{1}{dt},\quad f_N=\frac{f_s}{2}
\]
where \(f_N\) is Nyquist frequency.

For current default runtime:
- `dt = 0.005` s
- `f_s = 200` Hz
- `f_N = 100` Hz

Implication:
- A target band of `100-400 Hz` is **not** fully observable at 200 Hz.
- Only the 100 Hz edge can appear, which collapses band statistics.

To analyze up to 400 Hz, use at least:
- `f_s >= 800` Hz (`dt <= 0.00125` s)
- practical recommendation: `f_s = 1000` Hz (`dt = 0.001` s) for margin.

If staying at 200 Hz, analyze a valid band such as `10-90 Hz`.

## 4) How Reward Terms Shape Frequency Response

Reward in environment:
\[
r_t = -(w_a\,a_t^2 + w_p\,e_t^2 + w_c\,\Delta\tau_t^2)
\]
where:
- `a_t`: acceleration magnitude term
- `e_t`: position error term
- `\Delta\tau_t = \tau_t-\tau_{t-1}`

### 4.1 `w_a` (acceleration penalty)

Primary spectral effect:
- Pushes policy to reduce output magnitude `|H(j\omega)|` where disturbance energy exists.

Expected observations when `w_a` increases:
- Lower mean/peak FRF magnitude in 100-400 Hz.
- Potentially higher control effort if not balanced by `w_c`.

### 4.2 `w_p` (position error penalty)

Primary effect:
- Improves low-frequency setpoint holding and drift suppression.

Expected observations when `w_p` increases:
- Better low-frequency behavior and angle drift.
- Can reduce high-frequency attenuation if policy reallocates authority to pointing.

### 4.3 `w_c` (chatter penalty)

Primary spectral effect:
- Penalizes rapid control changes, acting like an implicit high-frequency control regularizer.

Expected observations when `w_c` increases:
- Smoother torque command spectrum.
- Possible reduction in very-high-frequency attenuation if over-weighted.

## 5) Bode-Style Interpretation Rules

Use these rules axis-by-axis:

1. Peak magnitude in 100-400 Hz is high and narrow.
- Interpretation: residual resonance/harmonic not suppressed.
- Typical fix: raise `w_a`; optionally increase `w_c` only if torque is too aggressive.

2. Broadly elevated magnitude across band with low torque RMS.
- Interpretation: policy is conservative.
- Typical fix: reduce `w_c` slightly or increase policy capacity/training duration.

3. Low band magnitude but large position drift.
- Interpretation: high-frequency suppression dominates, low-frequency pointing sacrificed.
- Typical fix: increase `w_p` moderately.

4. Strong attenuation but torque RMS near limits.
- Interpretation: policy is effective but expensive.
- Typical fix: raise `w_c`, then retrain and recheck attenuation loss.

5. SAC outperforms DDPG in attenuation stability.
- Interpretation: stochastic entropy-regularized learning handles disturbance diversity better.
- Typical action: keep SAC as primary deployment candidate.

## 6) Practical Workflow with Commands

Step 1: Run evaluation and record rollout.

```powershell
python scripts/eval.py `
  --algo sac `
  --model-path runs/sac_baseline/final_model `
  --env-config configs/env.yaml `
  --episodes 20 `
  --save-json runs/sac_baseline/eval_summary.json `
  --record-first-episode runs/sac_baseline/rollout_ep0.npz
```

Step 2: Run frequency analysis in target band.

```powershell
python scripts/analyze_frequency.py `
  --rollout-npz runs/sac_baseline/rollout_ep0.npz `
  --band-min-hz 10 `
  --band-max-hz 90 `
  --save-json runs/sac_baseline/frequency_summary.json `
  --save-csv runs/sac_baseline/frequency_trace.csv
```

Outputs:
- `frequency_summary.json`: per-axis band mean/peak FRF magnitude and peak frequency.
- `frequency_trace.csv`: frequency/magnitude/phase samples for plotting externally.

## 7) Suggested Acceptance Criteria

These are starting criteria; tune for your payload quality requirements:
- 0 terminated episodes in 20 deterministic eval episodes.
- At least 3 dB reduction in mean 100-400 Hz FRF magnitude vs baseline controller.
- No >10% increase in RMS position error.
- Torque RMS remains within thermal/current limits for your actuator profile.

## 8) Pitfalls and Corrections

Common pitfall: using one short rollout only.
- Correction: compare multiple seeds and disturbance draws before final decisions.

Common pitfall: interpreting phase where input energy is near zero.
- Correction: ignore bins where `Suu` is very small; prioritize magnitude + band peaks.

Common pitfall: tuning weights with only return value.
- Correction: include FRF metrics and torque RMS every iteration.

Common pitfall: over-smoothing with high `w_c`.
- Correction: lower `w_c` incrementally and watch band peak recovery.

## 9) How This Maps to Real Hardware

On real flight data, replace `disturbance_torque` proxy with a measured disturbance-related
reference channel (e.g., motor RPM harmonics projected to gimbal frame) and keep the
same FRF workflow.

If direct disturbance input is unavailable, compare closed-loop output PSD before/after
controller changes as a practical alternative.
