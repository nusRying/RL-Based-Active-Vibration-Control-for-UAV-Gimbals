# RLlib Debug: Next Steps

If RLlib logs show:
- `reward_mean=0.000`
- `len_mean=0.00`

then you should first verify episodes complete correctly.

## 1) Probe Env Outside RLlib

```powershell
conda activate comp_vision
python scripts/probe_env.py `
  --env-config configs/env_hw_merged.yaml `
  --episodes 5 `
  --policy random
```

Expected:
- non-zero episode lengths
- each episode ends by `terminated` or `truncated`
- done reason usually `horizon` or `angle_limit`

If this probe fails, fix env termination logic first.

For fast debug loops, use:
- `configs/env_hw_debug_120.yaml`
- this gives exactly 120 steps per episode at `dt=0.005`.

## 2) Print Correct RLlib Metrics

RLlib changed result schema across versions.
Use this extraction:

```python
result = algo.train()
er = result.get("env_runners", {})
reward_mean = er.get("episode_return_mean", result.get("episode_reward_mean", 0.0))
len_mean = er.get("episode_len_mean", result.get("episode_len_mean", 0.0))
print(f"reward_mean={reward_mean:.3f} len_mean={len_mean:.2f}")
print("episodes_this_iter:", result.get("episodes_this_iter"))
```

## 3) Force Complete Episodes in PPO Sampling

```python
config = (
    PPOConfig()
    .rollouts(
        num_rollout_workers=0,
        rollout_fragment_length=120,
        batch_mode="complete_episodes",
    )
    .training(train_batch_size=1200)
)
```

## 4) Metrics Exporter Agent Error

`Failed to establish connection to the metrics exporter agent` is usually non-fatal.
You can continue training while focusing on episode metrics and returns.
