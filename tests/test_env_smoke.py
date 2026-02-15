import numpy as np

from rl_avc_gimbal.envs import GimbalAVCEnv


def test_env_smoke() -> None:
    env = GimbalAVCEnv()
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape

    for _ in range(20):
        action = np.zeros(3, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(reward)
        assert "accel_term" in info
        if terminated or truncated:
            break
    env.close()
