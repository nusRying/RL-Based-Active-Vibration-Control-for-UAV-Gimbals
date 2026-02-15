from gymnasium.envs.registration import register

from .gimbal_avc_env import GimbalAVCEnv

try:
    register(id="GimbalAVC-v0", entry_point="rl_avc_gimbal.envs.gimbal_avc_env:GimbalAVCEnv")
except Exception:
    pass

__all__ = ["GimbalAVCEnv"]
