from gymnasium.envs.registration import register

from continuous_cartpole_env.envs import (
    ContinuousCartPoleEnv,
    ContinuousCartPoleVectorEnv,
)

register(
    id="continuous_cartpole_env/ContinuousCartpole-v0",
    entry_point="continuous_cartpole_env.envs:ContinuousCartPoleEnv",
    vector_entry_point="continuous_cartpole_env.envs:ContinuousCartPoleVectorEnv",
    max_episode_steps=500,
)

__all__ = ["ContinuousCartPoleEnv", "ContinuousCartPoleVectorEnv"]
