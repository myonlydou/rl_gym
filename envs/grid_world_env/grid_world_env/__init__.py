from gymnasium.envs.registration import register

from grid_world_env.envs import Actions, GridWorldEnv

register(
    id="grid_world_env/GridWorld-v0",
    entry_point="grid_world_env.envs:GridWorldEnv",
    max_episode_steps=100,
)

__all__ = ["Actions", "GridWorldEnv"]
