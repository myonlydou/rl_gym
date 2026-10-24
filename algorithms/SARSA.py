import grid_world_env
from gymnasium.wrappers import TimeLimit

import numpy as np
import pandas as pd


class SARSA:
    def __init__(self, env, alpha, gamma, epsilon, seed) -> None:
        self.env = env
        self.base_env: grid_world_env.GridWorldEnv = env.unwrapped

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.seed = seed

        self.n_states = self.base_env.size**2
        self.n_actions = self.base_env.action_space.n  # type: ignore

        self.pi = [[1 / self.n_actions] * self.n_actions for _ in range(self.n_states)]
        self.qsa = [[0] * self.n_actions for _ in range(self.n_states)]

        self.rng = np.random.default_rng(seed)

    def _action_value_update(self, s0, a0, r, s1, a1, terminated):
        next_qsa = 0
        if not terminated:
            next_qsa = self.qsa[s1][a1]
        self.qsa[s0][a0] = self.qsa[s0][a0] - self.alpha * (
            self.qsa[s0][a0] - r - self.gamma * next_qsa
        )

    def _policy_update(self, s):
        max_qsa = max(self.qsa[s])

        pi_a = 1 - (self.n_actions - 1) / self.n_actions * self.epsilon
        pi_other = 1 / self.n_actions * self.epsilon

        found = False
        for i, qsa in enumerate(self.qsa[s]):

            if qsa == max_qsa and not found:
                found = True
                self.pi[s][i] = pi_a
            else:
                self.pi[s][i] = pi_other

    def _sample_action(self, s):
        return self.rng.choice(self.n_actions, p=self.pi[s])

    def _agent_state(self, obs):
        return obs["agent"][0] * self.base_env.size + obs["agent"][1]

    def sarsa_iteration(self):
        obs, info = self.env.reset(seed=self.seed)
        episode_over = False

        cnt = 0
        total_reward = 0

        while not episode_over:
            s0 = self._agent_state(obs)
            a0 = self._sample_action(s0)

            obs, r, terminated, truncated, info = self.env.step(a0)

            s1 = self._agent_state(obs)
            a1 = self._sample_action(s1)

            self._action_value_update(s0, a0, r, s1, a1, terminated)
            self._policy_update(s0)

            cnt += 1
            total_reward += r

            episode_over = terminated or truncated

        return cnt, total_reward


def smooth(data, window=50):
    """滑动平均平滑曲线"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


if __name__ == "__main__":
    import random
    import time

    seed = int(10000 * random.random())
    size = 5
    max_episode_steps = 500
    iterate_steps = 500

    start_location = (0, 0)
    target_location = (2, 1)

    forbidden_locations = [(1, 0), (1, 1), (2, 2), (2, 3), (3, 1)]

    env = TimeLimit(
        grid_world_env.GridWorldEnv(
            size,
            start_cell=start_location,
            target_cell=target_location,
            forbidden_cells=forbidden_locations,
            render_mode=None,
        ),
        max_episode_steps=max_episode_steps,
    )

    base_env: grid_world_env.GridWorldEnv = env.unwrapped  # type: ignore

    it = SARSA(env, alpha=0.1, epsilon=0.1, gamma=0.9, seed=seed)

    rewards = []
    episode_lengths = []

    for cnt in range(iterate_steps):
        episode_length, reward = it.sarsa_iteration()

        episode_lengths.append(episode_length)
        rewards.append(reward)

        if cnt % 100 == 0:
            print(f"sarsa iterate {cnt}'th...")

    pi = it.pi
    env.close()

    env = TimeLimit(
        grid_world_env.GridWorldEnv(
            size,
            start_cell=start_location,
            target_cell=target_location,
            forbidden_cells=forbidden_locations,
            render_mode="human",
        ),
        max_episode_steps=max_episode_steps,
    )

    episode_over = False
    total_reward = 0.0

    base_env: grid_world_env.GridWorldEnv = env.unwrapped  # type: ignore

    env.reset(seed=seed)

    _, _, _, cells, states_idx = base_env.build_mdp()

    base_env.set_policy(pi)

    while not episode_over:
        s = states_idx[tuple(base_env._agent_location)]
        a = int(np.argmax(pi[s]))  # type: ignore

        obs, r, terminated, truncated, info = env.step(a)

        total_reward += float(r)

        print(f"s={cells[s]} a={grid_world_env.Actions(a).name} r={r:+.2f} 累计={total_reward:+.2f}")

        episode_over = terminated or truncated

    time.sleep(5)

    env.close()
