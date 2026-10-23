from algorithms.SARSA import SARSA
import numpy as np
from grid_world_env.envs.grid_world import GridWorldEnv, Actions
from gymnasium.wrappers import TimeLimit


class ExpectedSarsa(SARSA):
    def _action_value_update(self, s0, a0, r, s1, a1, terminated):
        E = 0
        if not terminated:
            for a, p in enumerate(self.pi[s1]):
                E += p * self.qsa[s1][a]

        self.qsa[s0][a0] = self.qsa[s0][a0] - self.alpha * (
            self.qsa[s0][a0] - r - self.gamma * E
        )

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
            a1 = 0

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

    # 这台机器的 Qt 装得不全：PySide6 只有 xcb 插件、没有 libqwayland，而 matplotlib
    # 默认的 qtagg 后端会崩（启动时报 "Could not find the Qt platform plugin wayland"
    # 段错误，或在 Qt 事件循环 backend_qt.start_main_loop 里段错误）。
    # 改用不依赖 Qt 的 Tk 后端（tkinter 8.6 已装，经 XWayland 显示）；
    # 若连窗口都不想要，换成 matplotlib.use("Agg")，图照样存成 training_curve.png。
    import matplotlib

    matplotlib.use("TkAgg")

    import matplotlib.pyplot as plt

    seed = int(10000 * random.random())
    size = 5
    max_episode_steps = 500
    iterate_steps = 500

    start_location = (0, 0)
    target_location = (2, 1)

    forbidden_locations = [(1, 0), (1, 1), (2, 2), (2, 3), (3, 1)]

    env = TimeLimit(
        GridWorldEnv(
            size,
            start_cell=start_location,
            target_cell=target_location,
            forbidden_cells=forbidden_locations,
            render_mode=None,
        ),
        max_episode_steps=max_episode_steps,
    )

    base_env: GridWorldEnv = env.unwrapped  # type: ignore

    it = ExpectedSarsa(env, alpha=0.1, epsilon=0.1, gamma=0.9, seed=seed)

    rewards = []
    episode_lengths = []

    for cnt in range(iterate_steps):
        episode_length, reward = it.sarsa_iteration()

        episode_lengths.append(episode_length)
        rewards.append(reward)

        if cnt % 100 == 0:
            print(f"expected sarsa iterate {cnt}'th...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---------- 左图：Rewards ----------
    axes[0].plot(rewards, alpha=0.3, color="steelblue", label="raw")
    axes[0].plot(smooth(rewards), color="blue", linewidth=2, label="smoothed")
    axes[0].set_title("Rewards per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_ylim(top=0)  # ★ reward 上限为 0
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # ---------- 右图：Episode Lengths ----------
    axes[1].plot(episode_lengths, alpha=0.3, color="salmon", label="raw")
    axes[1].plot(
        smooth(episode_lengths), color="darkred", linewidth=2, label="smoothed"
    )
    axes[1].set_title("Episode Lengths")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    pi = it.pi
    env.close()

    env = TimeLimit(
        GridWorldEnv(
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

    base_env: GridWorldEnv = env.unwrapped  # type: ignore

    env.reset(seed=seed)

    _, _, _, cells, states_idx = base_env.build_mdp()

    base_env.set_policy(pi)

    while not episode_over:
        s = states_idx[tuple(base_env._agent_location)]
        a = int(np.argmax(pi[s]))  # type: ignore

        obs, r, terminated, truncated, info = env.step(a)

        total_reward += float(r)

        print(f"s={cells[s]} a={Actions(a).name} r={r:+.2f} 累计={total_reward:+.2f}")

        episode_over = terminated or truncated

    time.sleep(5)

    env.close()
