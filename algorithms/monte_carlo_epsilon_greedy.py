from monte_carlo_exploring_starts import MonteCarloExploringStarts

import numpy as np
import grid_world_env
from gymnasium.wrappers import TimeLimit


class MonteCarloEpsilonGreedy(MonteCarloExploringStarts):
    def __init__(
        self,
        env,
        gamma,
        max_episode_length=100,
        seed=None,
        first_visit=True,
        epsilon=0.5,
    ) -> None:
        super().__init__(env, gamma, max_episode_length, seed, first_visit)

        self.epsilon = epsilon

    def _improve_policy(self, visited_sa):
        for sa in visited_sa:
            s = sa[0]
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


if __name__ == "__main__":
    import numpy as np
    import time
    import random

    seed = int(10000 * random.random())
    size = 5
    max_episode_steps = 100
    iterate_steps = 10000

    env = TimeLimit(
        grid_world_env.GridWorldEnv(render_mode=None, size=size), max_episode_steps=max_episode_steps
    )

    base_env: grid_world_env.GridWorldEnv = env.unwrapped  # type: ignore

    env.reset(seed=seed)

    it = MonteCarloEpsilonGreedy(env, 0.9, seed=seed, first_visit=True, epsilon=0.2)

    # MC 的样本是随机的，"π 连续若干局不变"并不代表收敛
    # （实测 2000~5000 局时出现过上千局的稳定 streak，但策略还没到最优），
    # 所以直接跑固定局数；size=10 这种状态空间要再放大（5e4 起）
    stable = 0
    for cnt in range(iterate_steps):
        old = [row[:] for row in it.pi]
        it.monte_carlo_update()
        stable = stable + 1 if it.pi == old else 0
        if cnt % 1000 == 0:
            print(f"monte carlo update {cnt}'th")

    print(f"monte carlo update {iterate_steps}'th, stable{stable}'th...")
    pi = it.pi

    env.close()
    # 把策略交给环境：render_mode="human" 时每格画出箭头/圆点，
    # 当前 agent 所在格的动作用红色高亮，跑 episode 时能直接看出它按策略在走

    episode_over = False
    total_reward = 0.0

    env = TimeLimit(
        grid_world_env.GridWorldEnv(render_mode="human", size=size),
        max_episode_steps=max_episode_steps,
    )

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
