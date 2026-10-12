import numpy as np
from grid_world_env.envs.grid_world import GridWorldEnv, Actions
from gymnasium.wrappers import TimeLimit


class MonteCarloExploringStarts:
    def __init__(
        self,
        env,
        gamma,
        max_episode_length=100,
        seed=None,
        first_visit=True,
    ) -> None:
        self.env = env
        self.base_env: GridWorldEnv = env.unwrapped  # type: ignore

        self.n_states = self.base_env.size**2
        self.n_actions = self.base_env.action_space.n  # type: ignore
        self.pi = [[1 / self.n_actions] * self.n_actions for _ in range(self.n_states)]

        self.seed = seed
        self.gamma = gamma
        self.max_episode_length = max_episode_length
        self.first_visit = first_visit

        self.qsa = [[0] * self.n_actions for _ in range(self.n_states)]
        self.n_visit_sa = [[0] * self.n_actions for _ in range(self.n_states)]

        self._starts = [
            s
            for s in range(self.n_states)
            if divmod(s, self.base_env.size) not in self.base_env._forbidden_locations
        ]
        self.rng = np.random.default_rng(seed)

    def _agent_state(self, obs):
        return int(obs["agent"][0]) * self.base_env.size + int(obs["agent"][1])

    def _sample_action(self, s):
        return self.rng.choice(self.n_actions, p=self.pi[s])

    def _generate_episode(self):
        cnt = 0
        episode_over = False
        episode = []

        self.env.reset(seed=self.seed)

        s0 = int(self.rng.choice(self._starts))
        a0 = int(self.rng.integers(self.n_actions))

        self.base_env._agent_location = np.array(
            divmod(s0, self.base_env.size), dtype=int
        )

        # 第一步用 (s0, a0)，之后按当前 pi 采样；s / a 必须跟着轨迹走
        s = s0
        a = a0

        while cnt < self.max_episode_length and not episode_over:
            cnt += 1
            obs, r, terminated, truncated, info = self.env.step(a)

            episode.append((s, a, float(r)))

            episode_over = terminated or truncated

            s = self._agent_state(obs)
            a = self._sample_action(s)

        return episode

    def _improve_policy(self, visited_sa):
        for sa in visited_sa:
            s = sa[0]
            max_qsa = max(self.qsa[s])
            n_max_qsa = self.qsa[s].count(max_qsa)
            self.pi[s] = [1 / n_max_qsa if qsa == max_qsa else 0 for qsa in self.qsa[s]]

    def _update(self, episode):
        g = 0.0
        visited_sa = set()
        # 回报要从后往前累加：G_t = r_{t+1} + gamma * G_{t+1}
        for i in reversed(range(len(episode))):
            s, a, r = episode[i]

            g = r + self.gamma * g

            if not self.first_visit or (s, a) not in visited_sa:
                visited_sa.add((s, a))
                self.n_visit_sa[s][a] += 1
                self.qsa[s][a] += (g - self.qsa[s][a]) / self.n_visit_sa[s][a]

            self._improve_policy(visited_sa)

        return self.pi

    def monte_carlo_update(self):
        episode = self._generate_episode()
        self._update(episode)
        return self.pi


if __name__ == "__main__":
    import numpy as np
    import time
    import random

    seed = int(10000 * random.random())
    size = 5
    max_episode_steps = 100
    iterate_steps = 10000

    env = TimeLimit(
        GridWorldEnv(render_mode=None, size=size), max_episode_steps=max_episode_steps
    )

    base_env: GridWorldEnv = env.unwrapped  # type: ignore

    env.reset(seed=seed)

    it = MonteCarloExploringStarts(env, 0.9, seed=seed, first_visit=True)

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
        GridWorldEnv(render_mode="human", size=size),
        max_episode_steps=max_episode_steps,
    )

    base_env: GridWorldEnv = env.unwrapped  # type: ignore

    env.reset(seed=seed)

    _, _, cells, states_idx = base_env.build_mdp()

    base_env.set_policy(pi, show_ties=True)

    while not episode_over:
        s = states_idx[tuple(base_env._agent_location)]
        a = int(np.argmax(pi[s]))  # type: ignore

        obs, r, terminated, truncated, info = env.step(a)

        total_reward += float(r)

        print(f"s={cells[s]} a={Actions(a).name} r={r:+.2f} 累计={total_reward:+.2f}")

        episode_over = terminated or truncated

    time.sleep(5)

    env.close()
