from grid_world_env.envs.grid_world import GridWorldEnv, Actions
from gymnasium.wrappers import TimeLimit


class ValueIteration:
    def __init__(self, P, R, theta, gamma) -> None:
        self.n_states = P.shape[0]
        self.n_actions = P.shape[1]
        self.v = [0] * self.n_states
        self.pi = [[1 / self.n_actions] * self.n_actions for _ in range(self.n_states)]

        self.P = P
        self.R = R
        self.theta = theta
        self.gamma = gamma

    def value_iteration(self):
        cnt = 0
        while True:
            delta = 0.0
            next_v = [0] * self.n_states
            for s in range(self.n_states):
                qsa_list = []
                for a in range(self.n_actions):
                    qsa = 0
                    for i, p in enumerate(self.P[s][a]):
                        qsa += p * self.R[s][a][i] + self.gamma * p * self.v[i]
                    qsa_list.append(qsa)

                max_qsa = max(qsa_list)
                n_max_qsa = qsa_list.count(max_qsa)
                self.pi[s] = [
                    1 / n_max_qsa if qsa == max_qsa else 0 for qsa in qsa_list
                ]

                next_v[s] = max(qsa_list)
                delta = max(delta, abs(next_v[s] - self.v[s]))

            cnt += 1
            print(f"value {cnt}'th iteration...")
            self.v = next_v

            if delta < self.theta:
                print(f"value iteration converged in {cnt} iterations...")
                break
        return self.pi, self.v


if __name__ == "__main__":
    import numpy as np
    import time

    env = TimeLimit(GridWorldEnv(render_mode="human", size=10), max_episode_steps=40)

    base_env: GridWorldEnv = env.unwrapped  # type: ignore

    env.reset()

    P, R, _, cells, states_idx = base_env.build_mdp()

    it = ValueIteration(P, R, 0.001, 0.9)

    pi, v = it.value_iteration()

    # 把策略交给环境：render_mode="human" 时每格画出箭头/圆点，
    # 当前 agent 所在格的动作用红色高亮，跑 episode 时能直接看出它按策略在走
    base_env.set_policy(pi)
    base_env.set_v(v)

    episode_over = False
    total_reward = 0.0

    while not episode_over:
        s = states_idx[tuple(base_env._agent_location)]
        a = int(np.argmax(pi[s]))

        obs, r, terminated, truncated, info = env.step(a)

        total_reward += float(r)

        print(f"s={cells[s]} a={Actions(a).name} r={r:+.2f} 累计={total_reward:+.2f}")

        episode_over = terminated or truncated

    time.sleep(5)

    env.close()
