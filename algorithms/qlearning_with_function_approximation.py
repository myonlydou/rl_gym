import numpy as np
import gymnasium as gym

from algorithms.SARSA_with_function_approximation import SARSAWithFunctionApproximation


class QLearningWithFunctionApproximation(SARSAWithFunctionApproximation):
    def _update_omega(self, obs0, a0, r, obs1, a1, terminated=False):
        delta = r - float(self._phi_t(obs0) @ self.omega[:, a0])
        if not terminated:
            delta += self.gamma * float((self._phi_t(obs1) @ self.omega).max())

        self.omega[:, a0] = self.omega[:, a0] + self.alpha * delta * self._phi_t(obs0)


if __name__ == "__main__":

    LOW = np.array([-2.4, -3.0, -0.21, -3.0])
    HIGH = np.array([2.4, 3.0, 0.21, 3.0])

    env = gym.make("CartPole-v1")
    it = QLearningWithFunctionApproximation(env, 0.99, 0.1, 1.0, LOW, HIGH)

    rewards = []
    n_episodes = 6000
    for ep in range(n_episodes):
        it.epsilon = (
            1.0 + (0.05 - 1.0) * ep / (0.6 * n_episodes)
            if ep < 0.6 * n_episodes
            else 0.05
        )
        _, total_reward = it.iteration()
        rewards.append(total_reward)
        if (ep + 1) % 300 == 0:
            print(
                f"第 {ep+1} 局, epsilon={it.epsilon:.3f}, average|rewards|={np.average(rewards):.2f}"
            )
            rewards.clear()

    scores = [it.run_episode(env)[1] for _ in range(20)]
    print(f"贪心评估: {np.mean(scores):.1f} ± {np.std(scores):.1f}")

    env.close()
    env = gym.make("CartPole-v1", render_mode="human")

    _, total_reward = it.run_episode(env)

    print(f"total_reward:{total_reward}...")

    env.close()
