import numpy as np
import gymnasium as gym


class SarsaWithFunctionApproximation:

    def __init__(self, env, gamma, alpha, epsilon, LOW, HIGH, seed=None) -> None:
        self.env = env

        self.LOW, self.HIGH = np.asarray(LOW), np.asarray(HIGH)

        self.n_actions = self.env.action_space.n  # type: ignore

        self.gamma, self.alpha, self.epsilon = gamma, alpha, epsilon

        self.omega = np.zeros((self._phi_t(LOW).shape[0], self.n_actions))

        self.seed = seed

        self.rng = np.random.default_rng(seed=seed)

    def _phi_t(self, obs):
        x, x_dot, theta, theta_dot = np.clip(obs, self.LOW, self.HIGH) / self.HIGH

        return np.asarray(
            [
                1.0,
                x,
                x_dot,
                theta,
                theta_dot,
                theta * theta_dot,  # 杆角×角速度：平衡的主要判据
                x * theta,
                x_dot * theta_dot,
                theta**2,
                x_dot**2,
            ]
        )

    def _action_values(self, obs):
        return self._phi_t(obs) @ self.omega

    def _sample_action(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self._action_values(obs)))

    def _update_omega(self, obs0, a0, r, obs1, a1, terminated=False):

        delta = r - float(self._phi_t(obs0) @ self.omega[:, a0])
        if not terminated:
            delta += self.gamma * float(self._phi_t(obs1) @ self.omega[:, a1])

        self.omega[:, a0] = self.omega[:, a0] + self.alpha * delta * self._phi_t(obs0)

    def iteration(self):
        obs, _ = self.env.reset(seed=self.seed)
        a = self._sample_action(obs)

        episode_over = False
        steps, total_reward = 0, 0.0

        while not episode_over:
            next_obs, r, terminated, truncated, _ = self.env.step(a)

            next_a = self._sample_action(next_obs)

            self._update_omega(obs, a, r, next_obs, next_a, terminated)

            obs = next_obs
            a = next_a

            steps += 1
            total_reward += r
            episode_over = terminated or truncated

        return steps, total_reward

    def run_episode(self, env, seed=None):

        obs, _ = env.reset(seed=seed)
        a = self._sample_action(obs)

        episode_over = False
        steps, total_reward = 0, 0.0

        while not episode_over:
            next_obs, r, terminated, truncated, _ = env.step(a)

            next_a = self._sample_action(next_obs, True)

            a = next_a

            steps += 1
            total_reward += r

            episode_over = terminated or truncated

        return steps, total_reward


if __name__ == "__main__":

    LOW = np.array([-2.4, -3.0, -0.21, -3.0])
    HIGH = np.array([2.4, 3.0, 0.21, 3.0])

    env = gym.make("CartPole-v1")
    it = SarsaWithFunctionApproximation(env, 0.99, 0.1, 1.0, LOW, HIGH)

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
