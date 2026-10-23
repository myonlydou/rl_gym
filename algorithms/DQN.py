import collections
import random

import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity, seed=None):
        self.buffer = collections.deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, s0, a, r, s1, done):
        self.buffer.append((s0, a, r, s1, done))

    def sample(self, size):
        transitions = self.rng.sample(self.buffer, size)
        s0, a, r, s1, done = zip(*transitions)

        return (
            np.asarray(s0, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(s1, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
        )

    def size(self):
        return len(self.buffer)


class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DQN:
    def __init__(
        self,
        env,
        hidden_dim,
        epsilon,
        gamma,
        learning_rate,
        device,
        target_update,
        batch_size,
        buffer_size,
        seed=None,
    ):
        self.env = env
        s, _ = env.reset()
        self.n_states = s.shape[0]
        self.n_actions = env.action_space.n

        self.main_qnet = Qnet(self.n_states, hidden_dim, self.n_actions).to(device)
        self.target_qnet = Qnet(self.n_states, hidden_dim, self.n_actions).to(device)

        self.target_qnet.load_state_dict(self.main_qnet.state_dict())

        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)

        self.epsilon = epsilon
        self.gamma = gamma

        self.optimizer = torch.optim.Adam(self.main_qnet.parameters(), lr=learning_rate)
        self.device = device

        self.target_update = target_update
        self.update_cnt = 0

        self.rng = np.random.default_rng(seed=seed)

    def _sample_action(self, s, greedy=False):

        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        # unsqueeze(0) (4, ) --> (1, 4)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            a = int(self.main_qnet(s).argmax(dim=1).item())

        return a

    def _update_qnet(self, s0, a, r, s1, done):
        s0 = torch.as_tensor(s0, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s1 = torch.as_tensor(s1, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            target_qsa = r.unsqueeze(dim=1) + self.target_qnet(s1).max(
                dim=1, keepdim=True
            ).values * (1.0 - done.unsqueeze(dim=1))

        qsa = self.main_qnet(s0).gather(1, a.unsqueeze(1))

        loss = F.mse_loss(qsa, target_qsa)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        self.update_cnt += 1
        if self.update_cnt % self.target_update == 0:
            self.target_qnet.load_state_dict(self.main_qnet.state_dict())

        return float(loss.item())

    def train(self):
        s0, _ = self.env.reset()

        episode_over = False

        total_reward = 0.0
        cnt = 0
        s1 = s0
        while not episode_over:
            a = self._sample_action(s1)
            s1, r, terminated, truncated, _ = self.env.step(a)

            episode_over = terminated or truncated
            self.buffer.add(s0, a, r, s1, episode_over)

            s0 = s1

            if self.buffer.size() >= self.batch_size * 10:
                self._update_qnet(*self.buffer.sample(self.batch_size))

            total_reward += r
            cnt += 1

        return cnt, total_reward

    def evaluate(self, n_episodes=20, seed=None):
        scorces = []

        for _ in range(n_episodes):
            s, _ = self.env.reset(seed=seed)
            total_reward = 0.0

            episode_over = False
            while not episode_over:
                a = self._sample_action(s, greedy=True)
                s, r, terminated, truncated, _ = self.env.step(a)
                total_reward += r

                episode_over = terminated or truncated
            scorces.append(total_reward)
        return np.asarray(scorces, dtype=np.float64)

    def play(self, env, seed=None):
        s, _ = env.reset(seed=seed)
        total_reward = 0.0

        episode_over = False
        while not episode_over:
            a = self._sample_action(s, greedy=True)
            s, r, terminated, truncated, _ = env.step(a)
            total_reward += r

            episode_over = terminated or truncated

        print(f"total_reward = {total_reward}")


if __name__ == "__main__":
    from tqdm import tqdm
    import gymnasium as gym

    n_episodes = 500

    env = gym.make("CartPole-v1")
    hidden_dim = 128
    learning_rate = 1e-3
    target_update = 100
    batch_size = 64
    buffer_size = 10240
    gamma = 0.98
    epsilon = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dqn = DQN(
        env,
        hidden_dim,
        epsilon,
        gamma,
        learning_rate,
        device,
        target_update,
        batch_size,
        buffer_size,
    )

    pbar = tqdm(range(1, n_episodes + 1), desc="训练", unit="局")
    for ep in pbar:
        dqn.train()
        if ep % 100 == 0:
            scorces = dqn.evaluate()
            pbar.set_postfix(
                score=f"{np.mean(scorces):.1f}",
                std=f"{np.std(scorces):.1f}",
            )

    scores = dqn.evaluate()
    print(f"贪心评估: {scores.mean():.1f} ± {scores.std():.1f}")

    play_env = gym.make("CartPole-v1", render_mode="human")
    dqn.play(play_env)

    env.close()
    play_env.close()
