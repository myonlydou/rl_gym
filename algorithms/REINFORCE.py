import numpy as np
import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


class REINFORCE:
    def __init__(
        self,
        env,
        hidden_dim,
        gamma,
        learning_rate,
        device,
    ):

        self.env = env
        s, _ = env.reset()
        self.n_states = s.shape[0]
        self.n_actions = env.action_space.n

        self.policy_net = PolicyNet(self.n_states, hidden_dim, self.n_actions).to(
            device
        )

        self.gamma = gamma

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.device = device

    def _sample_action(self, s, greedy=False):
        s = torch.as_tensor(s, dtype=torch.float, device=self.device).unsqueeze(0)
        a_probs = self.policy_net(s)

        if greedy:
            return a_probs.argmax(dim=-1).item()

        action_dist = torch.distributions.Categorical(a_probs)
        return action_dist.sample().item()

    def train(self):
        s, _ = self.env.reset()

        episode_over = False

        total_reward = 0.0
        cnt = 0

        s_list = []
        a_list = []
        r_list = []

        while not episode_over:
            s_list.append(s)

            a = self._sample_action(s)
            a_list.append(a)

            s, r, terminated, truncated, _ = self.env.step(a)
            r_list.append(r)

            episode_over = terminated or truncated

            total_reward += r
            cnt += 1

        qsa = 0
        self.optimizer.zero_grad()
        for i in reversed(range(cnt)):
            qsa = self.gamma * qsa + r_list[i]

            s = torch.as_tensor(
                s_list[i], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            a = torch.tensor([a_list[i]], dtype=torch.long).view(-1, 1).to(self.device)

            lnpi = torch.log(self.policy_net(s).gather(1, a))

            loss = -lnpi * qsa

            loss.backward()
        self.optimizer.step()

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
    gamma = 0.98
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reinforce = REINFORCE(
        env,
        hidden_dim,
        gamma,
        learning_rate,
        device,
    )

    pbar = tqdm(range(1, n_episodes + 1), desc="训练", unit="局")
    for ep in pbar:
        reinforce.train()
        if ep % 100 == 0:
            scorces = reinforce.evaluate()
            pbar.set_postfix(
                score=f"{np.mean(scorces):.1f}",
                std=f"{np.std(scorces):.1f}",
            )

    scores = reinforce.evaluate()
    print(f"贪心评估: {scores.mean():.1f} ± {scores.std():.1f}")

    play_env = gym.make("CartPole-v1", render_mode="human")
    reinforce.play(play_env)

    env.close()
    play_env.close()
