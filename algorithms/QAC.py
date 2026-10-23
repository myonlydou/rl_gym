import numpy as np
import torch
import torch.nn.functional as F


class ActorNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


class CriticNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(CriticNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class QAC:

    def __init__(
        self,
        env,
        hidden_dim,
        gamma,
        actor_lr,
        critic_lr,
        device=None,
    ):

        self.env = env
        s, _ = env.reset()
        self.state_dim = s.shape[0]
        self.n_actions = env.action_space.n

        self.device = device

        self.actor = ActorNet(self.state_dim, hidden_dim, self.n_actions).to(
            self.device
        )
        self.critic = CriticNet(self.state_dim, hidden_dim, self.n_actions).to(
            self.device
        )

        self.gamma = gamma

        # 两时间尺度：critic 学得快，actor 学得慢
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def _to_tensor(self, s):
        return torch.as_tensor(
            np.asarray(s), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

    def _sample_action(self, s, greedy=False):
        s = torch.as_tensor(s, dtype=torch.float, device=self.device).unsqueeze(0)
        a_probs = self.actor(s)

        if greedy:
            return int(a_probs.argmax(dim=-1).item())

        return int(torch.distributions.Categorical(a_probs).sample().item())

    def _update_critic(self, st0, at0, rt, st1, at1, terminated):
        with torch.no_grad():
            target = rt
            if not terminated:  # truncated 不是真终止，仍然要 bootstrap
                target = target + self.gamma * self.critic(st1).gather(1, at1)

        # (a - x) ^ 2 = -2x'(a-x)
        loss = F.mse_loss(self.critic(st0).gather(1, at0), target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def _update_actor(self, st0, at0):
        with torch.no_grad():
            qsa = self.critic(st0).gather(1, at0)

        lnpi = torch.log(self.actor(st0)).gather(1, at0)
        loss = -lnpi * qsa
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def train(self):
        s0, _ = self.env.reset()
        a0 = self._sample_action(s0)

        episode_over = False

        total_reward = 0.0
        cnt = 0

        while not episode_over:
            s1, r, terminated, truncated, _ = self.env.step(a0)
            a1 = self._sample_action(s1)  # 从 s1 采，只用于 bootstrap，本步不执行

            st0 = self._to_tensor(s0)
            at0 = torch.tensor([a0], dtype=torch.long, device=self.device).view(-1, 1)
            st1 = self._to_tensor(s1)
            at1 = torch.tensor([a1], dtype=torch.long, device=self.device).view(-1, 1)
            rt = torch.tensor([r], dtype=torch.float32, device=self.device).view(-1, 1)

            self._update_critic(st0, at0, rt, st1, at1, terminated)
            self._update_actor(st0, at0)

            episode_over = terminated or truncated

            s0 = s1
            a0 = self._sample_action(s1)
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
    gamma = 0.99
    actor_lr = 3e-4
    critic_lr = 1e-2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qac = QAC(env, hidden_dim, gamma, actor_lr, critic_lr, device=device)

    pbar = tqdm(range(1, n_episodes + 1), desc="训练", unit="局")
    for ep in pbar:
        qac.train()
        if ep % 100 == 0:
            scorces = qac.evaluate()
            pbar.set_postfix(
                score=f"{np.mean(scorces):.1f}",
                std=f"{np.std(scorces):.1f}",
            )

    scores = qac.evaluate()
    print(f"贪心评估: {scores.mean():.1f} ± {scores.std():.1f}")

    play_env = gym.make("CartPole-v1", render_mode="human")
    qac.play(play_env)

    env.close()
    play_env.close()
