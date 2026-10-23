import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym


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
    def __init__(self, state_dim, hidden_dim):
        super(CriticNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class A2C:
    """
    Advantage actor-critic / TD actor-critic（并行版）。

    δ_t     = r_{t+1} + γ v(s_{t+1}, w_t) − v(s_t, w_t)
    w_{t+1} = w_t + α_w δ_t ∇_w v(s_t, w_t)
    θ_{t+1} = θ_t + α_θ δ_t ∇_θ ln π(a_t | s_t, θ_t)
    """

    def __init__(
        self, n_envs, hidden_dim, gamma, actor_lr, critic_lr, device=None, seed=None
    ):

        self.env = gym.make_vec(
            "CartPole-v1", num_envs=n_envs, vectorization_mode="vector_entry_point"
        )
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        single_obs_space = getattr(
            self.env, "single_observation_space", self.env.observation_space
        )
        single_act_space = getattr(
            self.env, "single_action_space", self.env.action_space
        )

        self.n_envs = int(getattr(self.env, "num_envs", 1))
        self.state_dim = int(np.prod(single_obs_space.shape))  # type: ignore
        self.n_actions = int(single_act_space.n)  # type: ignore

        self.actor = ActorNet(self.state_dim, hidden_dim, self.n_actions).to(
            self.device
        )
        self.critic = CriticNet(self.state_dim, hidden_dim).to(self.device)

        self.gamma = gamma

        # 两时间尺度：critic 学得快，actor 学得慢
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        obs, _ = self.env.reset(seed=seed)
        self._obs = np.asarray(obs, dtype=np.float32)
        self._actions = self._sample_actions(self._obs)
        self._prev_done = np.zeros(self.n_envs, dtype=bool)

        self.rng = np.random.default_rng(seed=seed)

    # ------------------------------------------------------------------
    def _to_tensor(self, s):
        x = torch.as_tensor(np.asarray(s), dtype=torch.float32, device=self.device)
        return x.unsqueeze(0) if x.dim() == 1 else x

    @torch.no_grad()
    def _sample_actions(self, s):
        a_probs = self.actor(self._to_tensor(s))
        return (
            torch.distributions.Categorical(a_probs)
            .sample()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

    @torch.no_grad()
    def _greedy_action(self, s):
        return int(self.actor(self._to_tensor(s)).argmax(dim=-1).item())

    # ------------------------------------------------------------------
    def _update_critic(self, st0, delta):

        loss = -(delta * self.critic(st0)).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def _update_actor(self, st0, at0, delta):

        lnpi = torch.log(self.actor(st0)).gather(1, at0)
        loss = -(delta * lnpi).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def train(self):
        """推进一次向量环境，用同一份 δ 更新 critic 和 actor。"""

        obs, actions = self._obs, self._actions

        next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_actions = self._sample_actions(next_obs)

        # 自动 reset 会制造 (终止态 → 新局初始态) 的假 transition，丢掉
        valid = ~self._prev_done

        if valid.any():
            st0 = self._to_tensor(obs[valid])
            st1 = self._to_tensor(next_obs[valid])
            at0 = torch.as_tensor(
                actions[valid], dtype=torch.long, device=self.device
            ).view(-1, 1)
            rt = torch.as_tensor(
                rewards[valid], dtype=torch.float32, device=self.device
            ).view(-1, 1)
            term = torch.as_tensor(
                terminated[valid], dtype=torch.float32, device=self.device
            ).view(-1, 1)

            # δ_t：用更新【前】的 v（w_t）算一次，critic 和 actor 共用
            with torch.no_grad():
                target = rt + self.gamma * self.critic(st1) * (1.0 - term)
                delta = target - self.critic(st0)

            self._update_critic(st0, delta)
            self._update_actor(st0, at0, delta)

        self._obs, self._actions = next_obs, next_actions
        self._prev_done = np.logical_or(terminated, truncated)

        return int(valid.sum()), float(rewards[valid].mean()) if valid.any() else 0.0

    # ------------------------------------------------------------------
    def evaluate(self, n_episodes=20, seed=None):
        env = gym.make("CartPole-v1")
        scorces = []

        for _ in range(n_episodes):
            s, _ = env.reset(seed=int(self.rng.integers(1 << 31)))
            total_reward = 0.0

            episode_over = False
            while not episode_over:
                s, r, terminated, truncated, _ = env.step(self._greedy_action(s))
                total_reward += float(r)

                episode_over = terminated or truncated
            scorces.append(total_reward)

        env.close()
        return np.asarray(scorces, dtype=np.float64)

    def play(self, seed=None):
        env = gym.make("CartPole-v1", render_mode="human")
        s, _ = env.reset(seed=int(self.rng.integers(1 << 31)))
        total_reward = 0.0

        episode_over = False
        while not episode_over:
            s, r, terminated, truncated, _ = env.step(self._greedy_action(s))
            total_reward += float(r)

            episode_over = terminated or truncated

        print(f"total_reward = {total_reward}")


if __name__ == "__main__":
    from tqdm import tqdm

    n_envs = 4096
    n_steps = 10000  # 向量 step 数；总环境步数 = n_steps * n_envs

    hidden_dim = 128
    gamma = 0.99
    actor_lr = 3e-4
    critic_lr = 1e-2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a2c = A2C(n_envs, hidden_dim, gamma, actor_lr, critic_lr, device=device)

    pbar = tqdm(range(1, n_steps + 1), desc="训练", unit="步")
    for it in pbar:
        a2c.train()
        if it % 500 == 0:
            scorces = a2c.evaluate()
            pbar.set_postfix(
                score=f"{np.mean(scorces):.1f}",
                std=f"{np.std(scorces):.1f}",
            )

    scores = a2c.evaluate()
    print(f"贪心评估: {scores.mean():.1f} ± {scores.std():.1f}")

    a2c.play()
