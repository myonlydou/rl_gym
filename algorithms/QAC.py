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
    """并行版 QAC：n_envs 个环境同时跑，每个向量 step 产出 n_envs 条 transition，
    合成一个 batch 做一次更新（batch 大小 = n_envs），不用等一整局结束。"""

    def __init__(
        self, n_envs, hidden_dim, gamma, actor_lr, critic_lr, device=None, seed=None
    ):

        self.env = gym.make_vec(
            "CartPole-v1", num_envs=n_envs, vectorization_mode="vector_entry_point"
        )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 向量环境：action_space 是 MultiDiscrete，没有 .n，要用 single_*_space
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
        self.critic = CriticNet(self.state_dim, hidden_dim, self.n_actions).to(
            self.device
        )

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
        return x.unsqueeze(0) if x.dim() == 1 else x  # 兼容单环境 (4,) 和并行 (N,4)

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
    def _update_critic(self, st0, at0, rt, st1, at1, terminated):
        with torch.no_grad():
            # terminated 是 (B,1) 的 0/1 张量，批量版不能再写 if not terminated
            target = rt + self.gamma * self.critic(st1).gather(1, at1) * (
                1.0 - terminated
            )

        # (a - x) ^ 2 = -2x'(a-x)
        loss = F.mse_loss(self.critic(st0).gather(1, at0), target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def _update_actor(self, st0, at0):
        with torch.no_grad():
            q_all = self.critic(st0)
            qsa = q_all.gather(1, at0)

        lnpi = torch.log(self.actor(st0)).gather(1, at0)
        loss = -(lnpi * qsa).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    # ------------------------------------------------------------------
    def train(self):
        """推进一次向量环境（= n_envs 个环境各走一步），并做一次批量更新。"""

        obs, actions = self._obs, self._actions

        next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_actions = self._sample_actions(next_obs)

        # 向量环境是自动 reset 的：上一帧刚结束的 env，这一帧拿到的 obs 已经是新一局
        # 的初始状态，于是 (s, a, r=0, s') 是一条假 transition，必须丢掉；否则 critic
        # 会被拉向"终止态 → 新局初始态"这个根本不存在的转移。
        valid = ~self._prev_done

        if valid.any():
            st0 = self._to_tensor(obs[valid])
            st1 = self._to_tensor(next_obs[valid])
            at0 = torch.as_tensor(
                actions[valid], dtype=torch.long, device=self.device
            ).view(-1, 1)
            at1 = torch.as_tensor(
                next_actions[valid], dtype=torch.long, device=self.device
            ).view(-1, 1)
            rt = torch.as_tensor(
                rewards[valid], dtype=torch.float32, device=self.device
            ).view(-1, 1)
            term = torch.as_tensor(
                terminated[valid], dtype=torch.float32, device=self.device
            ).view(-1, 1)

            self._update_critic(st0, at0, rt, st1, at1, term)
            self._update_actor(st0, at0)

        self._obs, self._actions = next_obs, next_actions
        self._prev_done = np.logical_or(terminated, truncated)

        return int(valid.sum()), float(rewards[valid].mean()) if valid.any() else 0.0

    # ------------------------------------------------------------------
    def evaluate(self, n_episodes=20, seed=None):
        """并行训练、单环境贪心评估（评估用的环境单独开一个）。"""
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

    qac = QAC(n_envs, hidden_dim, gamma, actor_lr, critic_lr, device=device)

    pbar = tqdm(range(1, n_steps + 1), desc="训练", unit="步")
    for it in pbar:
        qac.train()
        if it % 500 == 0:
            scorces = qac.evaluate()
            pbar.set_postfix(
                score=f"{np.mean(scorces):.1f}",
                std=f"{np.std(scorces):.1f}",
            )

    scores = qac.evaluate()
    print(f"贪心评估: {scores.mean():.1f} ± {scores.std():.1f}")

    qac.play()
