import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

import continuous_cartpole_env


class ActorNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # 环境的动作空间是 Box(-1, 1, (1,))，tanh 正好把输出压到这个区间
        return torch.tanh(x)


class CriticNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CriticNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DPG:
    """确定性策略梯度（deterministic policy gradient）。

    β(a|s)  = μ(s) + ε,  ε ~ N(0, σ²I)           行为策略（只负责采样动作）
               σ = noise_std，训练时可以退火
    δ_t     = r_{t+1} + γ q(s_{t+1}, μ(s_{t+1}, θ_t), w_t) − q(s_t, a_t, w_t)
    w_{t+1} = w_t + α_w δ_t ∇_w q(s_t, a_t, w_t)
    θ_{t+1} = θ_t + α_θ ∇_θ μ(s_t, θ_t) · ∇_a q(s_t, a, w_{t+1})|_{a=μ(s_t)}

    和 A2C 的区别在于 actor 输出的是一个**确定的动作**而不是分布：
    - 策略梯度里没有 log π，actor 的梯度要靠 ∇_a q 从 critic 反传回来，
      所以 actor 更新必须留在 critic 的计算图里；
    - 目标里的下一动作取 μ(s_{t+1})（不打噪声），探索噪声只进行为策略 β；
    - critic 更新用更新【前】的 w 算 δ，actor 更新用更新【后】的 w（图中顺序）。
    """

    def __init__(
        self,
        n_envs,
        hidden_dim,
        gamma,
        actor_lr,
        critic_lr,
        noise_std=0.3,
        device=None,
        seed=None,
    ):

        self.env = gym.make_vec(
            "continuous_cartpole_env/ContinuousCartpole-v0",
            num_envs=n_envs,
            vectorization_mode="vector_entry_point",
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
        self.action_dim = int(np.prod(single_act_space.shape))  # type: ignore

        self.actor = ActorNet(self.state_dim, hidden_dim, self.action_dim).to(
            self.device
        )
        self.critic = CriticNet(self.state_dim, self.action_dim, hidden_dim).to(
            self.device
        )

        self.gamma = gamma
        self.noise_std = noise_std  # 探索噪声标准差，训练时可以外部退火

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
        """行为策略 β(a|s) = μ(s) + ε，ε ~ N(0, σ²)，按批采样（向量环境用）。

        ``torch.randn_like`` 抽的就是标准正态，乘 ``noise_std`` 得到 σ 标准差。
        加完噪声后裁到 [-1, 1]：环境本身也会 clip，但训练时 q(s, a) 必须用
        和真正执行的那个 a 一致，所以在这里先裁好（等价于截断正态）。
        """
        mu = self.actor(self._to_tensor(s))
        noise = torch.randn_like(mu) * self.noise_std  # N(0, σ²)
        return (mu + noise).clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _greedy_action(self, s):
        """确定性策略 μ(s)，评估 / 播放用，不加噪声。"""
        return float(self.actor(self._to_tensor(s))[0, 0].item())

    # ------------------------------------------------------------------
    def _update_critic(self, st0, at0, delta):
        # δ 当常数乘子：梯度正好是 δ·∇_w q(s,a,w)，和图中 w 的更新式一致
        loss = -(delta * self.critic(st0, at0)).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def _update_actor(self, st0):
        # 对 q(s, μ(s)) 求 θ 的梯度，链式法则自动带出 ∇_a q · ∇_θ μ；
        # 这里已经是更新后的 critic（w_{t+1}），所以不再乘 δ
        loss = -self.critic(st0, self.actor(st0)).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def train(self):
        """推进一次向量环境，用同一批 transition 更新 critic 和 actor。"""

        obs, actions = self._obs, self._actions

        next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_actions = self._sample_actions(next_obs)

        # 自动 reset 会制造 (终止帧 → 新局初始帧) 的假 transition，丢掉
        valid = ~self._prev_done

        if valid.any():
            st0 = self._to_tensor(obs[valid])
            st1 = self._to_tensor(next_obs[valid])
            at0 = torch.as_tensor(
                actions[valid], dtype=torch.float32, device=self.device
            ).view(-1, self.action_dim)
            rt = torch.as_tensor(
                rewards[valid], dtype=torch.float32, device=self.device
            ).view(-1, 1)
            term = torch.as_tensor(
                terminated[valid], dtype=torch.float32, device=self.device
            ).view(-1, 1)

            # δ_t：下一动作取 μ(s_{t+1})，v 用更新【前】的 w
            with torch.no_grad():
                target = rt + self.gamma * self.critic(st1, self.actor(st1)) * (
                    1.0 - term
                )
                delta = target - self.critic(st0, at0)

            self._update_critic(st0, at0, delta)
            self._update_actor(st0)

        self._obs, self._actions = next_obs, next_actions
        self._prev_done = np.logical_or(terminated, truncated)

        return int(valid.sum()), float(rewards[valid].mean()) if valid.any() else 0.0

    # ------------------------------------------------------------------
    def evaluate(self, n_episodes=20):
        env = gym.make("continuous_cartpole_env/ContinuousCartpole-v0")
        scores = []

        for _ in range(n_episodes):
            s, _ = env.reset(seed=int(self.rng.integers(1 << 31)))
            total_reward = 0.0

            episode_over = False
            while not episode_over:
                a = np.array([self._greedy_action(s)], dtype=np.float32)
                s, r, terminated, truncated, _ = env.step(a)
                total_reward += float(r)

                episode_over = terminated or truncated
            scores.append(total_reward)

        env.close()
        return np.asarray(scores, dtype=np.float64)

    def play(self):
        env = gym.make(
            "continuous_cartpole_env/ContinuousCartpole-v0", render_mode="human"
        )
        s, _ = env.reset(seed=int(self.rng.integers(1 << 31)))
        total_reward = 0.0

        episode_over = False
        while not episode_over:
            a = np.array([self._greedy_action(s)], dtype=np.float32)
            s, r, terminated, truncated, _ = env.step(a)
            total_reward += float(r)

            episode_over = terminated or truncated

        print(f"total_reward = {total_reward}")
        env.close()


if __name__ == "__main__":
    from tqdm import tqdm

    n_envs = 32
    n_steps = 20000  # 向量 step 数；总环境步数 = n_steps * n_envs

    hidden_dim = 128
    gamma = 0.99
    # 两时间尺度：critic 学得快（3e-3），actor 学得慢（1e-5），DPG 对 actor 的
    # 步长很敏感，调大到 1e-4 会「学到 490 然后崩」
    actor_lr = 1e-5
    critic_lr = 3e-3
    # 探索噪声 N(0, σ²) 从 0.3 退火到 0.1。**不能退到 0**：β 一旦失去探索，
    # critic 就只在最优轨迹附近准确，actor 会追着它的误差跑偏（实测 0.02 时
    # 有一半种子在拿到 500 之后掉回两位数，0.1 则 8/8 种子稳定收敛）
    noise_std, noise_min = 0.3, 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dpg = DPG(
        n_envs,
        hidden_dim,
        gamma,
        actor_lr,
        critic_lr,
        noise_std=noise_std,
        device=device,
    )

    pbar = tqdm(range(1, n_steps + 1), desc="训练", unit="步")
    for it in pbar:
        # 探索噪声线性退火：前 60% 从 noise_std 降到 noise_min
        if it < 0.6 * n_steps:
            dpg.noise_std = noise_std + (noise_min - noise_std) * it / (0.6 * n_steps)
        else:
            dpg.noise_std = noise_min

        dpg.train()
        if it % 1000 == 0:
            scores = dpg.evaluate()
            pbar.set_postfix(
                score=f"{np.mean(scores):.1f}",
                std=f"{np.std(scores):.1f}",
                noise=f"{dpg.noise_std:.2f}",
            )

    scores = dpg.evaluate()
    print(f"贪心评估: {scores.mean():.1f} ± {scores.std():.1f}")

    dpg.play()
