import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import random
import collections

import continuous_cartpole_env


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity, seed=None):
        self.buffer = collections.deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, float(reward), next_obs, bool(done)))

    def add_batch(self, obs, actions, rewards, next_obs, dones, mask=None):
        obs = np.asarray(obs, dtype=np.float32)

        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)

        if mask is None:
            mask = np.ones(obs.shape[0], dtype=bool)

        for i in np.flatnonzero(mask):
            self.add(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

    def sample(self, size):
        transitions = self.rng.sample(self.buffer, size)
        obs, action, reward, next_obs, done = zip(*transitions)

        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
        )

    def size(self):
        return len(self.buffer)


class ActorNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorNet, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

        torch.nn.init.uniform_(self.fc2.weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.fc2.bias, -3e-3, 3e-3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x)
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
        x = F.elu(x)
        x = self.fc2(x)
        return x


class DDPG:
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
        tau,
        actor_lr,
        critic_lr,
        noise_std,
        batch_size,
        buffer_size,
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

        self.main_actor = ActorNet(self.state_dim, hidden_dim, self.action_dim).to(
            self.device
        )
        self.target_actor = ActorNet(self.state_dim, hidden_dim, self.action_dim).to(
            self.device
        )
        self.target_actor.load_state_dict(self.main_actor.state_dict())

        self.main_critic = CriticNet(self.state_dim, self.action_dim, hidden_dim).to(
            self.device
        )
        self.target_critic = CriticNet(self.state_dim, self.action_dim, hidden_dim).to(
            self.device
        )
        self.target_critic.load_state_dict(self.main_critic.state_dict())

        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, seed=seed)

        self.tau = tau
        self.gamma = gamma
        self.noise_std = noise_std  # 探索噪声标准差，训练时可以外部退火

        # 两时间尺度：critic 学得快，actor 学得慢
        self.actor_optimizer = torch.optim.Adam(
            self.main_actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.main_critic.parameters(), lr=critic_lr
        )

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
        mu = self.main_actor(self._to_tensor(s))
        noise = torch.randn_like(mu) * self.noise_std  # N(0, σ²)
        return (mu + noise).clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _greedy_action(self, s):
        """确定性策略 μ(s)，评估 / 播放用，不加噪声。"""
        return float(self.main_actor(self._to_tensor(s))[0, 0].item())

    # ------------------------------------------------------------------
    def _update_critic(self, st0, at0, target):
        loss = F.mse_loss(self.main_critic(st0, at0), target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def _update_actor(self, st0):
        # 对 q(s, μ(s)) 求 θ 的梯度，链式法则自动带出 ∇_a q · ∇_θ μ；
        # 这里已经是更新后的 critic（w_{t+1}），所以不再乘 δ
        loss = -self.main_critic(st0, self.main_actor(st0)).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    @torch.no_grad()
    def _soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    def train(self):

        obs, actions = self._obs, self._actions

        next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_actions = self._sample_actions(next_obs)

        valid = ~self._prev_done

        self.buffer.add_batch(obs, actions, rewards, next_obs, terminated, valid)

        if valid.any() and self.buffer.size() >= 10 * self.batch_size:

            (
                sample_obs,
                sample_actions,
                sample_rewards,
                sample_next_obs,
                sample_dones,
            ) = self.buffer.sample(self.batch_size)

            st0 = self._to_tensor(sample_obs)
            st1 = self._to_tensor(sample_next_obs)
            at0 = torch.as_tensor(
                sample_actions, dtype=torch.float32, device=self.device
            ).view(-1, self.action_dim)
            rt = torch.as_tensor(
                sample_rewards, dtype=torch.float32, device=self.device
            ).view(-1, 1)
            term = torch.as_tensor(
                sample_dones, dtype=torch.float32, device=self.device
            ).view(-1, 1)

            # δ_t：下一动作取 μ(s_{t+1})，v 用更新【前】的 w
            with torch.no_grad():
                target = rt + self.gamma * self.target_critic(
                    st1, self.target_actor(st1)
                ) * (1.0 - term)

            self._update_critic(st0, at0, target)
            self._update_actor(st0)
            self._soft_update(self.main_critic, self.target_critic)
            self._soft_update(self.main_actor, self.target_actor)

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
    tau = 0.005
    gamma = 0.99
    actor_lr = 1e-5
    critic_lr = 3e-3
    noise_std, noise_min = 0.3, 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2048
    buffer_size = 200000

    ddpg = DDPG(
        n_envs,
        hidden_dim,
        gamma,
        tau,
        actor_lr,
        critic_lr,
        noise_std,
        batch_size,
        buffer_size,
        device=device,
    )

    pbar = tqdm(range(1, n_steps + 1), desc="训练", unit="步")
    for it in pbar:
        # 探索噪声线性退火：前 60% 从 noise_std 降到 noise_min
        if it < 0.6 * n_steps:
            ddpg.noise_std = noise_std + (noise_min - noise_std) * it / (0.6 * n_steps)
        else:
            ddpg.noise_std = noise_min

        ddpg.train()
        if it % 1000 == 0:
            scores = ddpg.evaluate()
            pbar.set_postfix(
                score=f"{np.mean(scores):.1f}",
                std=f"{np.std(scores):.1f}",
                noise=f"{ddpg.noise_std:.2f}",
            )

    scores = ddpg.evaluate()
    print(f"贪心评估: {scores.mean():.1f} ± {scores.std():.1f}")

    ddpg.play()
