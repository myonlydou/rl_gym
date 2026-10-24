"""
CartPole 物理，连续动作版。

原始实现来自 Rich Sutton 等人（http://incompleteideas.net/sutton/book/code/pole.c，
镜像 https://perma.cc/C9ZM-652R）。代码结构、动力学、观测空间、奖励、终止条件和
渲染都沿用 gymnasium 的 CartPole，只把动作空间从"两档固定力"换成"连续可调的力"：

    CartPole-v1         Discrete(2)         action 0/1 → 固定 -10 / +10 N
    ContinuousCartPole  Box(-1, 1, (1,))    action a   → force = a · force_mag

a = ±1 正好落在原来那两档上，轨迹和 CartPole-v1 逐位相同；a 取中间值则是离散
动作空间无法表达的控制。
"""

import math

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space


def action_to_force(action, force_mag: float = 10.0):
    """把 [-1, 1] 的连续动作换算成牛顿力。

    ``a = +1`` -> ``+force_mag``，等同于 ``CartPole-v1`` 的 ``action = 1``
    ``a = -1`` -> ``-force_mag``，等同于 ``CartPole-v1`` 的 ``action = 0``
    ``a =  0`` -> ``0 N``，``Discrete(2)`` 表达不了这一档

    越界值直接 clip 而不是报错：策略网络（比如 DPG 的 tanh 输出乘缩放）经常会
    轻微越界，训练循环里抛异常代价太大。标量进标量出，数组进数组出，形状不变。
    """
    a = np.asarray(action, dtype=np.float64)
    return np.clip(a, -1.0, 1.0) * float(force_mag)


def _as_single_action(action) -> float:
    """单环境一次只收一个动作：标量 / ``(1,)`` 数组 / 只有一个元素的 list 都行。"""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size != 1:
        raise ValueError(f"单环境一次只能收一个动作，收到 shape={np.shape(action)}")
    return float(a[0])


class ContinuousCartPoleEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    ## 描述

    倒立摆：小车在无摩擦轨道上左右移动，杆子通过一个无驱动的关节连在小车上，
    目标是通过施加力让杆保持直立。

    本类和 gymnasium 的 `CartPole-v1` 是同一套物理，**只有动作空间不同**：
    推力连续可调，而不是固定 ±10 N。

    ## 动作空间

    动作是 shape `(1,)`、dtype `float32` 的连续值 `a ∈ [-1, 1]`，按
    `force = a * force_mag`（默认 `force_mag = 10`）换算成牛顿力：

    | `a`  | 作用力  | 等价的 `CartPole-v1` 动作 |
    |------|---------|---------------------------|
    | `+1` | `+10 N` | `1`（右推）               |
    | `0`  | `0 N`   | 无（离散版做不到）        |
    | `-1` | `-10 N` | `0`（左推）               |

    中间值给出成比例的推力；越界的值会被 clip 回 `[-1, 1]`。

    **注意**：推力对速度的影响不是固定的，还取决于杆当前的角度——杆的重心位置
    会改变让小车移动到其下方所需的能量。

    ## 观测空间

    观测是 shape `(4,)` 的 `float32` 向量：

    | 序号 | 含义     | 最小值              | 最大值            |
    |------|----------|---------------------|-------------------|
    | 0    | 小车位置 | -4.8                | 4.8               |
    | 1    | 小车速度 | -Inf                | Inf               |
    | 2    | 杆角度   | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3    | 杆角速度 | -Inf                | Inf               |

    上面是**观测空间**的边界，不是未终止回合里实际能取到的范围：小车位置一越过
    ±2.4 就终止，杆角一越过 ±0.2095 rad（±12°）就终止。

    ## 奖励

    默认每走一步（终止那一步也算）得 `+1`。`sutton_barto_reward=True` 时改成
    非终止步 `0`、终止步 `-1`。

    ## 初始状态

    `reset()` 时四个分量各自在 `(-0.05, 0.05)` 内均匀采样；可以用
    `options={"low": ..., "high": ...}` 覆盖这个范围。

    ## 回合结束

    满足任一条即结束：

    1. 终止：杆角超过 ±12°
    2. 终止：小车位置超过 ±2.4（小车中心到达画面边缘）
    3. 截断：回合长度超过 `max_episode_steps`（注册值 500）

    和官方实现一样，**本类的 `step()` 永远返回 `truncated=False`**：截断由
    `gym.make` 依据注册的 `max_episode_steps` 添加的 `TimeLimit` wrapper 负责。
    直接实例化类时要自己包一层::

        env = TimeLimit(ContinuousCartPoleEnv(), max_episode_steps=500)

    ## 参数

    | 参数                  | 类型      | 默认值  | 说明                                                     |
    |-----------------------|-----------|---------|----------------------------------------------------------|
    | `sutton_barto_reward` | **bool**  | `False` | `True` 时使用 Sutton & Barto 的原始奖励（0 / -1）        |
    | `render_mode`         | **str**   | `None`  | `None` / `"human"` / `"rgb_array"`                       |
    | `force_mag`           | **float** | `10.0`  | `a = ±1` 时施加的力；保持 `10.0` 才和 `CartPole-v1` 一致 |

    ## 向量环境

    `ContinuousCartPoleVectorEnv` 是原生批量实现，已注册为 `vector_entry_point`：

    ```python
    >>> import gymnasium as gym
    >>> import continuous_cartpole_env
    >>> envs = gym.make_vec("continuous_cartpole_env/ContinuousCartpole-v0",
    ...                     num_envs=3, vectorization_mode="vector_entry_point")
    >>> envs.single_action_space
    Box(-1.0, 1.0, (1,), float32)
    ```

    ## 版本

    * 本 fork：动作空间换成 `Box(-1, 1, (1,))`，`force = action * force_mag`；
      `a = ±1` 与 `CartPole-v1` 的 `action = 1 / 0` 逐位一致。
    * 上游：基于 gymnasium `CartPole-v1`（`max_episode_steps=500`）。
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        sutton_barto_reward: bool = False,
        render_mode: str | None = None,
        force_mag: float = 10.0,
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # 半杆长
        self.polemass_length = self.masspole * self.length
        self.force_mag = float(force_mag)
        self.tau = 0.02  # 每步 0.02 s（50 Hz）
        self.kinematics_integrator = "euler"

        # 杆角超过这个值就判定失败
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # 观测上界取 2 倍阈值，保证"终止那一刻"的观测仍然落在观测空间内
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state: np.ndarray | None = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.state is not None, "调用 step 之前先 reset。"
        # 整个环境里"连续"只体现在这一行：a ∈ [-1, 1] → 力 ∈ [-10, 10] N
        # a = +1 / -1 分别对应原来的离散动作 1 / 0
        force = float(action_to_force(_as_single_action(action), self.force_mag))
        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # 推导见：https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # 半隐式欧拉：先更新速度再更新位置
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            # 杆刚倒下，这一帧给终止奖励
            self.steps_beyond_terminated = 0

            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "环境已经返回过 terminated = True，却还在调用 step()。"
                    "收到 terminated = True 后应当立刻 reset()，之后再 step() 的行为未定义。"
                )
            self.steps_beyond_terminated += 1

            reward = -1.0 if self._sutton_barto_reward else 0.0

        if self.render_mode == "human":
            self.render()

        # 截断由 gym.make 加上的 TimeLimit wrapper 负责，所以这里恒为 False
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # 注意：自定义 reset 范围过大时，初始观测可能越界；默认范围 ±0.05
        low, high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "没有指定 render_mode 却调用了 render()。"
                "请在初始化时指定，例如： "
                f'gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "需要 pygame 才能渲染，先装一下： pip install pygame"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # render_mode == "rgb_array"，先画到离屏 Surface 上
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # 小车中心
        carty = 100  # 小车顶部
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class ContinuousCartPoleVectorEnv(VectorEnv):
    """原生批量版的 :class:`ContinuousCartPoleEnv`。

    所有 ``num_envs`` 个环境的物理用一份 ``(4, num_envs)`` 的 numpy 数组一起算，
    所以吞吐不会像 ``SyncVectorEnv`` 那样随环境数线性下降。

    动作是批量后的 ``Box(-1, 1, (num_envs, 1))``：每个环境一个连续力。
    autoreset 沿用 gymnasium 的 ``AutoresetMode.NEXT_STEP``：某个环境在第 t 步
    结束时 terminated/truncated，第 t+1 步会先照常算完物理、再把它重置成新一局，
    并丢弃这一步的动作——返回的 reward 为 0、terminated/truncated 为 False、
    观测是新一局的初始状态。所以 (终止帧 → 新局首帧) 那条 transition 不能用于训练。
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 500,
        render_mode: str | None = None,
        sutton_barto_reward: bool = False,
        force_mag: float = 10.0,
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # 半杆长
        self.polemass_length = self.masspole * self.length
        self.force_mag = float(force_mag)
        self.tau = 0.02  # 每步 0.02 s（50 Hz）
        self.kinematics_integrator = "euler"

        self.state = None

        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.prev_done = np.zeros(num_envs, dtype=np.bool_)

        # 杆角超过这个值就判定失败
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # 观测上界取 2 倍阈值，保证"终止那一刻"的观测仍然落在观测空间内
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.low = -0.05
        self.high = 0.05

        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.screen_width = 600
        self.screen_height = 400
        self.screens = None
        self.surf = None

        self.steps_beyond_terminated = None

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert self.state is not None, "调用 step 之前先 reset。"

        # (num_envs, 1) 和 (num_envs,) 两种形状都收，统一压成 (num_envs,) 的力
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_envs, -1)[:, 0]
        force = action_to_force(actions, self.force_mag)
        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # 推导见：https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # 半隐式欧拉：先更新速度再更新位置
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.stack((x, x_dot, theta, theta_dot))

        terminated: np.ndarray = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        if self._sutton_barto_reward:
            reward = -np.array(terminated, dtype=np.float32)
        else:
            reward = np.ones_like(terminated, dtype=np.float32)

        # 先把这一步的物理算完，再重置上一步结束的环境：重置会覆盖掉刚才的物理
        # 结果，等价于"这些环境这一步的动作被丢弃"，返回的观测因此是新一局的初始
        # 状态（与 gymnasium 的向量环境保持一致）
        self.state[:, self.prev_done] = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.prev_done.sum())
        )
        self.steps[self.prev_done] = 0
        reward[self.prev_done] = 0.0
        terminated[self.prev_done] = False
        truncated[self.prev_done] = False

        self.prev_done = np.logical_or(terminated, truncated)

        return self.state.T.astype(np.float32), reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # 注意：自定义 reset 范围过大时，初始观测可能越界
        # 默认范围是 ±0.05
        self.low, self.high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        self.state = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.num_envs)
        )
        self.steps_beyond_terminated = None
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_done = np.zeros(self.num_envs, dtype=np.bool_)

        return self.state.T.astype(np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "没有指定 render_mode 却调用了 render()。"
                "请在初始化时指定，例如： "
                f'gym.make_vec("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "需要 pygame 才能渲染，先装一下： pip install pygame"
            ) from e

        if self.screens is None:
            pygame.init()

            self.screens = [
                pygame.Surface((self.screen_width, self.screen_height))
                for _ in range(self.num_envs)
            ]

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            raise ValueError("环境还没 reset，state 是 None")

        for x, screen in zip(self.state.T, self.screens, strict=True):
            assert isinstance(x, np.ndarray) and x.shape == (4,)

            self.surf = pygame.Surface((self.screen_width, self.screen_height))
            self.surf.fill((255, 255, 255))

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cartx = x[0] * scale + self.screen_width / 2.0  # 小车中心
            carty = 100  # 小车顶部
            cart_coords = [(l, b), (l, t), (r, t), (r, b)]
            cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
            gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
            gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )

            pole_coords = []
            for coord in [(l, b), (l, t), (r, t), (r, b)]:
                coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
                coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
                pole_coords.append(coord)
            gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
            gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

            gfxdraw.aacircle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )

            gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

            self.surf = pygame.transform.flip(self.surf, False, True)
            screen.blit(self.surf, (0, 0))

        return [
            np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
            for screen in self.screens
        ]

    def close(self):
        if self.screens is not None:
            import pygame

            pygame.quit()


# ----------------------------------------------------------------------
# 自检 / 演示。跑法：python -m continuous_cartpole_env.envs.continuous_cartpole
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    import gymnasium as gym
    from gymnasium.wrappers import TimeLimit

    print("=" * 64)
    print("1) 动作 → 力")
    print("=" * 64)
    for a in (-1.0, -0.5, 0.0, 0.5, 1.0):
        print(f"    a = {a:+.2f}  →  force = {float(action_to_force(a)):+6.1f} N")
    print(f"    a = +3.0  →  force = {float(action_to_force(3.0)):+6.1f} N（被 clip）")

    print()
    print("=" * 64)
    print("2) a = ±1 必须和 CartPole-v1 逐位一致")
    print("=" * 64)
    ref = gym.make("CartPole-v1").unwrapped
    env = ContinuousCartPoleEnv()
    rng = np.random.default_rng(0)
    n_steps, max_diff = 0, 0.0
    for _ in range(20):
        init = rng.uniform(-0.05, 0.05, size=4)
        acts = rng.integers(0, 2, size=200)  # 随机离散动作，跑够一局
        # 先 reset 再覆盖 state：CartPole-v1 的 steps_beyond_terminated 只在
        # reset() 里归零，光改 state 会让第二局起的终止奖励变成 0
        ref.reset()
        env.reset()
        ref.state = init.copy()
        env.state = init.copy()
        for a in acts:
            o_ref, r_ref, term_ref, _, _ = ref.step(int(a))
            o_our, r_our, term_our, _, _ = env.step([2.0 * a - 1.0])  # 0→-1, 1→+1
            max_diff = max(max_diff, float(np.max(np.abs(o_ref - o_our))))
            assert r_ref == r_our and term_ref == term_our, "奖励 / 终止标志不一致"
            n_steps += 1
            if term_ref:
                break
    assert max_diff == 0.0, "a = ±1 时轨迹必须和 CartPole-v1 完全相同"
    print(f"    比对 {n_steps} 步，观测最大偏差 = {max_diff:.1e}  OK")

    print()
    print("=" * 64)
    print("3) 连续动作能不能控住（PD 控制器）")
    print("=" * 64)

    def _pd_controller(obs, kp=30.0, kd=4.0, kx=3.0, kv=4.0):
        x, x_dot, theta, theta_dot = obs
        u = kp * theta + kd * theta_dot + kx * x + kv * x_dot
        return np.array([np.clip(u, -1.0, 1.0)], dtype=np.float32)

    def rollout(policy, episodes=20):
        # 直接实例化不会自己截断 → 套一层 TimeLimit
        rollout_env = TimeLimit(ContinuousCartPoleEnv(), max_episode_steps=500)
        lengths = []
        for k in range(episodes):
            obs, _ = rollout_env.reset(seed=k)
            done, n = False, 0
            while not done:
                obs, r, term, trunc, _ = rollout_env.step(policy(obs))
                n += 1
                done = term or trunc
            lengths.append(n)
        return float(np.mean(lengths))

    rnd = np.random.default_rng(0)
    print(f"    随机策略 : 平均存活 {rollout(lambda o: rnd.uniform(-1, 1, 1), 20):5.1f} / 500 步")
    print(f"    PD 控制器: 平均存活 {rollout(_pd_controller, 20):5.1f} / 500 步")

    print()
    print("=" * 64)
    print("4) 向量环境：接口形状 + autoreset 语义")
    print("=" * 64)
    venv = ContinuousCartPoleVectorEnv(num_envs=2, max_episode_steps=3)
    print(f"    单环境动作空间 : {venv.single_action_space}")
    print(f"    批量动作空间   : {venv.action_space}")
    obs, _ = venv.reset(seed=0)
    for k in range(4):
        obs, rew, term, trunc, _ = venv.step(np.zeros((2, 1), dtype=np.float32))
        print(
            f"    step {k + 1}: rew {rew} term {term} trunc {trunc} "
            f"max|obs| {np.abs(obs).max():.3f}"
        )
    assert np.all(rew == 0.0) and not term.any() and not trunc.any()
    assert np.all(np.abs(obs) <= 0.05 + 1e-6)
    print("    → 重置帧：reward=0、done=False，观测就是新一局的初始状态（动作被丢弃）")
    venv.close()
    env.close()
