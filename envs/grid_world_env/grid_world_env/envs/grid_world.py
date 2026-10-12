from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    stay = 4


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(
        self,
        size=5,
        start_cell=None,
        target_cell=None,
        forbidden_cells=None,
        r_target=0,
        r_forbidden=-10,
        r_boundary=-10,
        r_other=-1,
        render_mode=None,
    ):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.r_target = r_target
        self.r_forbidden = r_forbidden
        self.r_boundary = r_boundary
        self.r_other = r_other

        self._fixed_target_location = (
            np.array(target_cell, dtype=int) if target_cell is not None else None
        )
        self._fixed_agent_location = (
            np.array(start_cell, dtype=int) if start_cell is not None else None
        )

        self._forbidden_locations = set()
        if forbidden_cells is not None:
            for cell in forbidden_cells:
                cell = tuple(int(v) for v in cell)

                assert (
                    0 <= cell[0] < self.size and 0 <= cell[1] < self.size
                ), "forbidden is out of boundary"

                self._forbidden_locations.add(cell)
        self._fixed_forbidden_locations = (
            self._forbidden_locations if forbidden_cells is not None else None
        )

        if self._fixed_agent_location is not None:
            assert (
                tuple(self._fixed_agent_location) not in self._forbidden_locations
            ), "start_cell should not be forbidden!"

        if self._fixed_target_location is not None:
            assert (
                tuple(self._fixed_target_location) not in self._forbidden_locations
            ), "target_cell should not be forbidden!"

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
            }
        )

        # We have 5 actions, corresponding to "right", "up", "left", "down", "right", "stay"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
            Actions.stay.value: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None

        """
        策略叠加层：调用 set_policy() 之后，_render_frame() 会在每格画出策略
        （箭头 = 动作方向，圆点 = stay），当前 agent 所在格的动作用高亮色叠在 agent 上。
        """
        self._policy_probs = None  # (size, size, n_actions) 概率，None 表示不叠加
        self._policy_show_ties = False
        self._policy_tie_tol = 1e-8
        self.policy_arrow_scale = 0.34  # 箭头长度（单位：格）
        self.policy_stay_radius = 0.14  # stay 圆点半径（单位：格）
        self.policy_color = (40, 40, 40)  # 策略本身的颜色
        self.policy_current_color = (220, 40, 40)  # 当前状态所选动作的高亮色

        """
        价值叠加层：调用 set_v() 之后，_render_frame() 会把 V(s) 写在每格左上角。
        """
        self._v_grid = None  # (size, size) 的价值，None 表示不叠加
        self.v_color = (95, 95, 95)  # 数值文字颜色
        self.v_bg_color = (255, 255, 255)  # 叠在 agent 圆点上时给文字垫的底色
        self.v_format = "{:.2f}"  # 数值格式
        self.v_font_scale = 0.2  # 字号 = 格子边长 * 该比例
        self._v_font = None  # 缓存 pygame 字体
        self._v_font_size = None

        if render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _is_out_of_boundary(self, agent_location):
        location = np.asarray(agent_location)
        return bool(np.any(location < 0) or np.any(location >= self.size))

    def _generate_random_forbidden(self, reserved, ratio_range=(0.05, 0.3)):
        total = self.size * self.size
        ratio = self.np_random.uniform(ratio_range[0], ratio_range[1])
        n_forbidden = int(round(total * ratio))

        candidates = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in reserved
        ]
        # 避免采样数超过可选数量
        n_forbidden = min(n_forbidden, len(candidates))

        # 不重复抽样
        idx = self.np_random.choice(len(candidates), size=n_forbidden, replace=False)
        return {candidates[i] for i in idx}

    def reset(self, *, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random

        self._agent_location = (
            self.np_random.integers(0, self.size, size=2, dtype=int)
            if self._fixed_agent_location is None
            else self._fixed_agent_location.copy()
        )

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = (
            self.np_random.integers(0, self.size, size=2, dtype=int)
            if self._fixed_target_location is None
            else self._fixed_target_location.copy()
        )

        if self._fixed_forbidden_locations is None:
            reserved = {tuple(self._agent_location), tuple(self._target_location)}
            self._forbidden_locations = self._generate_random_forbidden(reserved)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        terminated = False

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid

        self._agent_location = self._agent_location + direction

        if np.array_equal(self._agent_location, self._target_location):
            reward = self.r_target
            terminated = True
        elif self._is_out_of_boundary(self._agent_location):
            reward = self.r_boundary
            self._agent_location = self._agent_location - direction
        elif tuple(self._agent_location) in self._forbidden_locations:
            reward = self.r_forbidden
            self._agent_location = self._agent_location - direction
        else:
            reward = self.r_other

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):

        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _cell_to_screen(self, cell, pix_square_size):
        """逻辑坐标 (x, y)，y 向上增大；转成 pygame 屏幕坐标（y 向下增大）。"""
        x, y = int(cell[0]), int(cell[1])
        screen_x = x * pix_square_size
        screen_y = (self.size - 1 - y) * pix_square_size
        return screen_x, screen_y

    # ------------------------------------------------------------------
    # 策略叠加渲染
    # ------------------------------------------------------------------
    def _policy_to_action_probs(self, policy):
        """把不同写法的策略统一成 ``(size, size, n_actions)`` 的概率张量。

        约定与 ``build_mdp()`` 一致：``s = x * size + y``。支持四种输入：

        ==============================  ==================================
        形状                            含义
        ==============================  ==================================
        ``(n_states, n_actions)``       策略概率（如 ``PolicyIteration.pi``）
        ``(n_states,)``                 每状态的动作编号
        ``(size, size)``                每格的动作编号
        ``(size, size, n_actions)``     策略概率
        ==============================  ==================================
        """
        arr = np.asarray(policy)
        n_actions = self.action_space.n  # type: ignore
        n_states = self.size**2

        if arr.ndim == 3 and arr.shape == (self.size, self.size, n_actions):
            return arr.astype(float)
        if arr.ndim == 2 and arr.shape == (n_states, n_actions):
            return arr.astype(float).reshape(self.size, self.size, n_actions)
        if arr.ndim == 2 and arr.shape == (self.size, self.size):
            return self._action_indices_to_probs(arr)
        if arr.ndim == 1 and arr.shape[0] == n_states:
            return self._action_indices_to_probs(arr.reshape(self.size, self.size))

        raise ValueError(
            f"无法识别的策略形状 {arr.shape}，期望 (n_states,)、(n_states, {n_actions})、"
            f"({self.size}, {self.size}) 或 ({self.size}, {self.size}, {n_actions})"
        )

    def _action_indices_to_probs(self, indices):
        """动作编号网格 ``(size, size)`` -> one-hot 概率 ``(size, size, n_actions)``。"""
        idx = np.asarray(indices).astype(int)
        n_actions = self.action_space.n  # type: ignore
        if np.any(idx < 0) or np.any(idx >= n_actions):
            raise ValueError(
                f"动作编号必须在 [0, {n_actions - 1}] 内，收到 {idx.min()}~{idx.max()}"
            )

        probs = np.zeros((self.size, self.size, n_actions))
        xs, ys = np.meshgrid(np.arange(self.size), np.arange(self.size), indexing="ij")
        probs[xs, ys, idx] = 1.0
        return probs

    def set_policy(self, policy, *, show_ties=False, tie_tol=1e-8):
        """设置要在渲染里叠加显示的策略；传 ``None`` 表示清除。

        ``render_mode="human"``（或 ``"rgb_array"``）时，每格画出该状态的最优动作：
        **箭头 = 动作方向，圆点 = stay**；agent 所在格的动作用 ``policy_current_color``
        （默认红色）高亮叠在 agent 上，这样能直接看到 agent 每一步实际执行的动作。

        ``policy`` 的形状见 :meth:`_policy_to_action_probs`，
        ``PolicyIteration.policy_iteration()`` 返回的 ``pi`` 可直接传入。
        """
        if policy is None:
            self._policy_probs = None
            return

        self._policy_probs = self._policy_to_action_probs(policy)
        self._policy_show_ties = show_ties
        self._policy_tie_tol = tie_tol

    @property
    def policy(self):
        """当前叠加渲染的策略 ``(size, size, n_actions)``，未设置时为 ``None``。"""
        return self._policy_probs

    def set_v(self, v, *, fmt=None):
        """设置要在渲染里叠加显示的状态价值 ``V(s)``；传 ``None`` 表示清除。

        ``render_mode="human"``（或 ``"rgb_array"``）时，会把 ``V(s)`` 按 ``v_format``
        格式化后写在每格左上角（禁区格子跳过，和策略层一致）。

        ``v`` 接受长度 ``n_states`` 的一维数组/列表（``ValueIteration.v``、
        ``PolicyIteration.v`` 可直接传入），或 ``(size, size)`` 的二维数组；
        统一按 ``s = x * size + y`` 解释，与 :meth:`build_mdp` 的 ``cells`` 顺序一致。
        """
        if v is None:
            self._v_grid = None
            return

        arr = np.asarray(v, dtype=float)
        if arr.size != self.size**2:
            raise ValueError(
                f"v 的元素个数应为 {self.size**2}（或形状 ({self.size}, {self.size})），"
                f"收到 {arr.shape}"
            )

        self._v_grid = arr.reshape(self.size, self.size)
        if fmt is not None:
            self.v_format = fmt

    @property
    def v(self):
        """当前叠加渲染的价值 ``(size, size)``，未设置时为 ``None``。"""
        return self._v_grid

    def _policy_cell_actions(self, x, y):
        """算出 ``(x, y)`` 这格要画的动作。

        返回 ``(actions, spread_axis, offsets, length_factor, radius_factor)``：
        ``spread_axis`` / ``offsets`` 用于把并列最优的多个动作并排错开。
        """

        probs = self._policy_probs

        assert probs is not None

        if self._policy_show_ties:
            actions = np.flatnonzero(probs[x, y] > self._policy_tie_tol)
            if actions.size == 0:  # 概率全 0 的退化情况
                actions = np.array([int(np.argmax(probs[x, y]))])
        else:
            actions = np.array([int(np.argmax(probs[x, y]))])

        if actions.size > 1:
            # 并列最优：沿“第一个动作”的法向并排错开（统一错开轴，反向并列也不会重叠）
            first_direction = np.asarray(
                self._action_to_direction[int(actions[0])], dtype=float
            )
            spread_axis = np.array([-first_direction[1], first_direction[0]])
            if not np.any(spread_axis):  # 第一个动作是 stay，退化成水平错开
                spread_axis = np.array([1.0, 0.0])

            crowded = actions.size > 2  # 三个以上动作时缩小符号、加大间距
            spread = 0.3 if crowded else 0.2
            offsets = np.linspace(-spread, spread, actions.size)
            length_factor = 0.5 if crowded else 0.7
            radius_factor = 0.6 if crowded else 1.0
        else:
            spread_axis = np.zeros(2)
            offsets = np.zeros(1)
            length_factor = 1.0
            radius_factor = 1.0

        return actions, spread_axis, offsets, length_factor, radius_factor

    @staticmethod
    def _draw_arrow(canvas, start, end, color, line_width, head_width, head_length):
        """在 pygame 画布上画一根带三角箭头的线段。"""
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        vector = end - start
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return

        unit = vector / norm
        normal = np.array([-unit[1], unit[0]])
        head_base = end - unit * min(head_length, norm * 0.6)

        pygame.draw.line(
            canvas,
            color,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(head_base[0])), int(round(head_base[1]))),
            width=line_width,
        )
        pygame.draw.polygon(
            canvas,
            color,
            [
                (int(round(end[0])), int(round(end[1]))),
                tuple(int(v) for v in np.round(head_base + normal * head_width / 2)),
                tuple(int(v) for v in np.round(head_base - normal * head_width / 2)),
            ],
        )

    def _draw_values(self, canvas, pix_square_size, cells=None, background=False):
        """把 ``set_v()`` 设好的 ``V(s)`` 写在每格左上角。

        ``cells`` 不传时画所有非禁区格子；``background=True`` 会在文字下面垫一块
        ``v_bg_color`` 底色 —— agent 所在格的数值要叠在蓝色圆点上，垫底色才读得清。
        """
        if self._v_grid is None:
            return

        if not pygame.font.get_init():
            pygame.font.init()

        font_size = max(8, int(round(pix_square_size * self.v_font_scale)))
        if self._v_font is None or self._v_font_size != font_size:
            self._v_font = pygame.font.Font(None, font_size)
            self._v_font_size = font_size

        if cells is None:
            cells = [
                (x, y)
                for x in range(self.size)
                for y in range(self.size)
                if (x, y) not in self._forbidden_locations
            ]

        for x, y in cells:
            text = self._v_font.render(
                self.v_format.format(self._v_grid[x, y]), True, self.v_color
            )
            screen_x, screen_y = self._cell_to_screen((x, y), pix_square_size)
            position = (
                screen_x + pix_square_size * 0.06,
                screen_y + pix_square_size * 0.04,
            )

            if background:
                pad = max(1, int(round(pix_square_size * 0.02)))
                chip = pygame.Surface(
                    (text.get_width() + 2 * pad, text.get_height() + 2 * pad)
                )
                chip.fill(self.v_bg_color)
                canvas.blit(chip, (position[0] - pad, position[1] - pad))

            canvas.blit(text, position)

    def _draw_policy(self, canvas, pix_square_size, cells=None, color=None):
        """把 ``set_policy()`` 设好的策略画到 canvas 上：每格箭头 / stay 圆点。"""
        if self._policy_probs is None:
            return

        color = self.policy_color if color is None else color
        if cells is None:
            cells = [
                (x, y)
                for x in range(self.size)
                for y in range(self.size)
                if (x, y) not in self._forbidden_locations
            ]

        line_width = max(1, int(round(pix_square_size * 0.045)))
        head_width = pix_square_size * 0.18
        head_length = pix_square_size * 0.15

        for x, y in cells:
            screen_x, screen_y = self._cell_to_screen((x, y), pix_square_size)
            base_center = np.array(
                [screen_x + pix_square_size / 2, screen_y + pix_square_size / 2]
            )
            actions, spread_axis, offsets, length_factor, radius_factor = (
                self._policy_cell_actions(x, y)
            )
            length = pix_square_size * self.policy_arrow_scale * length_factor
            radius = pix_square_size * self.policy_stay_radius * radius_factor

            for action, offset in zip(actions, offsets):
                # 逻辑坐标 -> 屏幕坐标：y 轴要翻转
                center = (
                    base_center
                    + np.array([spread_axis[0] * offset, -spread_axis[1] * offset])
                    * pix_square_size
                )
                direction = np.asarray(
                    self._action_to_direction[int(action)], dtype=float
                )

                if not np.any(direction):  # stay -> 圆点
                    pygame.draw.circle(
                        canvas,
                        color,
                        (int(round(center[0])), int(round(center[1]))),
                        int(round(radius)),
                    )
                    continue

                screen_direction = np.array([direction[0], -direction[1]])
                start = center - screen_direction * length / 2
                end = center + screen_direction * length / 2
                self._draw_arrow(
                    canvas, start, end, color, line_width, head_width, head_length
                )

    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            ((103, 255, 254)),
            pygame.Rect(
                (self._cell_to_screen(self._target_location, pix_square_size)),
                (pix_square_size, pix_square_size),
            ),
        )

        # 画 forbidden
        for x, y in self._forbidden_locations:
            pygame.draw.rect(
                canvas,
                ((254, 192, 1)),
                pygame.Rect(
                    (self._cell_to_screen((x, y), pix_square_size)),
                    (pix_square_size, pix_square_size),
                ),
            )

        # 价值叠加层（背景）：V(s) 写在每格左上角（未调用 set_v() 时什么也不画）
        self._draw_values(canvas, pix_square_size)

        # 策略叠加层：箭头 = 动作方向，圆点 = stay（未调用 set_policy() 时什么也不画）
        self._draw_policy(canvas, pix_square_size)

        # Now we draw the agent
        sx, sy = self._cell_to_screen(self._agent_location, pix_square_size)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (sx + pix_square_size / 2, sy + pix_square_size / 2),
            pix_square_size / 3,
        )

        # 再把 agent 当前状态所选的动作用高亮色叠在 agent 上，方便看它这一步在干嘛
        if self._policy_probs is not None:
            self._draw_policy(
                canvas,
                pix_square_size,
                cells=[(int(self._agent_location[0]), int(self._agent_location[1]))],
                color=self.policy_current_color,
            )

        # agent 所在格的 V(s) 会被圆点盖住，最后垫一层底色重画一遍
        if self._v_grid is not None:
            self._draw_values(
                canvas,
                pix_square_size,
                cells=[(int(self._agent_location[0]), int(self._agent_location[1]))],
                background=True,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
            )

        if self.render_mode == "human":

            if self.window is not None:
                # The following line copies our drawings from `canvas` to the visible window
                self.window.blit(canvas, canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])

        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def build_mdp(self):
        cells = [(x, y) for x in range(self.size) for y in range(self.size)]
        states_idx = {cell: i for i, cell in enumerate(cells)}

        n_states = self.size**2
        n_actions = self.action_space.n  # type: ignore

        P = np.zeros((n_states, n_actions, n_states))
        R = np.zeros((n_states, n_actions, n_states))

        for x, y in cells:
            for action in range(n_actions):
                direction = self._action_to_direction[action]
                nx, ny = x + direction[0], y + direction[1]

                if self._is_out_of_boundary((nx, ny)):
                    nx, ny = x, y
                    reward = self.r_boundary
                elif (nx, ny) in self._forbidden_locations:
                    nx, ny = x, y
                    reward = self.r_forbidden
                elif (nx, ny) == tuple(self._target_location):
                    reward = self.r_target
                else:
                    reward = self.r_other

                state_id = states_idx[(x, y)]
                n_state_id = states_idx[(nx, ny)]
                P[state_id, action, n_state_id] = 1.0
                R[state_id, action, n_state_id] = reward

        return P, R, cells, states_idx

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
