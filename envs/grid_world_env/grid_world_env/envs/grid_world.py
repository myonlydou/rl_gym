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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        size=5,
        start_cell=None,
        target_cell=None,
        forbidden_cells=None,
        render_mode=None,
    ):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self._fixed_target_location = (
            np.array(target_cell, dtype=int) if target_cell is not None else None
        )
        self._fixed_agent_location = (
            np.array(start_cell, dtype=int) if start_cell is not None else None
        )

        self._forbidden_location = set()
        if forbidden_cells is not None:
            for cell in forbidden_cells:
                cell = tuple(int(v) for v in cell)

                assert (
                    0 <= cell[0] < self.size and 0 <= cell[1] < self.size
                ), "forbidden is out of boundary"

                self._forbidden_location.add(cell)

        if self._fixed_agent_location is not None:
            assert (
                tuple(self._fixed_agent_location) not in self._forbidden_location
            ), "start_cell should not be forbidden!"

        if self._fixed_target_location is not None:
            assert (
                tuple(self._fixed_target_location) not in self._forbidden_location
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

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid

        self._agent_location = self._agent_location + direction

        if np.array_equal(self._agent_location, self._target_location):
            reward = 1
        elif tuple(self._agent_location) in self._forbidden_location:
            reward = -1
            self._agent_location = self._agent_location - direction
        elif self._is_out_of_boundary(self._agent_location):
            reward = -1
            self._agent_location = np.clip(self._agent_location, 0, self.size - 1)
        else:
            reward = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, False, False, info

    def render(self):

        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _cell_to_screen(self, cell, pix_square_size):
        """逻辑坐标 (x, y)，y 向上增大；转成 pygame 屏幕坐标（y 向下增大）。"""
        x, y = int(cell[0]), int(cell[1])
        screen_x = x * pix_square_size
        screen_y = (self.size - 1 - y) * pix_square_size
        return screen_x, screen_y

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
        for x, y in self._forbidden_location:
            pygame.draw.rect(
                canvas,
                ((254, 192, 1)),
                pygame.Rect(
                    (self._cell_to_screen((x, y), pix_square_size)),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Now we draw the agent
        sx, sy = self._cell_to_screen(self._agent_location, pix_square_size)
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (sx + pix_square_size / 2, sy + pix_square_size / 2),
            pix_square_size / 3,
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

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
