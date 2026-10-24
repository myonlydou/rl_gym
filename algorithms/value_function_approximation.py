import grid_world_env
from gymnasium.wrappers import TimeLimit
import numpy as np


class ValueFunctionApproximation:

    def __init__(self, env, gamma, alpha, seed=None) -> None:
        self.env = env
        self.base_env: grid_world_env.GridWorldEnv = env.unwrapped

        self.size = self.base_env.size
        self.n_states = self.size**2
        self.n_actions = self.base_env.action_space.n  # type: ignore

        self.pi = np.full((self.n_states, self.n_actions), 1 / self.n_actions)

        self.omega = np.zeros(len(self._phi_t(0)))

        self.gamma = gamma
        self.alpha = alpha
        self.seed = seed

        self.visit_counts = np.zeros(self.n_states, dtype=np.int64)

        self.rng = np.random.default_rng(seed=seed)

    def _state2normalized_position(self, s):
        """返回 归一化 位置"""

        size = self.size
        x = s // size / (size - 1)
        y = s % size / (size - 1)

        return x, y

    def _phi_t(self, s):
        x, y = self._state2normalized_position(s)

        return np.asarray([1, x, y])

    def _state_value(self, s):
        return float(self._phi_t(s) @ self.omega)

    def values(self):
        return np.array([self._state_value(s) for s in range(self.n_states)])

    def _agent_state(self, obs):
        return obs["agent"][0] * self.size + obs["agent"][1]

    def _sample_action(self, s):
        return self.rng.choice(self.n_actions, p=self.pi[s])

    def _update_omega(self, s0, r, s1, terminated=False):
        delta = r - float(self._phi_t(s0) @ self.omega)
        if not terminated:
            delta += self.gamma * float(self._phi_t(s1) @ self.omega)

        self.omega = self.omega + self.alpha * delta * self._phi_t(s0)

    def iteration(self):
        obs, _ = self.env.reset(seed=self.seed)

        episode_over = False
        steps, total_reward = 0, 0.0

        s1 = self._agent_state(obs)
        while not episode_over:
            s0 = s1
            a0 = self._sample_action(s0)

            self.visit_counts[s0] += 1
            obs, r, terminated, truncated, _ = self.env.step(a0)

            s1 = self._agent_state(obs)
            self._update_omega(s0, r, s1, terminated)

            steps += 1
            total_reward += r
            episode_over = terminated or truncated

        return steps, total_reward


def draw_surface(
    ax, state_value, size, title="True state value", zlabel="V", cmap="Spectral"
):
    """3D 曲面：横轴 column(1..size)、纵轴 row(1..size)，视角接近课件图。"""
    col = np.arange(1, size + 1)
    row = np.arange(1, size + 1)
    X, Y = np.meshgrid(col, row)
    Z = np.array(
        [[state_value[(c - 1) * size + (size - r)] for c in col] for r in row],
        dtype=float,
    )

    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="k", linewidth=0.3, alpha=0.95)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=22, azim=-125)
    ax.tick_params(labelsize=8)


# ----------------------------------------------------------------------
def make_figure(figsize=(10, 4.6)):
    """一行三格：地图 / 数值 / 3D 曲面（第三格必须是 projection="3d"）。"""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax_surf_true = fig.add_subplot(1, 3, 1, projection="3d")
    ax_surf_approximation = fig.add_subplot(1, 3, 2, projection="3d")
    ax_surf_error = fig.add_subplot(1, 3, 3, projection="3d")
    return fig, ax_surf_true, ax_surf_approximation, ax_surf_error


if __name__ == "__main__":
    # 这台机器的 Qt 装得不全（PySide6 只有 xcb、没有 libqwayland），matplotlib 默认的
    # qtagg 后端会段错误。改用不依赖 Qt 的 Tk 后端；不想要窗口就换成 "Agg"。
    import matplotlib

    matplotlib.use("TkAgg")

    import matplotlib.pyplot as plt

    size = 5

    seed = 0
    gamma, alpha = 0.9, 0.001

    base_env = grid_world_env.GridWorldEnv(
        size=size,
        start_cell=(0, 0),
        target_cell=(1, 2),
        forbidden_cells=[(1, 0), (1, 1), (2, 2), (2, 3), (1, 3), (3, 1)],
        render_mode=None,
    )
    base_env.reset(seed=seed)
    env = TimeLimit(base_env, max_episode_steps=100)

    P, R, term, _, _ = base_env.build_mdp()

    it = ValueFunctionApproximation(env, gamma, alpha, seed)

    r_pi = np.einsum("sa,sat->s", it.pi, R)
    P_pi = np.einsum("sa,sat->st", it.pi, P * (~term)[..., None])

    true_state_value = np.linalg.solve(np.eye(it.n_states) - gamma * P_pi, r_pi)

    n_episodes, eval_every = 8000, 400
    for ep in range(1, n_episodes + 1):
        it.iteration()
        if ep % eval_every == 0:
            err = np.abs(it.values() - true_state_value)
            visited = it.visit_counts > 0
            print(f"第 {ep:>5} 局: average|Δ| = {np.average(err[visited]):.3f}")

    fig, ax_surf_true, ax_surf_approximation, ax_surf_error = make_figure()
    draw_surface(
        ax_surf_approximation, it.values(), size, title="Approximation state value"
    )
    draw_surface(ax_surf_true, true_state_value, size, title="True state value")
    fig.tight_layout()
    draw_surface(
        ax_surf_error, true_state_value - it.values(), size, "Error state value"
    )

    plt.show()

    env.close()
