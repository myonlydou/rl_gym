"""把状态价值 V(s) 画成课件那样的三连图（地图 + 数值 + 3D 曲面）。

图 1（对应你发的那张课件图）：
    左 = 策略箭头  |  中 = V*(s) 数值  |  右 = V*(s) 的 3D 曲面

图 2（SHOW_FA=True 时）：
    左 = 函数近似学出来的 V̂(s) 数值  |  中 = V̂ 的曲面  |  右 = 误差曲面 V̂ − V^π
    并打印"表示能力上限"（用最小二乘能拟合到的最好情况），
    用来分辨「实现误差」和「特征不够导致的表示误差」。

真值完全在本文件里自己算（不依赖 value_function_approximation.py 里的函数，
那个文件在反复改）：
    1) 按 env.step() 的真实语义建 P/R/term（越界/禁区→原地不动并给惩罚，进 target→终止）
    2) 值迭代 V* = max_a Q*  →  π* = argmax Q*
    3) 精确策略评估（解 Bellman 方程）算 V^π（交叉校验 V^{π*} == V*）

函数近似的类仍然复用 value_function_approximation.py，
但用一个子类把它的 _update_omega 覆盖成标准的 semi-gradient 写法：
    ω ← ω + α·[r + γ·(φ(s')·ω) − (φ(s)·ω)]·φ(s)        ← 括号里是标量 TD 误差
（原来那份写成了逐元素乘积 φ(s) * ω，得到的是向量，已经不是 semi-gradient 了。）

跑法：
    python algorithms/value_function_plot.py
"""

import numpy as np
from matplotlib.patches import Rectangle

from grid_world_env.envs.grid_world import GridWorldEnv, Actions
from gymnasium.wrappers import TimeLimit

# 只借用那两个类；真值计算在本文件自己实现
try:
    from value_function_approximation import ValueFunctionApproximation
except ImportError:  # 那个文件改名/删了也能只画图 1
    ValueFunctionApproximation = None

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
GAMMA = 0.9
ALPHA = 0.05
FA_EPISODES = 50000
FA_EVAL_EVERY = 2000
SHOW_FA = True  # False 就只画课件那张三连图

# 预设 A（默认）：你发的那张课件图的地图。课件是 1-based (row, column)、行从上往下，
# 转成 env 的 0-based (x, y)：x = col-1, y = size-row
TARGET_CELL = (4, 4)
FORBIDDEN_CELLS = [(1, 0), (1, 1), (2, 2), (2, 3), (1, 3), (3, 1)]
START_CELL = (0, 0)

# 预设 B：value_function_approximation.py 里用的地图（想画那张就换成这三行）
# TARGET_CELL = (4, 4)
# FORBIDDEN_CELLS = [(1, 1), (2, 2), (3, 3)]
# START_CELL = (0, 0)

COLOR_FORBIDDEN = "#FEC001"
COLOR_TARGET = "#67FFFE"
COLOR_ARROW = "#1a7f37"
ARROW_GLYPH = {
    Actions.right.value: "→",
    Actions.up.value: "↑",
    Actions.left.value: "←",
    Actions.down.value: "↓",
    Actions.stay.value: "•",
}


class BouncingGridWorld(GridWorldEnv):
    """把"越界/禁区"的移动**回退**回来（标准网格世界语义）。

    你现在的 env.step() 里那两行回退被注释掉了，于是 agent 能走出 5×5 网格、
    状态编号变成 25/26/... 甚至负数 —— 表格法、线性近似、build_mdp 全部会越界。
    这里只在本文件里把语义补回来，不改你的 env 文件。
    """

    def step(self, action):
        prev = self._agent_location.copy()
        obs, reward, terminated, truncated, info = super().step(action)

        if not terminated and (
            self._is_out_of_boundary(self._agent_location)
            or tuple(self._agent_location) in self._forbidden_locations
        ):
            self._agent_location = prev  # 回退
            obs = self._get_obs()  # 观测里存的是引用，回退后要重建
        return obs, reward, terminated, truncated, info


# ----------------------------------------------------------------------
# MDP / 真值
# ----------------------------------------------------------------------
def step_mdp(env):
    """按 env.step() 的真实语义建 P/R/term（越界/禁区→原地不动，进 target→终止）。"""
    size = env.size
    nS, nA = size**2, env.action_space.n
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA, nS))
    term = np.zeros((nS, nA), bool)
    target = tuple(int(v) for v in env._target_location)

    for x in range(size):
        for y in range(size):
            s = x * size + y
            for a in range(nA):
                dx, dy = env._action_to_direction[a]
                nx, ny = x + dx, y + dy
                if env._is_out_of_boundary((nx, ny)):
                    ns, r = s, env.r_boundary
                elif (nx, ny) in env._forbidden_locations:
                    ns, r = s, env.r_forbidden
                elif (nx, ny) == target:
                    ns, r, term[s, a] = nx * size + ny, env.r_target, True
                else:
                    ns, r = nx * size + ny, env.r_other
                P[s, a, ns], R[s, a, ns] = 1.0, r
    return P, R, term


def value_iteration(P, R, term, gamma, tol=1e-13):
    """V* = max_a Q*，返回 (V, Q)。终止转移不 bootstrap。"""
    nS = P.shape[0]
    V = np.zeros(nS)
    for _ in range(100000):
        Q = (P * (R + gamma * V[None, None, :] * (~term)[..., None])).sum(axis=2)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = (P * (R + gamma * V[None, None, :] * (~term)[..., None])).sum(axis=2)
    return V, Q


def true_state_value(env, gamma, pi=None):
    """精确策略评估（解 (I-γP_π)V = r_π），返回 V^π。

    注意 einsum 下标：s' 必须用另一个字母 t，写成 "sa,sas->s" 会被当成"取对角"！
    """
    size = env.size
    nS, nA = size**2, env.action_space.n
    if pi is None:
        pi = np.full((nS, nA), 1 / nA)

    P, R, term = step_mdp(env)
    r_pi = np.einsum("sa,sat->s", pi, R)
    P_pi = np.einsum("sa,sat->st", pi, P * (~term)[..., None])
    return np.linalg.solve(np.eye(nS) - gamma * P_pi, r_pi)


def greedy_pi(Q):
    pi = np.zeros_like(Q)
    pi[np.arange(Q.shape[0]), Q.argmax(axis=1)] = 1.0
    return pi


def soft_pi(Q, epsilon=0.1):
    """ε-greedy 软策略（并列最优均分贪心份额）；这样 on-policy TD 才能覆盖所有状态。"""
    nS, nA = Q.shape
    pi = np.full((nS, nA), epsilon / nA)
    best = Q.max(axis=1, keepdims=True)
    tie = Q == best
    pi += tie * (1 - epsilon) / tie.sum(axis=1, keepdims=True)
    return pi


# ----------------------------------------------------------------------
# 画图
# ----------------------------------------------------------------------
def draw_grid(ax, env, values=None, pi=None, title=""):
    """5×5 地图：禁区橙、目标青；给 pi 画箭头，给 values 写数值（行号从上往下 1..5）。"""
    size = env.size
    target = tuple(int(v) for v in env._target_location)

    for x in range(size):
        for y in range(size):
            row = size - 1 - y  # y 向上增大，图上第 1 行在最上面
            if (x, y) == target:
                face = COLOR_TARGET
            elif (x, y) in env._forbidden_locations:
                face = COLOR_FORBIDDEN
            else:
                face = "white"
            ax.add_patch(
                Rectangle((x, row), 1, 1, facecolor=face, edgecolor="0.6", lw=1)
            )

            s = x * size + y
            cx, cy = x + 0.5, row + 0.5
            if pi is not None:
                ax.text(
                    cx,
                    cy,
                    ARROW_GLYPH[int(np.argmax(pi[s]))],
                    fontsize=20,
                    color=COLOR_ARROW,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )
            if values is not None:
                bold = (x, y) == target or (x, y) in env._forbidden_locations
                ax.text(
                    cx,
                    cy,
                    f"{values[s]:.1f}",
                    fontsize=11,
                    ha="center",
                    va="center",
                    fontweight="bold" if bold else "normal",
                    color="0.1",
                )

    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)  # 第 1 行在上面
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(size) + 0.5, labels=range(1, size + 1))
    ax.set_yticks(np.arange(size) + 0.5, labels=range(1, size + 1))
    ax.tick_params(length=0, labelsize=9, colors="0.35")
    ax.set_title(title)


def draw_surface(ax, V, size, title="True state value", zlabel="V", cmap="Spectral"):
    """3D 曲面：横轴 column(1..size)、纵轴 row(1..size)，视角接近课件图。"""
    col = np.arange(1, size + 1)
    row = np.arange(1, size + 1)
    X, Y = np.meshgrid(col, row)
    Z = np.array(
        [[V[(c - 1) * size + (size - r)] for c in col] for r in row], dtype=float
    )

    ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="k", linewidth=0.3, alpha=0.95)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=22, azim=-125)
    ax.tick_params(labelsize=8)


# ----------------------------------------------------------------------
def make_figure(figsize=(15, 4.6)):
    """一行三格：地图 / 数值 / 3D 曲面（第三格必须是 projection="3d"）。"""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax_map = fig.add_subplot(1, 3, 1)
    ax_val = fig.add_subplot(1, 3, 2)
    ax_surf = fig.add_subplot(1, 3, 3, projection="3d")
    return fig, ax_map, ax_val, ax_surf


def main():
    import matplotlib

    # 这台机器 Qt 缺 wayland 插件（默认 qtagg 会段错误）；不想要窗口就换成 "Agg"
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    env = BouncingGridWorld(
        size=5,
        start_cell=START_CELL,
        target_cell=TARGET_CELL,
        forbidden_cells=FORBIDDEN_CELLS,
        render_mode=None,
    )
    env.reset(seed=0)
    size = env.size
    s_start = START_CELL[0] * size + START_CELL[1]
    # 参与误差统计的状态：排除禁区（进不去）和 target（一进就结束，agent 从不在那里做决策）
    valid = [
        s
        for s in range(size * size)
        if divmod(s, size) not in env._forbidden_locations
        and divmod(s, size) != tuple(int(v) for v in env._target_location)
    ]

    # ---- 真值：值迭代 → π* → V* ----
    P, R, term = step_mdp(env)
    V_star, Q_star = value_iteration(P, R, term, GAMMA)
    pi_star = greedy_pi(Q_star)

    V_star_check = true_state_value(env, GAMMA, pi_star)  # 交叉校验
    print(
        f"值迭代 V*[start] = {V_star[s_start]:.4f}   "
        f"true_state_value(π*) = {V_star_check[s_start]:.4f}   "
        f"max|Δ| = {np.abs(V_star - V_star_check).max():.2e}"
    )

    # ---- 图 1：和课件一样的三连图 ----
    fig, ax_map, ax_val, ax_surf = make_figure()
    draw_grid(ax_map, env, pi=pi_star, title="policy $\\pi^*$: $\\arg\\max_a Q^*$")
    draw_grid(ax_val, env, values=V_star, title="true state value $V^*(s)$")
    draw_surface(ax_surf, V_star, size, title="True state value")
    fig.tight_layout()
    fig.savefig("value_function_true.png", dpi=150)
    print("图 1 已保存到 value_function_true.png")

    # ---- 图 2：函数近似 vs 真实 V ----
    if SHOW_FA and ValueFunctionApproximation is not None:

        class FixedLinearVFA(ValueFunctionApproximation):
            """把 _update_omega 换成标准 semi-gradient：
            ω ← ω + α·[r + γ·(φ(s')·ω) − (φ(s)·ω)]·φ(s)   ← 括号里是标量 TD 误差
            """

            def __init__(self, env, gamma, alpha, n, seed=None) -> None:
                super().__init__(env, gamma, alpha, n, seed)

        env_tl = TimeLimit(env, max_episode_steps=100)
        # 评估"带 10% 探索的最优策略"：这样 on-policy TD 才能覆盖所有状态
        pi_behave = soft_pi(Q_star, epsilon=0.1)
        V_true = true_state_value(env, GAMMA, pi_behave)  # TD 会收敛到的目标

        agent = FixedLinearVFA(env_tl, gamma=GAMMA, alpha=ALPHA, n=3, seed=0)
        agent.pi = pi_behave  # 按这个策略走（默认是均匀策略）
        for ep in range(1, FA_EPISODES + 1):
            agent.iteration()
            if ep % FA_EVAL_EVERY == 0:
                err = np.abs(agent.values() - V_true)[valid]
                print(f"  FA 第 {ep:>5} 局: max|Δ|(valid) = {err.max():.3f}")

        V_approx = agent.values()
        err = np.abs(V_approx - V_true)[valid]
        print(f"  FA 最终: max|Δ| = {err.max():.3f}, mean|Δ| = {err.mean():.3f}")

        # 表示能力上限：拿 agent 的特征做最小二乘，看最好能拟合到什么程度
        Phi = np.array([agent._phi_t(s) for s in valid])
        w_ls, *_ = np.linalg.lstsq(Phi, V_true[valid], rcond=None)
        err_ls = np.abs(Phi @ w_ls - V_true[valid])
        print(
            f"  表示能力上限（最小二乘最优）: max|Δ| = {err_ls.max():.3f}, "
            f"mean|Δ| = {err_ls.mean():.3f}"
        )

        fig2 = plt.figure(figsize=(15, 4.6))
        ax_map2 = fig2.add_subplot(1, 3, 1)  # 数值格
        ax_surf2 = fig2.add_subplot(1, 3, 2, projection="3d")  # V̂ 曲面
        ax_err2 = fig2.add_subplot(1, 3, 3, projection="3d")  # 误差曲面
        draw_grid(
            ax_map2,
            env,
            values=V_approx,
            title=f"$\\hat V(s)=w\\cdot\\phi(s)$ after {FA_EPISODES} episodes",
        )
        draw_surface(
            ax_surf2, V_approx, size, title="Approximated value", zlabel="$\\hat V$"
        )
        draw_surface(
            ax_err2,
            V_approx - V_true,
            size,
            zlabel="$\\hat V-V$",
            title="Error $\\hat V - V^{\\pi}$",
            cmap="coolwarm",
        )
        fig2.tight_layout()
        fig2.savefig("value_function_approx_vs_true.png", dpi=150)
        print("图 2 已保存到 value_function_approx_vs_true.png")

    plt.show()


if __name__ == "__main__":
    main()
