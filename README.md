# rl_gym

> 学习 **《强化学习的数学原理》——从零开始到透彻理解（完结）** 之后，对强化学习基本算法的练习代码。
> 环境基于 [Gymnasium](https://gymnasium.farama.org)（OpenAI Gym 的官方继任者，`gym` 的维护版）实现，算法部分刻意用最朴素的表格法手写，
> 目的是把课程里的公式一行一行落到代码上，而不是调库。

- 课程视频：[【强化学习的数学原理】课程视频+书籍+PPT——从零开始透彻理解强化学习](https://www.bilibili.com/read/cv18745978/)
- 配套书籍：[*Mathematical Foundations of Reinforcement Learning*](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)（赵世钰，西湖大学）

## 目录结构

```
rl_gym/
├── algorithms/
│   ├── policy_iteration.py      # 表格法策略迭代（策略评估 + 策略改进）已实现
│   └── value_iteration.py       # 值迭代（占位，待实现）
├── envs/
│   └── grid_world_env/          # 自定义 Gymnasium 环境包（可 pip install -e）
│       ├── grid_world_env/
│       │   ├── __init__.py      # 注册 grid_world_env/GridWorld-v0
│       │   ├── envs/
│       │   │   └── grid_world.py    # GridWorldEnv + build_mdp()
│       │   └── wrappers/            # Gymnasium 官方示例 wrapper（原样保留）
│       ├── pyproject.toml
│       └── LICENSE              # MIT (c) 2023 Farama Foundation
└── README.md
```

---

## 环境：`grid_world_env`

在 Gymnasium 官方[环境创建示例](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)的 `GridWorldEnv` 基础上，增加了 **forbidden cell（禁区）**。

### 网格与动作

- `size × size` 网格（默认 5×5），坐标为 `(x, y)`，`x` 向右增大、`y` 向上增大。
- 状态编号：`cells = [(x, y) for x in range(size) for y in range(size)]`，即 `s = x * size + y`。
- 动作空间 `Discrete(5)`，对应 `Actions` 枚举：

  | 值 | 动作 | 方向向量 |
  |---|---|---|
  | 0 | `right` | `(1, 0)` |
  | 1 | `up` | `(0, 1)` |
  | 2 | `left` | `(-1, 0)` |
  | 3 | `down` | `(0, -1)` |
  | 4 | `stay` | `(0, 0)` |

### 奖励与终止

| 情况 | 奖励 | 位置变化 |
|---|---|---|
| 走到 `target` | `+1` | 进入目标格 |
| 走出边界 / 撞上 `forbidden` | `-1` | 留在原地（回退该动作） |
| 其他 | `0` | 正常移动 |

`step()` 固定返回 `(obs, reward, False, False, info)`：**目标格不是吸收态**，到达后继续 `stay` 会不断拿到 `+1`（因此 `v(target) = 1/(1-γ)`）。需要 `TimeLimit` 或自行判停。

### 观测与注册

- 观测：`Dict(agent=Box(0, size-1, (2,)), target=Box(0, size-1, (2,)))`
- 注册 id：`grid_world_env/GridWorld-v0`（`max_episode_steps=100`）
- `start_cell` / `target_cell` / `forbidden_cells` 不传则**随机生成**；`forbidden` 数量为 `round(size²·U(0.05, 0.3))`（5×5 时实测 1~7 个）。

### `build_mdp()` 的 MDP 约定

```python
P, R, cells, states_idx = env.build_mdp()
```

| 数组 | 形状 | 含义 |
|---|---|---|
| `P[s, a, s']` | `(S, A, S)` | 转移概率，本环境中每条 `(s, a)` 只有一条概率为 `1.0` 的确定性转移（行和为 1） |
| `R[s, a, s']` | `(S, A, S)` | **逐转移（per-transition）奖励** |

因为 `R` 是 per-transition 的，Bellman 期望应写成

$$q_\pi(s,a)=\sum_{s'} p(s'\mid s,a)\bigl[R(s,a,s')+\gamma v_\pi(s')\bigr]$$

即 `qsa += p * R[s][a][s'] + gamma * p * v[s']`。**不要**把它和 `R[s, a]` 标量（`R(s,a)` 与 `s'` 无关）的写法混用：
标量约定下 `R` 只能在概率求和之外加一次，写成 `Σ p·R` 会在 `Σp ≠ 1` 时静默算错。

---

## 算法

### Policy Iteration（`algorithms/policy_iteration.py`）✅

```python
it = PolicyIteration(P, R, theta=0.001, gamma=0.9)
pi = it.policy_iteration()
```

| 方法 | 说明 |
|---|---|
| `_policy_evaluation()` | 迭代策略评估。同步（Jacobi）更新：`next_v` 与 `self.v` 分开，一轮扫完再整体写回；用 sup-norm 判停 `max_s|v_new(s) − v_old(s)| < theta` |
| `_policy_improvement()` | 策略改进。对每个状态取 `argmax_a q(s, a)`；并列最优的动作均分概率 |
| `policy_iteration()` | 交替执行 `评估 → 改进`，直到 `old_pi == new_pi`（策略不再变化，即收敛到最优策略） |

停止准则的误差上界为 $\lVert v-v^\* \rVert_\infty \le \dfrac{\theta\gamma}{1-\gamma}$：`theta=1e-3, gamma=0.9` 时约 `9e-3`，实测误差与该上界吻合。

### Value Iteration（`algorithms/value_iteration.py`）🚧

占位文件，待实现。

---

## 安装

```bash
conda create -n rl_env python=3.10 -y
conda activate rl_env

# 以可编辑模式安装自定义环境（自动带上 gymnasium / pygame）
pip install -e envs/grid_world_env
```

参考版本（本仓库开发环境）：Python 3.10.22、gymnasium 1.3.0、pygame 2.6.1、numpy 2.2.6、pandas 2.3.3。

## 运行

```bash
python algorithms/policy_iteration.py
```

程序会：`reset()` 环境 → `build_mdp()` 建表 → 策略迭代求出 `pi` → 在 `GridWorldEnv` 中按 `pi` 实跑一个 episode。



---

## 参考

- 课程：[【强化学习的数学原理】从零开始到透彻理解（完结）](https://www.bilibili.com/read/cv18745978/) ｜ [中国大学 MOOC 版](https://www.icourse163.org/course/XHUN-1470436188) ｜ [赵世钰实验室](https://shiyuzhao.westlake.edu.cn/Shiyu_Zhao.htm)
- 书籍主页：[MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
- 环境 API：[Gymnasium 官方文档](https://gymnasium.farama.org) ｜ [环境创建教程](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- 代码参考：[动手学强化学习 Hands-on Reinforcement Learning](https://hrl.boyuai.com/)
- 环境包源码来自 Gymnasium 官方示例（MIT License, (c) 2023 Farama Foundation）

## TODO

- [x] 带 forbidden cell 的 `GridWorldEnv`
- [x] 策略迭代（Policy Iteration）
- [ ] 值迭代（Value Iteration）
- [ ] 蒙特卡洛方法（MC Prediction / MC Control）
- [ ] 时序差分：Sarsa、Q-learning、Expected Sarsa
- [ ] 值函数近似 / 策略梯度（Actor-Critic）
