# rl_gym

> 强化学习算法的手写实现：从动态规划到 Actor-Critic，一个算法一个文件。
> 刻意不调用 RL 库，把课程里的公式一行行落到代码上。

**课程**：[强化学习的数学原理](https://www.bilibili.com/read/cv18745978/) ｜
**教材**：[Mathematical Foundations of Reinforcement Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)（赵世钰）

## 算法一览

| 类别 | 算法 | 文件 | 环境 |
| :--- | :--- | :--- | :--- |
| 动态规划 | 策略迭代 | [`policy_iteration.py`](algorithms/policy_iteration.py) | GridWorld |
| | 值迭代 | [`value_iteration.py`](algorithms/value_iteration.py) | GridWorld |
| 蒙特卡洛 | Exploring Starts | [`monte_carlo_exploring_starts.py`](algorithms/monte_carlo_exploring_starts.py) | GridWorld |
| | ε-greedy | [`monte_carlo_epsilon_greedy.py`](algorithms/monte_carlo_epsilon_greedy.py) | GridWorld |
| 时序差分 | SARSA | [`SARSA.py`](algorithms/SARSA.py) | GridWorld |
| | n-step SARSA | [`n_step_sarsa.py`](algorithms/n_step_sarsa.py) | GridWorld |
| | Expected SARSA | [`expected_sarsa.py`](algorithms/expected_sarsa.py) | GridWorld |
| | Q-learning | [`qlearning_on_policy.py`](algorithms/qlearning_on_policy.py) | GridWorld |
| 值函数近似 | V(s) 线性近似 + 半梯度 TD | [`value_function_approximation.py`](algorithms/value_function_approximation.py) | GridWorld |
| | SARSA + 线性特征 | [`SARSA_with_function_approximation.py`](algorithms/SARSA_with_function_approximation.py) | CartPole |
| | Q-learning + 线性特征 | [`qlearning_with_function_approximation.py`](algorithms/qlearning_with_function_approximation.py) | CartPole |
| 深度 RL | DQN（replay + target net） | [`DQN.py`](algorithms/DQN.py) | CartPole |
| | REINFORCE | [`REINFORCE.py`](algorithms/REINFORCE.py) | CartPole |
| | QAC（最简单的 actor-critic） | [`QAC.py`](algorithms/QAC.py) | CartPole ×8 |
| | A2C（δ 作乘子） | [`A2C.py`](algorithms/A2C.py) | CartPole ×8 |

QAC / A2C 是并行版：8 个环境同步推进，每个向量步产出一个 batch 做一次更新。

## 目录结构

```text
rl_gym/
├── algorithms/            # 一个算法一个文件，__main__ 里可直接运行
├── envs/
│   └── grid_world_env/    # 自定义 GridWorld（带禁区），可 pip install -e
└── README.md
```

## 安装与运行

```bash
conda create -n rl_env python=3.10 -y
conda activate rl_env

pip install -e envs/grid_world_env   # GridWorld 环境（自动带上 gymnasium / pygame）
pip install torch tqdm matplotlib    # 深度 RL 部分
```

| 想跑什么 | 命令 |
| :--- | :--- |
| 动态规划 | `python algorithms/policy_iteration.py` ｜ `python algorithms/value_iteration.py` |
| 蒙特卡洛 | `python algorithms/monte_carlo_exploring_starts.py` ｜ `python algorithms/monte_carlo_epsilon_greedy.py` |
| 时序差分 | `python algorithms/SARSA.py` ｜ `python algorithms/n_step_sarsa.py` ｜ `python algorithms/expected_sarsa.py` ｜ `python algorithms/qlearning_on_policy.py` |
| 值函数近似 | `python algorithms/value_function_approximation.py` ｜ `python algorithms/SARSA_with_function_approximation.py` |
| 深度 RL | `python algorithms/DQN.py` ｜ `python algorithms/REINFORCE.py` ｜ `python algorithms/QAC.py` ｜ `python algorithms/A2C.py` |

开发环境：Python 3.10.22 ｜ gymnasium 1.3.0 ｜ numpy 2.2.6 ｜ torch 2.14.1。

## 环境：GridWorld

`size × size` 网格（默认 5×5），在 Gymnasium 官方
[环境创建示例](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
的 `GridWorldEnv` 上加了**禁区（forbidden cell）**。

- **观测** `Dict(agent, target)`，每个分量是 `Box(0, size-1, (2,))`
- **动作** `Discrete(5)` → `right` / `up` / `left` / `down` / `stay`
- **状态编号** `s = x * size + y`（`x` 向右、`y` 向上）
- **注册 id** `grid_world_env/GridWorld-v0`，`max_episode_steps=100`
- **目标格不是吸收态**：到达后继续 `stay` 会一直拿 `+1`，因此 `v(target) = 1/(1-γ)`，
  需要 `TimeLimit` 或自行判停

<details>
<summary><b>奖励表 / MDP 约定 / 渲染叠加层</b></summary>

### 奖励与位置变化

| 情况 | 奖励 | 位置 |
| :--- | :--- | :--- |
| 走到 `target` | `r_target` | 进入目标格 |
| 走出边界 / 撞上 `forbidden` | `r_boundary` / `r_forbidden` | 留在原地（回退该动作） |
| 其他 | `r_other` | 正常移动 |

`start_cell` / `target_cell` / `forbidden_cells` 不传则随机生成，
禁区数量为 `round(size² · U(0.05, 0.3))`（5×5 时实测 1~7 个）。

### `build_mdp()` 的约定

```python
P, R, cells, states_idx = env.build_mdp()   # P, R 形状均为 (S, A, S)
```

`R` 是**逐转移（per-transition）**奖励，所以 Bellman 期望要把 `R` 放进求和的内部：

$$q_\pi(s,a)=\sum_{s'} p(s'\mid s,a)\bigl[R(s,a,s')+\gamma v_\pi(s')\bigr]$$

写成代码即 `qsa += p * R[s][a][s'] + gamma * p * v[s']`。**不要**和 `R(s,a)` 的标量约定混用：
标量约定下 `R` 只能在求和之外加一次，写成 `Σ p·R` 会在 `Σp ≠ 1` 时静默算错。

### 在渲染里看策略和价值

```python
env = GridWorldEnv(render_mode="human", size=10)
env.reset()                              # 先 reset，目标格 / 禁区才定下来
P, R, cells, states_idx = env.build_mdp()

pi, v = PolicyIteration(P, R, theta=1e-3, gamma=0.9).policy_iteration()
env.set_policy(pi)                       # 每帧叠加策略箭头
env.set_v(v)                             # 每帧叠加 V(s) 数值
```

| 元素 | 含义 |
| :--- | :--- |
| 箭头 / 圆点 | 该状态的动作（`stay` 画成中心圆点） |
| 箭头长度 / 圆点半径 | 该动作的概率 `π(a\|s)` |
| 青 / 橙 | `target` / `forbidden`（禁区不画策略） |

`set_policy()` 接受 `(S, A)` 概率、`(S,)` 或 `(size, size)` 动作编号；
`show_ties=True`（默认）会画出所有概率大于 `tie_tol` 的动作。
`set_v()` 接受长度 `S` 或 `(size, size)` 的数组，可用 `fmt` 改数值格式。
两个叠加层互不影响，传 `None` 即可清除。策略是针对某张地图算出来的，
重新 `reset()` 换了地图后需要重算。

</details>

## 参考

- [强化学习的数学原理（课程）](https://www.bilibili.com/read/cv18745978/) ｜ [MOOC 版](https://www.icourse163.org/course/XHUN-1470436188) ｜ [赵世钰实验室](https://shiyuzhao.westlake.edu.cn/Shiyu_Zhao.htm)
- [MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)
- [Gymnasium 文档](https://gymnasium.farama.org) ｜ [环境创建教程](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [动手学强化学习](https://hrl.boyuai.com/)
- 环境包源码来自 Gymnasium 官方示例（MIT License, © 2023 Farama Foundation）

## TODO

- [x] 动态规划：策略迭代、值迭代
- [x] 蒙特卡洛：Exploring Starts、ε-greedy
- [x] 时序差分：SARSA、n-step SARSA、Expected SARSA、Q-learning
- [x] 值函数近似：线性 V(s)、SARSA / Q-learning + 线性特征
- [x] 深度 RL：DQN、REINFORCE、QAC、A2C
- [ ] 策略梯度进阶：PPO
- [ ] 连续动作：DDPG、SAC
