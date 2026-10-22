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
│   ├── monte_carlo_epsilon_greedy.py
│   ├── monte_carlo_exploring_starts.py
│   ├── policy_iteration.py      # 策略迭代（策略评估 + 策略改进）
│   └── value_iteration.py       # 值迭代
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
| 走到 `target` | `r_target` | 进入目标格 |
| 走出边界 / 撞上 `forbidden` | `r_boundary/r_fobidden` | 留在原地（回退该动作） |
| 其他 | `r_other` | 正常移动 |

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

### 在渲染里直接看策略：`set_policy()`

把策略交给环境后，**每一帧渲染都会把策略画在地图上**（`render_mode="human"` 和 `"rgb_array"` 走同一套逻辑）：

```python
env = GridWorldEnv(render_mode="human", size=10)
env.reset()                                     # 先 reset，目标格/禁区才定下来
P, R, cells, states_idx = env.build_mdp()
pi, v = PolicyIteration(P, R, theta=1e-3, gamma=0.9).policy_iteration()

env.set_policy(pi)                              # 之后每帧都叠加策略
env.set_v(v)                                    # 顺便把 V(s) 也显示出来

while not episode_over:                         # 实跑时能看着策略走
    obs, r, terminated, truncated, info = env.step(int(np.argmax(pi[s])))
```

| 画面元素 | 含义 |
|---|---|
| 箭头 | 该状态的策略动作（`right` / `up` / `left` / `down`），**从格子中心向外指**（多个动作就是同一个中心射出的几根箭头） |
| 圆点 | 该状态的策略动作是 `stay`，画在格子中心 |
| **箭头长度 / 圆点半径** | **该动作的概率 `π(a\|s)`**（`p=1` 满长，`p=0` 保留 `policy_min_scale` 的保底长度，概率小的动作不会消失） |
| 青色 /橙色 | `target` / `forbidden`（禁区格不画策略，因为实际交互中永远走不到） |

- `set_policy(policy)` 的 `policy` 接受 `(n_states, n_actions)` 概率（`pi` 直接可传）、`(n_states,)` / `(size, size)` 动作编号、`(size, size, n_actions)` 概率；`set_policy(None)` 清除叠加。
- **`show_ties` 默认 `True`**：概率大于 `tie_tol`（默认 `1e-8`）的动作全部画出，所以 ε-greedy（如 `[0.84, 0.04, 0.04, 0.04, 0.04]` 就是"一根长箭头 + 四根小短箭头 + 一个小圆点"）和并列最优都能一眼看出权重。想看每格只留一个 `argmax` 就传 `set_policy(pi, show_ties=False)`——此时箭头长度仍表示该动作的概率。
- 不调用 `set_policy()` 时，渲染与原来完全一致（只有 agent / target / forbidden），不会有任何额外开销。
- 外观可调：`env.policy_arrow_scale`（`p=1` 时的箭头长度，单位格）、`env.policy_stay_radius`（`p=1` 时的圆点半径）、`env.policy_min_scale`（`p=0` 时的保底长度比例，默认 `0.3`）、`env.policy_color`、`env.policy_current_color`；播放速度由 `metadata["render_fps"]` 控制。
- 策略是**针对某张地图**（目标格 + 禁区）算出来的：如果重新 `reset()` 换了目标格，需要重新算 `pi` 再 `set_policy()`。

### 在渲染里直接看价值：`set_v()`

和 `set_policy()` 平行的另一层：`set_v(v)` 之后，**每帧都会把 `V(s)` 写在每格左上角**（两个叠加层互不影响，可以只开一个，也可以同时开）：

```python
env.set_v(v)                # 一维长度 n_states（ValueIteration.v / PolicyIteration 的 v 直接可传）
env.set_v(v, fmt="{:.1f}")  # 顺便改数值格式；也可以传 (size, size) 二维数组
env.set_v(None)             # 清除价值叠加
```

| 细节 | 说明 |
|---|---|
| 数值来源 | 长度 `n_states` 的一维数组/列表，或 `(size, size)`；按 `s = x*size + y` 解释，与 `build_mdp()` 的 `cells` 顺序一致 |
| 位置 | 每格左上角，灰字（`v_color`）；禁区格跳过（和策略层一致） |
| agent 所在格 | 数值会被蓝圆点挡住，所以这一格最后单独重画一次并垫一块 `v_bg_color` 底色，保证读得清 |
| 可调 | `env.v_color`、`env.v_bg_color`、`env.v_format`（如 `"{:+.2f}"`）、`env.v_font_scale`（字号 = 格子边长 × 该比例，默认 `0.2`） |
| 报错 | 元素个数不等于 `size²` 时抛 `ValueError`（提示应为 `n_states` 或 `(size, size)`） |
| 未调用时 | 完全等价于原渲染，没有任何额外开销（`env.v is None`） |


---

## 算法

### Policy Iteration（`algorithms/policy_iteration.py`）✅

| 方法 | 说明 |
|---|---|
| `_policy_evaluation()` | 迭代策略评估。同步（Jacobi）更新：`next_v` 与 `self.v` 分开，一轮扫完再整体写回；用 sup-norm 判停 `max_s(v_new(s) − v_old(s))< theta`|
| `_policy_improvement()` | 策略改进。对每个状态取 `argmax_a q(s, a)`；并列最优的动作均分概率 |
| `policy_iteration()` | 交替执行 `_policy_evaluation() → _policy_improvement()`，直到 `old_pi == new_pi`（策略不再变化，即收敛到最优策略） |

### Value Iteration（`algorithms/value_iteration.py`）✅

| 方法 | 说明 |
|---|---|
| `value_iteration()` | 对每个状态取 `argmax_a q(s, a)`，取使`q(s, a)`最大的动作为新策略, 取最大`q(s, a)`为`v_new(s)`，直到 `max_s(v_new(s) − v_old(s))< theta` |

### Monte Carlo Exploring Starts ✅

| 方法 | 说明 |
|---|---|
| `_generate_epsilon()` | 随机取`s0,a0` 产生一个新的 `epsilon` |
| `_update(epsilon)` | 使用获得到的`epsilon`来更新`action value`和 `pi`|

### Monte Carlo EpsilonGreedy ✅

| 方法 | 说明 |
|---|---|
| `_generate_epsilon()` | 随机取`s0,a0` 产生一个新的 `epsilon` |
| `_update(epsilon)` | 使用获得到的`epsilon`来更新`action value`和 `pi`|

### Deep Q-Network（`algorithms/deep_qlearning.py`）✅

环境：`CartPole-v1`。按“主网络 + 目标网络 + 经验回放”实现 DQN。

| 组件/方法 | 说明 |
|---|---|
| `ReplayBuffer` | 固定容量 `deque` 保存 `(s, a, r, s', done)`；`sample()` 均匀抽取 mini-batch |
| `Qnet` | 两层 MLP：`state_dim → hidden_dim → action_dim`，隐藏层 ReLU |
| `take_action()` | ε-greedy；`greedy=True` 时直接取主网络的 `argmax`（评估/演示） |
| `update()` | 计算 `y_T = r + γ·max_a q̂(s', a; w_T)`（终止样本只保留 `r`），MSE 反传更新主网络；每 `C` 次更新执行 `w_T = w` |
| `train()` | 每步写入 buffer，样本数 ≥ `minimal_size` 后每步更新；ε 从 1.0 线性衰减；定期贪心评估并保留最佳参数 |
| `evaluate()` | 贪心策略跑 20 局，报告平均回报 |

实现细节：

- 构造时先执行 `target_qnet.load_state_dict(qnet.state_dict())`，保证两个网络初始参数相同。
- 目标值在 `torch.no_grad()` 下计算，目标网络不接收梯度；梯度按范数裁剪到 10。
- `terminated` 和 `truncated` 都作为 `done`，时间上限样本不再从 `s'` 自举。
- DQN 训练后期偶尔会突然变差：每 50 局用贪心策略评估 10 局，保留评估最好的参数。
- 默认超参数：`hidden_dim=128`、`lr=1e-3`、`γ=0.98`、`C=100`、`batch_size=64`、`buffer_size=10000`、`minimal_size=1000`、800 局。

```bash
python algorithms/deep_qlearning.py                        # 训练 + 贪心评估
python algorithms/deep_qlearning.py --seed 2 --episodes 800
python algorithms/deep_qlearning.py --render               # 训练结束后渲染一局
```

---

## 安装

```bash
conda create -n rl_env python=3.10 -y
conda activate rl_env

# 以可编辑模式安装自定义环境（自动带上 gymnasium / pygame）
pip install -e envs/grid_world_env
```

参考版本（本仓库开发环境）：Python 3.10.22、gymnasium 1.3.0、pygame 2.6.1、numpy 2.2.6、pandas 2.3.3。
DQN 示例额外依赖 PyTorch 与 tqdm（参考版本 torch 2.7.1、tqdm 4.70.0）。

## 运行

```bash
python algorithms/policy_iteration.py   # 策略迭代 
python algorithms/value_iteration.py    # 价值迭代
python algorithms/monte_carlo_exploring_starts.py  
python algorithms/monte_carlo_epsilon_greedy.py # MC
python algorithms/deep_qlearning.py     # DQN（CartPole）
```
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
- [x] 值迭代（Value Iteration）
- [x] 蒙特卡洛方法（MC Exploring Starts / MC Epsilon Greedy）
- [ ] 时序差分：Sarsa、Q-learning、Expected Sarsa
- [ ] 值函数近似 / 策略梯度（Actor-Critic）
