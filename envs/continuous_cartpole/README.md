# continuous_cartpole_env

Gymnasium `CartPole-v1` 的连续动作版本：**只改动作空间**，观测空间、动力学、
奖励、终止条件和渲染都与官方实现逐位一致。

| | `CartPole-v1` | 本环境 |
| :--- | :--- | :--- |
| 动作空间 | `Discrete(2)` | `Box(-1, 1, (1,))` |
| 作用力 | 固定 ±10 N | `force = a · 10 N`（连续可调） |

| `a` | 作用力 | 等价离散动作 |
| :--- | :--- | :--- |
| `+1` | `+10 N` | `1` |
| `0` | `0 N` | 无（离散版做不到） |
| `-1` | `-10 N` | `0` |

`a = ±1` 与 `CartPole-v1` 的轨迹逐位相同，所以可以直接做"同一物理、只换动作空间"
的对照实验；`a` 取中间值则是离散版无法表达的连续控制问题。

## 安装

```bash
pip install -e envs/continuous_cartpole
```

## 使用

```python
import gymnasium as gym
import continuous_cartpole_env  # 触发注册

env = gym.make("continuous_cartpole_env/ContinuousCartpole-v0", render_mode="human")
obs, _ = env.reset(seed=0)
obs, reward, terminated, truncated, _ = env.step([0.3])  # a ∈ [-1, 1]

# 原生批量向量环境（一份 numpy 数组推进所有环境）
venv = gym.make_vec(
    "continuous_cartpole_env/ContinuousCartpole-v0",
    num_envs=8,
    vectorization_mode="vector_entry_point",
)
```

注意：和官方 `CartPole-v1` 一样，**截断由 `gym.make` 加上的 `TimeLimit` 负责**，
`step()` 自己永远返回 `truncated=False`。直接实例化类时请自己包一层：

```python
from gymnasium.wrappers import TimeLimit
from continuous_cartpole_env.envs import ContinuousCartPoleEnv

env = TimeLimit(ContinuousCartPoleEnv(), max_episode_steps=500)
```

## 自检

```bash
python -m continuous_cartpole_env.envs.continuous_cartpole
```

会依次打印动作→力映射、与 `CartPole-v1` 的逐位比对、PD 控制器存活步数、
向量环境的接口形状和 autoreset 语义。
