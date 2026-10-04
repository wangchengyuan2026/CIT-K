"""
baseline.rl_baseline —— 强化学习基线（外部奖励）
================================================
论文在项目结构里列了 `baseline/rl_baseline.py`（「对标 FEP / RL 基线代码」），
但**正文与结果中从未报告过 RL 的任何数据**。本文件把这条基线补齐。

按论文对 RL 的定性——「现有强化学习与主动推理框架普遍依赖外部奖励或人工
预定义目标，智能体行为受外部信号约束」——这里实现的是**外部奖励 + Q 学习**：

    状态   s = 当前格子 index（10000 个）
    动作   a ∈ {上, 下, 左, 右, 停留}
    奖励   r = +1   首次进入「兴趣区」（φ ≥ 0.5）的格子
            r = 0   其余情况
    更新   Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
    策略   ε-greedy，ε 线性退火

⚠ 注意：该基线**不使用任何内在动机信号**，是"HIM 要替代的那类方法"的直系代表。
   如需"HIM 的 ΔL 信号 + RL 的信用分配机器"这一对照，请使用
   him_core.motivation_engine.HIMAgent 去掉希望机制的消融版本（--agents him_nohope）。
"""

from __future__ import annotations

import random

from him_core.grid_env import ACTION_STAY, N_ACTIONS


class RandomAgent:
    """均匀随机游走。仅用作覆盖率指标的量纲校准参考，不是论文中的基线。"""

    name = "random"

    def __init__(self, env, rng: random.Random):
        self.env = env
        self.rng = rng
        self.pos = -1
        self.steps = 0
        self.moves = 0
        self._nbr = env.nbr

    def reset(self, start: int):
        self.pos = start
        self.steps = 0
        self.moves = 0

    def step(self) -> int:
        rng = self.rng
        nbr = self._nbr[self.pos]
        a = rng.randrange(N_ACTIONS)
        j = nbr[a]
        if j >= 0 and j != self.pos:
            self.pos = j
            self.moves += 1
        self.steps += 1
        return self.pos


class RLAgent:
    """表格型 Q 学习 + 外部稀疏奖励。"""

    name = "rl"

    def __init__(self, env, rng: random.Random, alpha: float = 0.2,
                 gamma: float = 0.95, eps_start: float = 1.0,
                 eps_end: float = 0.05, eps_decay_steps: int = 6000):
        self.env = env
        self.rng = rng
        self.n = env.n_cells
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        self.Q = [[0.0] * N_ACTIONS for _ in range(self.n)]
        self.visited_rewarded = [False] * self.n
        self.pos = -1
        self.steps = 0
        self.moves = 0
        self.total_reward = 0.0
        self._nbr = env.nbr
        self._poi = env.poi_flags

    # ------------------------------------------------------------------
    def reset(self, start: int):
        self.Q = [[0.0] * N_ACTIONS for _ in range(self.n)]
        self.visited_rewarded = [False] * self.n
        self.pos = start
        self.steps = 0
        self.moves = 0
        self.total_reward = 0.0
        self.visited_rewarded[start] = True

    def epsilon(self) -> float:
        if self.steps >= self.eps_decay_steps:
            return self.eps_end
        frac = self.steps / self.eps_decay_steps
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    # ------------------------------------------------------------------
    def select_action(self) -> int:
        rng = self.rng
        nbr = self._nbr[self.pos]
        if rng.random() < self.epsilon():
            valid = [a for a in range(N_ACTIONS) if nbr[a] >= 0]
            return valid[rng.randrange(len(valid))]

        q = self.Q[self.pos]
        best = None
        best_actions = []
        for a in range(N_ACTIONS):
            if nbr[a] < 0 and a != ACTION_STAY:
                continue
            v = q[a]
            if best is None or v > best + 1e-12:
                best = v
                best_actions = [a]
            elif abs(v - best) <= 1e-12:
                best_actions.append(a)
        if not best_actions:
            return ACTION_STAY
        return best_actions[rng.randrange(len(best_actions))]

    # ------------------------------------------------------------------
    def step(self) -> int:
        s = self.pos
        a = self.select_action()
        j = self._nbr[s][a]
        if j < 0:
            j = s
        if j != s:
            self.moves += 1

        r = 0.0
        if not self.visited_rewarded[j] and self._poi[j]:
            r = 1.0
            self.visited_rewarded[j] = True
        self.total_reward += r

        nxt = self.Q[j]
        best_next = max(nxt)
        self.Q[s][a] += self.alpha * (r + self.gamma * best_next - self.Q[s][a])

        self.pos = j
        self.steps += 1
        return self.pos


__all__ = ["RLAgent", "RandomAgent"]
