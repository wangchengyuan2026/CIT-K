"""
baseline.fep_model —— 标准自由能原理（FEP）主动推理基线
========================================================
论文对基线的全部描述只有两句：

    「对比对象为本文 HIM 模型与标准自由能原理主动推理（FEP）模型」
    「FEP 以最小化惊奇度（Surprise）为核心优化目标……智能体倾向停留在
      低不确定性区域，'保持不动'成为最优策略，极易陷入局部极小值」

论文**没有给出 FEP 基线实现细节**（生成模型、精度、先验、探索噪声一概未提），
因此本文件是该基线的一次**重建**，重建口径如下：

期望自由能（对候选格子 j）
--------------------------
    G(j) = w_u · 0.5·ln(2π·(σ_j² + σ_m²))        ← 不确定性/复杂度项
         + (μ_j − μ_prior)² / (2·σ_m²)           ← 与先验的不符度/accuracy 项

其中
    μ_j, σ_j²  —— 智能体对格子 j 的感官值高斯信念（访问后做正态-正态共轭更新）
    μ_prior    —— 世界先验均值（取字段全局均值，"世界大体是平静的"）
    σ_m²       —— 生成模型自身的观测噪声方差（固定）
    w_u        —— 不确定性项权重

两项合起来的行为正好对应论文的描述：**未访问过的格子因信念不确定而受罚，
偏离先验的格子（兴趣点）也受罚** → 智能体被压回"已熟悉 + 平淡"的区域，停止探索。

另外两个实现细节（论文未规定，此处显式登记）
--------------------------------------------
1. `eps` 探索噪声：ε-greedy 随机动作。这是绝大多数主动推理实现的必备成分。
   ε=0 时该智能体会在头几步之内冻结（见 experiments/fep_sensitivity.py 的扫描）。
2. `stay_penalty`：对"原地停留"加一个小罚项，否则在 ε=0 时它会永久停在起点。

⚠ 该基线是重建而非复现：论文已发表的 FEP 覆盖率 31.8% 无法从论文文本推出，
   相关差异已在 docs/results_report.md 的第 4 节登记。
"""

from __future__ import annotations

import math
import random

from him_core.grid_env import ACTION_STAY, N_ACTIONS

LOG_2PI = math.log(2.0 * math.pi)


class FEPAgent:
    """主动推理基线：最小化期望自由能（= 最小化惊奇度）。"""

    name = "fep"

    def __init__(self, env, rng: random.Random, var_prior: float = 1.0,
                 model_var: float = 0.35 ** 2, unc_weight: float = 1.0,
                 obs_var: float = 0.05 ** 2, eps: float = 0.15,
                 stay_penalty: float = 0.05):
        self.env = env
        self.rng = rng
        self.n = env.n_cells

        self.var_prior = var_prior
        self.var_m = model_var
        self.w_u = unc_weight
        self.obs_var = obs_var
        self.eps = eps
        self.stay_penalty = stay_penalty

        self.mu_prior = env.prior_mean
        self.mu = [self.mu_prior] * self.n
        self.var = [self.var_prior] * self.n

        self.pos = -1
        self.steps = 0
        self.moves = 0
        self._nbr = env.nbr
        self._field = env.field

    # ------------------------------------------------------------------
    def reset(self, start: int):
        self.mu = [self.mu_prior] * self.n
        self.var = [self.var_prior] * self.n
        self.pos = start
        self.steps = 0
        self.moves = 0
        self._update(start, self.env.sense(start))

    # ------------------------------------------------------------------
    def _update(self, idx: int, obs: float):
        """正态-正态共轭更新。"""
        prec = 1.0 / self.var[idx] + 1.0 / self.obs_var
        new_var = 1.0 / prec
        self.mu[idx] = new_var * (self.mu[idx] / self.var[idx] + obs / self.obs_var)
        self.var[idx] = new_var

    def _free_energy(self, j: int) -> float:
        s = self.var[j] + self.var_m
        uncertain = 0.5 * (LOG_2PI + math.log(s))
        inaccuracy = (self.mu[j] - self.mu_prior) ** 2 / (2.0 * self.var_m)
        return self.w_u * uncertain + inaccuracy

    # ------------------------------------------------------------------
    def select_action(self) -> int:
        rng = self.rng
        nbr = self._nbr[self.pos]

        if self.eps > 0.0 and rng.random() < self.eps:
            valid = [a for a in range(N_ACTIONS) if nbr[a] >= 0]
            return valid[rng.randrange(len(valid))]

        best_g = None
        best_actions = []
        for a in range(N_ACTIONS):
            j = nbr[a]
            if j < 0:
                continue
            g = self._free_energy(j)
            if a == ACTION_STAY:
                g += self.stay_penalty
            if best_g is None or g < best_g - 1e-12:
                best_g = g
                best_actions = [a]
            elif abs(g - best_g) <= 1e-12:
                best_actions.append(a)
        if not best_actions:
            return ACTION_STAY
        return best_actions[rng.randrange(len(best_actions))]

    # ------------------------------------------------------------------
    def step(self) -> int:
        a = self.select_action()
        j = self._nbr[self.pos][a]
        if j >= 0 and j != self.pos:
            self.pos = j
            self.moves += 1
        self._update(self.pos, self.env.sense(self.pos))
        self.steps += 1
        return self.pos

    # ------------------------------------------------------------------
    def surprise(self, idx: int, obs: float) -> float:
        """−ln p(o | 模型)，用于诊断。"""
        s = self.var[idx] + self.obs_var
        return 0.5 * (LOG_2PI + math.log(s)) + (obs - self.mu[idx]) ** 2 / (2.0 * s)


__all__ = ["FEPAgent"]
