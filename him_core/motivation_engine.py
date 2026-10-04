"""
him_core.motivation_engine —— 内在动机引擎（HIM 完整模型）
==========================================================
对应论文 §2.4 伪代码与【3版】§3「完整逻辑链」：

    Select action to maximize cumulative ΔL_t
    action = argmax([drive_of(next_state) for next_state in neighbors])
    drive  = ΔL_t + hope_drive(L_t, hope_buffer)

四个变体
--------
    him_v1      （论文原式）  好奇心权重 = 0，前瞻深度 = 1
    him         （主模型）    ΔL + 希望 + 区域新奇度，前瞻深度 = lookahead
    him_nohope  （消融）      去掉希望机制，其余同 him
    him_blocknov（消融）      区域新奇度退回「最近访问时刻 + 双线性插值」直译

为什么论文原式跑不出探索行为
----------------------------
「最大化累积 ΔL_t」按字面就是**多步**目标，但论文的动作选择写成
`argmax([drive_of(next_state) for next_state in neighbors])`，实际是单步贪心。
单步贪心在数学上就是舒适度场的**梯度上升**，必然收敛到局部极大值：

    实测（him_v1）：覆盖率 0.23%，收敛率 100%，位移仅 35 步 / 20000 步
    —— 智能体爬上第一个兴趣点后就不再移动，与论文声称的
       「88.2% 覆盖率、0.2% 陷入率、持续探索」直接矛盾。

根因有三条，互相独立：
    1. 「最大化 ΔL 累积值」= 最大化最终 L。只要环境与需求都静止，最优策略
       就是停在最高点，这与论文「不追求静态绝对舒适度极大值」自相矛盾。
    2. 希望算子 max(0, peak − L − threshold) 在停在峰值时恒为 0，且它是
       L 的**减函数**——越不舒服驱动越大。它只能把智能体拉回记忆中的高点，
       不可能是"离开舒适区"的力。
    3. 论文没有给出任何"区域新奇度 / 好奇心"的**可计算形式**。
       【C版】§6.3 把它当作核心公理，【3版】§5.2 主张"ΔL ≈ 0 会产生内在
       压力去主动尝试新行为"，但公式里 drive = ΔL + hope_drive 在 ΔL = 0
       处恒为 0，argmax 永远选"原地不动"。

所以本实现补入两个明确标注的内生算子，并保留所有对照组：

    A. 区域新奇度 novelty（本实现补全的缺环）
    B. 多步前瞻（lookahead 步贪心 rollout），把「累积 ΔL」落成真正的多步目标

A 的两种实现（novelty_mode）
----------------------------
  (a) `block` —— 「最近访问时刻」的直译
      novelty(j) = clip((now − last_visit(block(j))) / horizon, 0, 1)
      区域按 block_size 分块，再用双线性插值铺成空间连续势场。

      ⚠ 实测这个形式**不可能产生探索行为**（覆盖率 0.19% ~ 0.21%），原因是
      数学性的：双线性插值把一个分段常值场铺平，在"当前区块（刚被刷新、
      novelty = 0）"与"相邻已陈旧区块（novelty = 1）"的交界处，插值曲面
      出现**局部极大值**，而它恰好位于智能体脚下。动作集里 `stay` 的
      新奇度最高，于是智能体原地不动 / 在 3–4 个区块交界处来回振荡，
      永远不出区块。探针实测：20000 步只触碰 4/169 个区块、17 个格子。

  (b) `field` —— 距离势场新奇度（**默认**）

        novelty(j) = cw · [ NB·D·(1 − V[j])  +  (D − d(j)) ]

        V[j]   = 格子 j 是否已被访问（0/1）              ——「目标信号」
        d(j)   = j 到最近**未访问**格子的曼哈顿距离，
                 已按 D 截断（d ≤ D）                    ——「导航信号」
        NB     = near_bonus（默认 1.0，单位是 D）
        D      = far_cap（默认 96 格）
        cw     = curiosity_weight（默认 0.2）

      两个因子分工清晰、缺一不可：

      · NB·D·(1 − V[j])：只要 j 没走过就额外给满额奖励。
          – 未访问格（d ≡ 0）恒优于任何已访问格（最多 D），**严格占优、
            不会被 d 场的滞后刷新破坏**（这是必须把"未访问"显式写成独立
            奖励项、而不能只靠 d 的原因：d 每 10 步才重算一次，刚踩过的
            格子在 d 场里仍是 0，若只用 d 判定就会出现 stay 与未访问邻格
            并列、被 ΔL 拉回原地的假象）。
          – 智能体脚下刚被标记为已访问 ⇒ novelty(stay) 恒低于任一未访问
            邻格 ⇒ **`stay` 被否决，智能体永远无法停滞** ——
            这就是【3版】§5.2 想要、但从未写进公式的那个"停滞压力"。

      · (D − d[j])：**长程导航**。d 是逐格的精确曼哈顿距离，把"我在已探索
        区域的什么位置"编码成一个平滑势场：
          – **每走一步 d 减 1 ⇒ novelty 增加 cw**。这个梯度是**常数**，
            与 D 无关、与"离前沿多远"无关，因此只要 cw 取到 0.2，就比
            ΔL 的量级（|ΔL| ≤ (1−β)·|Δφ| ≈ 0.03）大一个数量级 ⇒
            方向性明确。
            （早期写成 (1 − d/D) 时梯度只有 cw/D ≈ 0.004，在兴趣点陡坡上
             被 ΔL ≈ 0.016 反超，智能体会卡死在兴趣点顶上，实测覆盖率
             只有 0.79%。）
          – 势场只在未访问格取极大，**不可能在已探索区域内部造出假顶点**
            （真距离函数没有局部极大值，这正是弃用双线性插值的理由）。
          – 智能体一旦被自己围住（四邻皆已访问），(1−V) 全为 0，唯一
            剩下的信号就是 d 的梯度 ⇒ **它总能自己走回前沿**。
            （只保留 (1−V) 项、不加长程项时，实测智能体有 78.8% 的步数
             卡在已探索区域内部空转，覆盖率只有 42%。）

      · 全图走完 ⇒ d ≡ D 且 V ≡ 1 ⇒ novelty ≡ 0 ⇒ 所有动作并列 ⇒ 由 ΔL
        决定 ⇒ 收敛到高舒适度区。**收敛性内生于该算子**，不需要额外开关。

      维护代价：d 每 `far_interval` 步用一次 4 邻域就地松弛全量重算
      （100×100、cap 轮 numpy 向量化，约 0.8 ms），摊到每步 ~0.08 ms。

ΔL 的符号约定
-------------
    drive = ΔL + hope + novelty。ΔL 决定"往舒适度高的方向微调"，
    novelty 决定"往没走过的地方去"，hope 是记忆高点回来的拉力。
    量级：novelty 的**一步梯度** = cw = 0.2，ΔL ~ 1e−3 ~ 1e−2。
    因此只要还有新格子可走，novelty 主导；全图走完后由 ΔL 决定收敛点 ——
    这个优先级正是论文叙事（88.2% 覆盖率、不追求静态舒适度极大值）所要求的。
"""

from __future__ import annotations

import math
import random

import numpy as np

from .grid_env import ACTION_STAY, N_ACTIONS
from .homeostasis import (
    BOREDOM_REF_DEFAULT, HOPE_MIN_LEN, Homeostasis, MEMORY_LEN, TREND_GAIN,
)

CURIOSITY_WEIGHT_DEFAULT = 0.20
NOVELTY_HORIZON_DEFAULT = 1500.0
BLOCK_SIZE_DEFAULT = 8
LOOKAHEAD_DEFAULT = 1          # 论文原式为 1（单步贪心）
LOOKAHEAD_DISCOUNT = 0.95      # 前瞻折扣，论文未给；1.0 即等权累加
NOVELTY_MODE_DEFAULT = "field"
NEAR_BONUS_DEFAULT = 1.0
FAR_CAP_DEFAULT = 96.0         # D：导航势场的截断距离（格）= 松弛轮数上限
FAR_INTERVAL_DEFAULT = 10      # d 场的重算间隔（步）
X_MODE_DEFAULT = "trend"       # trend = B 版趋势外推；spatial = A 版 D 体语义
X_LAMBDA_DEFAULT = 160.0       # R：空间记忆锥的衰减半径（格）≈ 全图直径
X_SOURCE_MIN_DEFAULT = 0.3     # θ：感官值 ≥ θ 的已访问格子才算"已知舒适源"
X_NEUTRAL_DEFAULT = 0.5        # 无任何已知舒适源时的中性记忆判断值
_INF = float("inf")


class HIMAgent:
    """内稳态内在动机智能体。"""

    name = "him"

    def __init__(self, env, rng: random.Random, hope_gain: float = 0.1,
                 use_hope: bool = True, beta_max: float = 0.8,
                 beta_tau: float = 300.0, demand: float = 1.0,
                 curiosity_weight: float = CURIOSITY_WEIGHT_DEFAULT,
                 novelty_horizon: float = NOVELTY_HORIZON_DEFAULT,
                 block_size: int = BLOCK_SIZE_DEFAULT,
                 lookahead: int = LOOKAHEAD_DEFAULT,
                 boredom_weight: float = 0.0,
                 boredom_ref: float = BOREDOM_REF_DEFAULT,
                 variant: str = "him",
                 novelty_mode: str = NOVELTY_MODE_DEFAULT,
                 near_bonus: float = NEAR_BONUS_DEFAULT,
                 far_cap: float = FAR_CAP_DEFAULT,
                 far_interval: int = FAR_INTERVAL_DEFAULT,
                 x_mode: str = X_MODE_DEFAULT,
                 x_lambda: float = X_LAMBDA_DEFAULT,
                 x_source_min: float = X_SOURCE_MIN_DEFAULT,
                 x_scramble: bool = False):
        self.env = env
        self.rng = rng
        self.variant = variant
        self.curiosity_weight = curiosity_weight
        self.novelty_horizon = novelty_horizon
        self.novelty_mode = novelty_mode
        self._inv_h = 1.0 / novelty_horizon
        self.lookahead = max(1, int(lookahead))
        self.n_cells = env.n_cells
        self._field = env.field
        self._nbr = env.nbr

        # ---- X 的两种实现（记忆项：记忆需求的可计算形式）----
        # trend   = B 版原式：感官历史趋势外推（对位置不敏感）
        # spatial = A 版 D 体语义：「我记得那里舒服、我记得怎么过去」的
        #           空间记忆判断（作者 2026-09-24 指正后补入）
        self.x_mode = x_mode
        self.x_lambda = float(x_lambda)        # R：记忆锥衰减半径
        self.x_source_min = float(x_source_min)  # θ：舒适源判定阈值
        self.x_neutral = X_NEUTRAL_DEFAULT
        # x_scramble：脱钩对照（2026-09-24「记忆需求由感官需求塑造」检验臂）。
        # True 时空间记忆不再记录真实感官值，每格首次到访记一个随机数——
        # 记忆的「结构」（哪里走过、哪里是源）保留，「内容」（那里有多舒服）
        # 与感官彻底脱钩。若记忆需求确由感官需求塑造，该臂的回家行为应失效。
        self.x_scramble = bool(x_scramble)
        self._scr: dict[int, float] = {}
        if x_mode == "spatial":
            self.best2 = np.zeros((env.size, env.size), dtype=np.float32)
            self.src_pos_val: dict[int, float] = {}
            self._src_np = None
            self._src_dirty = True

        if novelty_mode == "field":
            self._init_field(env, near_bonus, far_cap, far_interval)
        else:
            self._init_block_field(env, block_size)

        self.homeo = Homeostasis(
            demand=demand, beta_max=beta_max, beta_tau=beta_tau,
            hope_gain=hope_gain, use_hope=use_hope,
            boredom_weight=boredom_weight, boredom_ref=boredom_ref,
        )
        self.pos = -1
        self.steps = 0
        self.moves = 0

    # ================= 空间记忆判断（X_spatial）=================
    def _remember_raw(self, pos: int, raw: float) -> None:
        """把本次感官体验写进空间记忆：每格保留历史最佳感官值；
       感官值 ≥ θ 的格子登记为「已知舒适源」（记忆需求判断的原料）。
       x_scramble=True 时写入的是与感官无关的随机数（脱钩对照）。"""
        if self.x_scramble:
            v = self._scr.get(pos)
            if v is None:
                v = self.rng.random()
                self._scr[pos] = v
        else:
            v = raw if raw > 0.0 else 0.0
        if v > float(self.best2.flat[pos]):
            self.best2.flat[pos] = v
        if v >= self.x_source_min:
            old = self.src_pos_val.get(pos)
            if old is None or v > old:
                self.src_pos_val[pos] = v
                self._src_dirty = True

    def _x_spatial_at(self, j: int) -> float:
        """X_spatial(j) = max(0.5, max_{已知舒适源 k} best[k]·max(0, 1 − d(j,k)/R))

        A 版语义：记忆需求 = 「我知道哪里能满足需求、我知道怎么过去」。
        - best[k]：记忆中 k 处的历史最佳感官值（「那里有多舒服」的记忆）
        - 1 − d/R：曼哈顿距离线性衰减锥（「过去要走的路」的记忆）。
          用线性而不是指数：长程梯度处处非零（= best/R），与区域新奇度
          算子的同一教训——指数衰减在远处梯度消失，智能体回不了家。
        - 取 max 而非求和：回家只回「记忆里最好的那一个地方」；
          max 锥除锥顶外无内部极大值，不会造出假顶点。
        - 下限 0.5：无任何已知舒适源时，记忆判断为中性（X 未知 ≠ 记忆
          需求未满足；B 版趋势外推在记忆 < 2 条时同样返回 0.5）。
        """
        if not self.src_pos_val:
            return self.x_neutral
        if self._src_dirty:
            self._src_np = np.array(list(self.src_pos_val.items()),
                                    dtype=np.float32)
            self._src_dirty = False
        arr = self._src_np                      # (K, 2)：[pos, best]
        size = self.env.size
        y = j // size
        x = j - y * size
        py = arr[:, 0] // size
        px = arr[:, 0] - py * size
        d = np.abs(py - y) + np.abs(px - x)     # 曼哈顿距离（无墙环境）
        v = arr[:, 1] * np.maximum(0.0, 1.0 - d / self.x_lambda)
        m = float(v.max())
        return m if m > self.x_neutral else self.x_neutral

    # ================= 区域新奇度实现 =================
    def _init_field(self, env, near_bonus: float, far_cap: float,
                    far_interval: int) -> None:
        """见模块 docstring：novelty = cw·[NB·D·(1−V) + (D − d)]。"""
        size = env.size
        self.nb = near_bonus
        self.far_cap = float(far_cap)
        self.far_interval = max(1, int(far_interval))
        self._next_far = 0
        self.v2 = np.zeros((size, size), dtype=np.uint8)   # 已访问掩膜 V
        self.vflat = self.v2.ravel()
        self.d2 = np.zeros((size, size), dtype=np.float32)  # 到最近未访问格的距离 d
        self.dflat = self.d2.ravel()
        self._n_unvisited = size * size

    def _init_block_field(self, env, block_size: int) -> None:
        """论文『最近访问时刻』的直译：区块级时间戳 + 双线性插值。"""
        self.block_size = block_size
        self.n_blocks_side = nbs = (env.size + block_size - 1) // block_size
        self.block_time = [-(10 ** 9)] * (nbs * nbs)

        bs = block_size
        size = env.size
        half = (bs - 1) / 2.0
        cell_blocks = []
        for yy in range(size):
            by0 = int((yy - half) // bs)
            fy = ((yy - half) / bs) - by0
            by0 = min(max(by0, 0), nbs - 1)
            by1 = min(by0 + 1, nbs - 1)
            for xx in range(size):
                bx0 = int((xx - half) // bs)
                fx = ((xx - half) / bs) - bx0
                bx0 = min(max(bx0, 0), nbs - 1)
                bx1 = min(bx0 + 1, nbs - 1)
                cell_blocks.append((
                    by0 * nbs + bx0, (1.0 - fy) * (1.0 - fx),
                    by0 * nbs + bx1, (1.0 - fy) * fx,
                    by1 * nbs + bx0, fy * (1.0 - fx),
                    by1 * nbs + bx1, fy * fx,
                ))
        self.cell_blocks = tuple(cell_blocks)

    def _block_of(self, i: int) -> int:
        nbs = self.n_blocks_side
        return (i // self.env.size) // self.block_size * nbs \
            + (i % self.env.size) // self.block_size

    def _recompute_far(self) -> None:
        """d(j) = j 到最近未访问格子的曼哈顿距离（上限 far_cap）。

        4 邻域就地松弛 + 向量化推进：每轮做 4 次 np.minimum，
        由于是就地更新，单个方向的一次扫描即可把距离沿该方向传播到整行/列。
        上限 20 轮足够收敛到 min(真实距离, far_cap)。"""
        cap = int(self.far_cap)
        big = np.float32(self.far_cap)
        d = self.d2
        # 未访问格 d = 0，已访问格 d = big（用 cap 作饱和值）
        np.copyto(d, np.where(self.v2 != 0, big, np.float32(0.0)))
        if self._n_unvisited == 0:
            d.fill(big)
            return
        for _ in range(cap):
            prev = float(d.sum())
            np.minimum(d[1:, :], d[:-1, :] + 1.0, out=d[1:, :])
            np.minimum(d[:-1, :], d[1:, :] + 1.0, out=d[:-1, :])
            np.minimum(d[:, 1:], d[:, :-1] + 1.0, out=d[:, 1:])
            np.minimum(d[:, :-1], d[:, 1:] + 1.0, out=d[:, :-1])
            if float(d.sum()) == prev:
                break

    # ================= 生命周期 =================
    def reset(self, start: int):
        self.pos = start
        self.steps = 0
        self.moves = 0
        if self.novelty_mode == "field":
            self.v2.fill(0)
            self.vflat[start] = 1
            self._n_unvisited = self.n_cells - 1
            self._recompute_far()
            self._next_far = self.far_interval
        else:
            self.block_time = [-(10 ** 9)] * (self.n_blocks_side ** 2)
            self.block_time[self._block_of(start)] = 0
        h = self.homeo
        self.homeo = Homeostasis(
            demand=h.demand, beta_max=h.beta_max, beta_tau=h.beta_tau,
            hope_gain=h.hope_gain, use_hope=h.use_hope,
            boredom_weight=h.boredom_weight, boredom_ref=h.boredom_ref,
        )
        raw = self.env.sense(self.pos)
        if self.x_mode == "spatial":
            self.best2.fill(0.0)
            self.src_pos_val = {}
            self._scr = {}
            self._src_dirty = True
            self._remember_raw(self.pos, raw)
            self.homeo.observe(raw, x_override=self._x_spatial_at(self.pos))
        else:
            self.homeo.observe(raw)

    # ================= 舒适度 / 新奇度 =================
    @staticmethod
    def _x_of(mem) -> float:
        n = len(mem)
        if n == 0:
            return 0.5
        if n == 1:
            return mem[0]
        first = mem[n - MEMORY_LEN] if n > MEMORY_LEN else mem[0]
        last = mem[n - 1]
        x = last + TREND_GAIN * (last - first)
        if x > 1.0:
            return 1.0
        if x < 0.0:
            return 0.0
        return x

    def _curiosity(self, j: int, now: int) -> float:
        cw = self.curiosity_weight
        if cw <= 0.0:
            return 0.0
        if self.novelty_mode == "field":
            # 「导航信号」D − d：已访问格越靠近未探索区，值越高；梯度恒为 1/步
            v = self.far_cap - self.dflat[j]
            # 「目标信号」未访问格额外给满额奖励，严格占优（不受 d 场滞后影响）
            if self.vflat[j]:
                pass
            else:
                v += self.nb * self.far_cap
            return cw * v

        # ---- block 模式（论文直译，保留作消融对照）----
        inv_h = self._inv_h
        b0, q0, b1, q1, b2, q2, b3, q3 = self.cell_blocks[j]
        bt = self.block_time
        age = (now - bt[b0]) * inv_h
        v = (age if age < 1.0 else 1.0) * q0
        age = (now - bt[b1]) * inv_h
        v += (age if age < 1.0 else 1.0) * q1
        age = (now - bt[b2]) * inv_h
        v += (age if age < 1.0 else 1.0) * q2
        age = (now - bt[b3]) * inv_h
        v += (age if age < 1.0 else 1.0) * q3
        return cw * v

    def _comfort_at(self, j, mem, beta_t):
        w = self._field[j] * self._inv_demand
        if w > 1.0:
            w = 1.0
        elif w < 0.0:
            w = 0.0
        if self.x_mode == "spatial":
            x = self._x_spatial_at(j)
        else:
            x = self._x_of(mem)
        return (1.0 - beta_t) * w + beta_t * x, w

    # ================= 动作价值 =================
    def _value(self, first_j, mem, beta_t, l_prev, peak, hope_on,
               hope_thr, hope_gain, cw, bd_w) -> float:
        """从当前状态出发、第一步走到 first_j，之后按贪心继续走
        (lookahead − 1) 步的累积 drive（折扣累加）。
        lookahead = 1 时退化为论文原式的单步贪心。"""
        now = self.steps
        total = 0.0
        disc = 1.0
        cur = first_j
        m = list(mem)
        l_prev_sim = l_prev
        nbr_table = self._nbr
        steps_left = self.lookahead

        while steps_left > 0:
            l_next, w = self._comfort_at(cur, m, beta_t)
            d_l = l_next - l_prev_sim
            drive = d_l
            if hope_on:
                gap = peak - l_next - hope_thr
                if gap > 0.0:
                    drive += gap * hope_gain
            if bd_w > 0.0:
                drive += self.homeo.boredom_drive(d_l)
            if cw > 0.0:
                drive += self._curiosity(cur, now)
            total += disc * drive
            disc *= LOOKAHEAD_DISCOUNT
            steps_left -= 1
            if steps_left == 0:
                break

            m = m + [w]
            if len(m) > MEMORY_LEN:
                del m[:-MEMORY_LEN]
            l_prev_sim = l_next

            # 前瞻内部的贪心选择：用同一套准则挑下一步
            nbr = nbr_table[cur]
            best_d = -_INF
            nxt = cur
            for b in range(N_ACTIONS):
                jj = nbr[b]
                if jj < 0:
                    continue
                ll, _ = self._comfort_at(jj, m, beta_t)
                ddl = ll - l_prev_sim
                dd = ddl
                if hope_on:
                    g = peak - ll - hope_thr
                    if g > 0.0:
                        dd += g * hope_gain
                if bd_w > 0.0:
                    dd += self.homeo.boredom_drive(ddl)
                if cw > 0.0:
                    dd += self._curiosity(jj, now)
                if dd > best_d + 1e-12:
                    best_d = dd
                    nxt = jj
            cur = nxt
        return total

    def select_action(self) -> int:
        h = self.homeo
        self._inv_demand = 1.0 / h.demand
        beta_t = h.beta_max * (1.0 - math.exp(-h.t / h.beta_tau))
        if beta_t > 1.0:
            beta_t = 1.0
        peak = h.hope_peak
        hope_on = h.use_hope and len(h.hope_buffer) >= HOPE_MIN_LEN
        hope_thr = h.hope_threshold
        hope_gain = h.hope_gain
        cw = self.curiosity_weight
        bd_w = h.boredom_weight
        mem = h.memory
        l_prev = h.L_prev

        nbr = self._nbr[self.pos]
        best = -_INF
        best_actions = None
        for a in range(N_ACTIONS):
            j = nbr[a]
            if j < 0:
                continue
            total = self._value(j, mem, beta_t, l_prev, peak, hope_on,
                                hope_thr, hope_gain, cw, bd_w)
            if total > best + 1e-12:
                best = total
                best_actions = [a]
            elif total > best - 1e-12:
                best_actions.append(a)

        if not best_actions:
            return ACTION_STAY
        return best_actions[self.rng.randrange(len(best_actions))]

    def step(self) -> int:
        a = self.select_action()
        j = self._nbr[self.pos][a]
        if j >= 0 and j != self.pos:
            self.pos = j
            self.moves += 1
        raw = self.env.sense(self.pos)
        if self.x_mode == "spatial":
            self._remember_raw(self.pos, raw)
            self.homeo.observe(raw, x_override=self._x_spatial_at(self.pos))
        else:
            self.homeo.observe(raw)
        self.steps += 1
        if self.novelty_mode == "field":
            if not self.vflat[self.pos]:
                self.vflat[self.pos] = 1
                self._n_unvisited -= 1
                if self._n_unvisited == 0:
                    # 全图走完：把 d 全置为 far_cap ⇒ novelty ≡ 0 ⇒ 由 ΔL 收敛
                    self._recompute_far()
                    self._next_far = 1 << 60
            if self.steps >= self._next_far:
                self._recompute_far()
                self._next_far = self.steps + self.far_interval
        else:
            self.block_time[self._block_of(self.pos)] = self.steps
        return self.pos


__all__ = ["HIMAgent", "CURIOSITY_WEIGHT_DEFAULT", "NOVELTY_HORIZON_DEFAULT",
           "BLOCK_SIZE_DEFAULT", "LOOKAHEAD_DEFAULT",
           "NOVELTY_MODE_DEFAULT", "NEAR_BONUS_DEFAULT",
           "FAR_CAP_DEFAULT", "FAR_INTERVAL_DEFAULT",
           "X_MODE_DEFAULT", "X_LAMBDA_DEFAULT", "X_SOURCE_MIN_DEFAULT"]
