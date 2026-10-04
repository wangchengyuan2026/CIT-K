"""
him_core.state_machine —— 【A 版 2018】离散性质体状态机
========================================================
实现《反思智能：存在自主动机的最简机器模型设计》（王程远, 2018）§6–§7 的
「存在自主动机的最简抽象智能机器」。

论文的符号体系（17 个性质体）在这里只落地能与网格环境对应的那部分：

    G  外感官                 G-1 ∈ {A..F} 六档感官量化等级；G-2 ∈ {A, B} 粗粒度
    Q  外感官感知体           Q-1 = f(G-1)，Q-2 = f(G-2)（论文：1:1 对应）
    M  非记忆需求体           M-A = 需求高（饿），M-B = 需求低（饱）
    C  重叠程度体             C-1 当感官等级满足 M 所要求的等级，否则 C-0
    W  非记忆舒服程度感知体   C-0 → W-0，C-1 → W-1
    Z  记忆体                 记录「哪个 G-1 等级曾带来过 W-1」
    S-TZ / S-T                当前识别出的记忆符号 / 记忆里的需求符号
    D  记忆重叠程度体         D-1 当 S-TZ 与 S-T 相符，否则 D-0
    X  记忆舒服程度感知体     D-0 → X-0，D-1 → X-1
    L  舒服程度感知体         L ∈ {L-00, L-01, L-10, L-11}：(W, X) 四种组合
    T  记忆需求体             W-0 时：「找出使 W-0 转变为 W-1 的路径」
                              W-1 时：「找出比前面更长远有效防止 W-1 转变为 W-0 的方法」
    H  控制体                 只控制「感官在外部世界中的运动方向」，不控制位置
    近路原则                  能提升 L 或防止 L 下跌的多条路径中，必选最近的一条

实现中的三处澄清（已登记在 docs/results_report.md）
--------------------------------------------------
1. **C 的判据**。论文 §7.3 第 6 条把 C-1（→舒服）定义为 Q-2 与 M **取值相同**，
   照字面实现会让「需求低 M-B + 环境好 Q-2-A」被判为不适（W-0），
   即吃饱后待在食物充足处反而难受。本实现改为「感官等级是否达到 M 要求的等级」：
   C-1 ⟺ G-1 等级 ≥ level(M)。这与原表在 Q-2-A/M-A、Q-2-B/M-A 上完全一致，
   仅把 Q-2-B/M-B 中的等级 0 一档由 C-1 改为 C-0。
2. **M 的动力学**：W-1 → M-B，W-0 → M-A，实现论文「不同的机体状态对感官状态
   要求不同」；level(M-A)=2 表示"需要较好的环境"，level(M-B)=1 表示"已有余裕，
   只需不恶化"。
3. **Z 的写入时机**：论文称「'W-0 转变为 W-1'过程中的关键 Z 将被抽离出来写到 T」，
   本实现在 W-0→W-1 跃迁发生时把该 G-1 等级记入 Z。
"""

from __future__ import annotations

import random

from .grid_env import ACTION_STAY, N_ACTIONS

G1_LEVELS = 6                                  # G-1 的六种状态 A..F
L_STATES = ("L-00", "L-01", "L-10", "L-11")    # (W, X) 组合
M_A, M_B = 0, 1                                # 非记忆需求体的两种状态
LEVEL_OF_M = {M_A: 2, M_B: 1}                  # M-A 需要等级≥2，M-B 只需≥1


class CI2018Agent:
    """A 版离散状态机智能体。"""

    name = "ci2018"

    def __init__(self, env, rng: random.Random):
        self.env = env
        self.rng = rng
        self.quant_edges = [i / G1_LEVELS for i in range(1, G1_LEVELS)]

        self.pos = -1
        self.steps = 0
        self.moves = 0
        self.M = M_A
        self.W = 0
        self.X = 0
        self.L_state = "L-00"
        self.Z = [False] * G1_LEVELS
        self.last_g1 = 0
        self._nbr = env.nbr
        self._field = env.field

    # ------------------------------------------------------------------
    def _quantize(self, phi: float) -> int:
        lv = 0
        for e in self.quant_edges:
            if phi >= e:
                lv += 1
            else:
                break
        return min(lv, G1_LEVELS - 1)

    def reset(self, start: int):
        self.pos = start
        self.steps = 0
        self.moves = 0
        self.M = M_A
        self.W = 0
        self.X = 0
        self.L_state = "L-00"
        self.Z = [False] * G1_LEVELS
        self._update(self.env.sense(self.pos), commit_z=False)

    # ------------------------------------------------------------------
    def _update(self, raw: float, commit_z: bool = True):
        g1 = self._quantize(raw)
        prev_W = self.W

        # O 开启 → Q-2 与 M 共同塑造 C，C 决定 W
        C = 1 if g1 >= LEVEL_OF_M[self.M] else 0
        self.W = C

        # S-TZ 与 S-T → D → X
        self.X = 1 if self.Z[g1] else 0

        # 「W-0 转变为 W-1」时的关键 Z 被抽离出来写到 T
        if commit_z and prev_W == 0 and self.W == 1:
            self.Z[g1] = True

        # 需求状态随舒服状态翻转
        self.M = M_B if self.W == 1 else M_A

        self.L_state = L_STATES[self.W * 2 + self.X]
        self.last_g1 = g1

    # ------------------------------------------------------------------
    def select_action(self) -> int:
        """T（记忆需求体）+ 近路原则 决定动作；H 只决定 G 在 B 中的运动方向。"""
        nbr = self._nbr[self.pos]
        field = self._field
        rng = self.rng

        cand = []
        for a in range(N_ACTIONS):
            j = nbr[a]
            if j < 0:
                continue
            cand.append((a, j, self._quantize(field[j]), field[j]))

        cur_level = self._quantize(field[self.pos])

        if self.W == 0:
            # T：找出使 W-0 转变为 W-1 的路径
            need = LEVEL_OF_M[self.M]
            good = [c for c in cand if c[2] >= need]
            if good:
                # 近路原则：等长路径中取感官等级最高者（最短、最有效）
                top = max(c[2] for c in good)
                best = [c[0] for c in good if c[2] == top]
                return best[rng.randrange(len(best))]
            # 无一步可达的满足态 → 朝等级更高处走（G 在 B 各状态间运动）
            top = max(c[2] for c in cand)
            if top > cur_level:
                best = [c[0] for c in cand if c[2] == top]
                return best[rng.randrange(len(best))]
            return cand[rng.randrange(len(cand))][0]
        else:
            # T：找出比前面更长远有效防止 W-1 转变为 W-0 的方法
            #     → 提高舒适裕度，朝 φ 更高处走
            cur = field[self.pos]
            top = max(c[3] for c in cand)
            if top > cur + 1e-12:
                best = [c[0] for c in cand if abs(c[3] - top) <= 1e-12]
                return best[rng.randrange(len(best))]
            # 已是局部最高 → H 让 G 在 B 各状态间运动（随机探索）
            return cand[rng.randrange(len(cand))][0]

    # ------------------------------------------------------------------
    def step(self) -> int:
        a = self.select_action()
        j = self._nbr[self.pos][a]
        if j >= 0 and j != self.pos:
            self.pos = j
            self.moves += 1
        self._update(self.env.sense(self.pos))
        self.steps += 1
        return self.pos


__all__ = ["CI2018Agent", "G1_LEVELS", "L_STATES", "M_A", "M_B", "LEVEL_OF_M"]
