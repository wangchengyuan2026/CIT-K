"""
him_core.homeostasis —— 舒适度函数与内生奖励信号
==================================================
严格对应论文 §2「HIM 模型定义」与【3版】§2「三条公式与三座桥梁」。

三条公式
--------
Eq1（定义层，仅【3版】保留，且论文自认"不可直接计算"）：
    舒适程度 = 感官实时状态 / 感官需求状态 + 记忆实时状态 / 记忆需求状态

Eq2（预测层，可计算形式）：
    L_t = (1 - β_t) · W_t + β_t · X_t

Eq3（驱动层，内生奖励信号）：
    ΔL_t = L_t - L_{t-1}

三座桥梁【3版】§2.2 / §2.4 / §2.5
----------------------------------
桥梁1「映射函数」：把 Eq1 的比值结构落成
    W_t = sensory_experience(raw, demand) = clamp(raw / demand, 0, 1)
    X_t = predictive_expectation(memory)
桥梁2「差分算子」：ΔL_t = compute_comfort_gradient(L_t, L_prev)
额外算子「希望机制」：
    hope_drive = max(0, peak(history) - L_t - threshold)

实现说明（与论文的偏差，均已在 docs/results_report.md 中登记）
------------------------------------------------------------
1. demand 默认取 1.0，即 W_t = clamp(φ, 0, 1)。原始 Eq1 的两个分母
   （感官需求状态 / 记忆需求状态）在论文中未给出可计算形式，这里将其
   归一化掉；它是可配置参数，扫描它会改变梯度锐度。
2. β_t 取 β_max·(1 - e^{-t/τ})，实现论文「探索初期更依赖感官（β小），
   记忆积累后更依赖预期（β大）」的描述。
3. predictive_expectation 采用【3版】附录的实现：
       if len(memory) < 2: return 0.5
       recent = memory[-10:]; trend = recent[-1] - recent[0]
       return clamp(recent[-1] + 0.3 * trend, 0, 1)
4. hope_drive 采用【3版】附录的 *0.1 增益（正文未给增益，附录给了）。
   该增益偏小，因此在实验中额外做了 hope_gain 敏感性扫描。
5. 论文里希望缓冲池写作 "W Memory Buffer"，与本文件中的 W_t（感官经验）
   属同字母不同义 —— 这里沿用论文命名，同时在代码注释中标出该冲突。
"""

from __future__ import annotations

import math

BETA_MAX_DEFAULT = 0.8
BETA_TAU_DEFAULT = 300.0
MEMORY_LEN = 10
TREND_GAIN = 0.3
HOPE_THRESHOLD_DEFAULT = 0.15
HOPE_GAIN_DEFAULT = 0.1      # 【3版】附录值
HOPE_BUFFER_K = 64           # 只保留 top-k 高舒适度状态
HOPE_MIN_LEN = 5             # 【3版】附录：len(buffer) < 5 时不出力
BOREDOM_WEIGHT_DEFAULT = 0.03   # 停滞压力权重（0 = 退回论文原式）
BOREDOM_REF_DEFAULT = 0.01      # |ΔL| 达到该量级即视为"有进展"


def clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class Homeostasis:
    """单个智能体的内稳态层：产出 W / X / L / ΔL / hope。"""

    __slots__ = (
        "demand", "beta_max", "beta_tau", "hope_threshold", "hope_gain",
        "use_hope", "boredom_weight", "boredom_ref",
        "memory", "hope_buffer", "hope_peak", "L_prev", "t",
        "last_W", "last_X", "last_L", "last_dL", "last_hope",
    )

    def __init__(
        self,
        demand: float = 1.0,
        beta_max: float = BETA_MAX_DEFAULT,
        beta_tau: float = BETA_TAU_DEFAULT,
        hope_threshold: float = HOPE_THRESHOLD_DEFAULT,
        hope_gain: float = HOPE_GAIN_DEFAULT,
        use_hope: bool = True,
        boredom_weight: float = BOREDOM_WEIGHT_DEFAULT,
        boredom_ref: float = BOREDOM_REF_DEFAULT,
    ):
        self.demand = demand
        self.beta_max = beta_max
        self.beta_tau = beta_tau
        self.hope_threshold = hope_threshold
        self.hope_gain = hope_gain
        self.use_hope = use_hope
        # 停滞压力（boredom）：论文文字主张"ΔL≈0 会产生内在压力去探索新可能"，
        # 但公式 drive = ΔL + hope_drive 在 ΔL=0 处取不到任何东西，
        # 导致 argmax 永远选择"原地不动"。boredom_weight=0 即退回论文原式。
        self.boredom_weight = boredom_weight
        self.boredom_ref = boredom_ref

        self.memory: list[float] = []        # 感官经验序列（X_t 的原料）
        self.hope_buffer: list[float] = []   # 论文的 "W Memory Buffer"，存历史高舒适度
        self.hope_peak: float = 0.0          # max(hope_buffer)，增量维护以免每步 O(k)
        self.L_prev: float = 0.0
        self.t: int = 0
        self.last_W = 0.0
        self.last_X = 0.0
        self.last_L = 0.0
        self.last_dL = 0.0
        self.last_hope = 0.0

    # ---------------- 桥梁1：映射函数 ----------------
    def sensory_experience(self, raw: float) -> float:
        """W_t = clamp(raw / demand, 0, 1)。raw/demand 即 Eq1 的「状态/需求」比值。"""
        return clamp01(raw / self.demand)

    def _x_from(self, mem_tail: list[float]) -> float:
        if len(mem_tail) < 2:
            return 0.5
        recent = mem_tail[-MEMORY_LEN:]
        trend = recent[-1] - recent[0]
        return clamp01(recent[-1] + TREND_GAIN * trend)

    def predictive_expectation(self) -> float:
        """X_t：基于历史记忆形成的未来状态预期。"""
        return self._x_from(self.memory)

    def predictive_expectation_with(self, w_next: float) -> float:
        """保留接口；等价于当前 X（见 prospective_drive 的一致性说明）。"""
        return self.predictive_expectation()

    # ---------------- β_t 自适应权重 ----------------
    def beta(self, t: int | None = None) -> float:
        """β_t ∈ [0,1]：探索初期偏感官（β小），记忆积累后偏预期（β大）。"""
        tt = self.t if t is None else t
        return self.beta_max * (1.0 - math.exp(-tt / self.beta_tau))

    # ---------------- Eq2：舒适度 ----------------
    @staticmethod
    def comfort(W: float, X: float, beta: float) -> float:
        return (1.0 - beta) * W + beta * X

    # ---------------- 额外算子：希望机制 ----------------
    def hope_drive(self, L: float) -> float:
        """hope_drive(L_t, buffer) = max(0, peak(buffer) - L_t - threshold) * gain"""
        if not self.use_hope or len(self.hope_buffer) < HOPE_MIN_LEN:
            return 0.0
        peak = self.hope_peak
        gap = peak - L - self.hope_threshold
        if gap <= 0.0:
            return 0.0
        return gap * self.hope_gain

    # ---------------- 前瞻：给定候选格子的感官经验，预测 drive ----------------
    def boredom_drive(self, d_l: float) -> float:
        """停滞压力：论文【3版】§5.2 主张「只要 ΔL ≈ 0，就会产生内在压力，
        主动尝试新行为以寻求舒适度提升」。原式 drive = ΔL + hope_drive 在
        ΔL = 0 处恒为 0，argmax 永远选"原地不动"，与该主张矛盾。
        这里把它显式化为：|ΔL| 越小，惩罚越接近 boredom_weight。
        boredom_weight = 0 时完全退回论文原式。"""
        if self.boredom_weight <= 0.0:
            return 0.0
        frac = abs(d_l) / self.boredom_ref
        if frac >= 1.0:
            return 0.0
        return -self.boredom_weight * (1.0 - frac)

    def prospective_drive(self, w_next: float, beta_t: float) -> float:
        """ΔL' + hope_drive(L') + boredom —— 论文 §3 逻辑链末端的行动判决量。

        ⚠ 关键一致性：X' 必须与 observe() 里算 X 的口径完全相同。
        observe() 计算 X_t 时用的是"当前记忆"（还不含本步的 W_t），
        所以前瞻时 X' 也应当取当前记忆算出的 X，**与候选动作无关**。
        早期版本让 X' 依赖 w_next（把 w_next 当作已进入记忆），
        结果"原地不动"被算成 ΔL = −0.075 而非 0，动作排序被彻底扭曲。
        """
        x_next = self.predictive_expectation()
        l_next = self.comfort(w_next, x_next, beta_t)
        d_l = l_next - self.L_prev
        return d_l + self.hope_drive(l_next) + self.boredom_drive(d_l)

    # ---------------- 观测更新 ----------------
    def observe(self, raw: float,
                x_override: float | None = None) -> tuple[float, float, float, float]:
        """感知一次 → 返回 (W, X, L, ΔL)。

        x_override：外部提供的记忆项 X。x_mode="spatial"（A 版 D 体语义的
        空间记忆判断）时由动机引擎传入；None 时用内部趋势外推（B 版原式），
        保持向后兼容。"""
        beta_t = self.beta()
        W = self.sensory_experience(raw)
        X = self.predictive_expectation() if x_override is None \
            else clamp01(x_override)
        L = self.comfort(W, X, beta_t)
        dL = L - self.L_prev

        self.memory.append(W)
        if len(self.memory) > MEMORY_LEN * 4:
            del self.memory[:-MEMORY_LEN]

        # Update W_buffer with top-k high-comfort states
        if len(self.hope_buffer) < HOPE_BUFFER_K:
            self.hope_buffer.append(L)
            if L > self.hope_peak:
                self.hope_peak = L
        else:
            lo = min(self.hope_buffer)
            if L > lo:
                self.hope_buffer.remove(lo)
                self.hope_buffer.append(L)
                if L > self.hope_peak:
                    self.hope_peak = L

        self.L_prev = L
        self.t += 1
        self.last_W, self.last_X, self.last_L, self.last_dL = W, X, L, dL
        return W, X, L, dL


__all__ = ["Homeostasis", "clamp01", "MEMORY_LEN", "TREND_GAIN",
           "HOPE_THRESHOLD_DEFAULT", "HOPE_GAIN_DEFAULT",
           "BOREDOM_WEIGHT_DEFAULT", "BOREDOM_REF_DEFAULT"]
