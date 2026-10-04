"""
experiments.metrics —— 实验指标口径定义
=======================================
论文 §3.1 / §4 使用了四个指标，但**没有给出任何一个的计算口径**。
为了让结果可复算、可质疑、可修正，这里把每个指标的口径显式写死并写进报告。

1. 环境覆盖率 coverage
       coverage(t) = |{被访问过的格子}| / N²            N = 100
   即「智能体轨迹扫过的格子占全网格的比例」。

2. 收敛步数 convergence_step
       定义与【3版】§4.3 一致：
       「到覆盖率增长首次低于阈值（连续 W 步新增格子 < δ）时所经过的步数」，
       W = 500，δ = 0.1% × N² = 10 格。
   若整段 episode 都未触发，记 max_steps 且 converged = False。

3. 陷入局部极小值 local_min_trap
       当且仅当：converged == True 且 coverage(收敛时刻) < 50%。
   即"探索提前死亡，且死在一个覆盖率很低的角落里"。

4. 能耗 energy
       论文只给「高于基准线 12.5%」，未说能量单位。这里同时报告三个口径：
         steps     总步数（每步 = 一次完整感知 + L/ΔL 计算 + 动作选择）
         moves     实际位移步数
         per_cell  steps / 已覆盖格子数，「单位覆盖能耗」——检验"多花能耗是否划算"

5. 兴趣点发现 poi_cover_ratio
       被访问过的「兴趣区」格子占全部兴趣区格子的比例。
       这是【A 版】"智能测试方法"的可量化代理：智能体是否主动走向
       曾经让自己更舒服的地方。

6. 末段舒适度 phi_mean_tail / final_phi
       最后 TAIL_WINDOW 步所在格子的 φ 均值，以及终止位置的 φ。
       这是论文「不追求静态绝对舒适度极大值，但最终仍收敛在较高舒适度区域」
       这句话的**可证伪检验**：如果智能体探索完之后随机游走，末段 φ 应当
       接近全场均值（φ_mean_all）；如果它真的"回归舒适"，
       末段 φ 应当显著高于全场均值。
"""

from __future__ import annotations

import random

DEFAULT_MAX_STEPS = 12000
CONV_WINDOW = 500
CONV_THRESHOLD_RATIO = 0.001     # 0.1%
TRAP_COVERAGE_THRESHOLD = 0.5
HIST_STRIDE = 25                 # 覆盖率历史下采样步长（省内存）
TAIL_WINDOW = 1000               # 末段舒适度的统计窗口


def run_episode(agent, env, seed: int, max_steps: int = DEFAULT_MAX_STEPS,
                rng: random.Random | None = None, keep_hist: bool = True):
    """跑一轮 episode，返回指标字典。"""
    rng = rng or random.Random(seed)
    n = env.n_cells
    start = env.random_start(rng)
    field = env.field

    visited = bytearray(n)
    visited[start] = 1
    n_visited = 1

    agent.reset(start)

    conv_threshold = int(CONV_THRESHOLD_RATIO * n)
    conv_step = None
    coverage_at_conv = None

    # 滑动窗口用增量计数，避免每步 sum(500)
    window_sum = CONV_WINDOW
    window = [1] * CONV_WINDOW
    wpos = 0

    # 末段舒适度：定长环形缓冲
    tail = [field[start]] * TAIL_WINDOW
    tail_sum = field[start] * TAIL_WINDOW
    tpos = 0
    phi_sum_all = field[start]

    hist = [1.0 / n] if keep_hist else None
    step_hist_cells = [1] if keep_hist else None

    steps = 0
    while steps < max_steps:
        pos = agent.step()
        steps += 1
        if visited[pos]:
            new = 0
        else:
            visited[pos] = 1
            n_visited += 1
            new = 1

        # 环形窗口
        window_sum += new - window[wpos]
        window[wpos] = new
        wpos += 1
        if wpos == CONV_WINDOW:
            wpos = 0

        # 末段舒适度
        phi = field[pos]
        phi_sum_all += phi
        tail_sum += phi - tail[tpos]
        tail[tpos] = phi
        tpos += 1
        if tpos == TAIL_WINDOW:
            tpos = 0

        if keep_hist and steps % HIST_STRIDE == 0:
            hist.append(n_visited / n)
            step_hist_cells.append(n_visited)

        if conv_step is None and steps >= CONV_WINDOW and window_sum < conv_threshold:
            conv_step = steps
            coverage_at_conv = n_visited / n

    coverage = n_visited / n
    poi_found = sum(1 for i in range(n) if visited[i] and env.poi_flags[i])

    return {
        "seed": seed,
        "start": start,
        "max_steps": max_steps,
        "steps": steps,
        "moves": agent.moves,
        "coverage": coverage,
        "cells_visited": n_visited,
        "poi_found": poi_found,
        "poi_total_cells": env.poi_cells,
        "poi_cover_ratio": poi_found / max(1, env.poi_cells),
        "converged": conv_step is not None,
        "convergence_step": conv_step if conv_step is not None else max_steps,
        "coverage_at_convergence": (
            coverage_at_conv if coverage_at_conv is not None else coverage
        ),
        "local_min_trap": bool(
            conv_step is not None
            and coverage_at_conv is not None
            and coverage_at_conv < TRAP_COVERAGE_THRESHOLD
        ),
        "energy_steps": steps,
        "energy_moves": agent.moves,
        "energy_per_cell": steps / max(1, n_visited),
        "phi_mean_all": phi_sum_all / steps,
        "phi_mean_tail": tail_sum / TAIL_WINDOW,
        "final_phi": field[pos],
        "coverage_hist": hist,
    }


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def summarize(runs: list[dict]) -> dict:
    """把多轮结果聚合成论文表 1 的形态。"""
    m = len(runs)
    if m == 0:
        return {}
    cov = [r["coverage"] for r in runs]
    conv = [r["convergence_step"] for r in runs]
    conv_only = [r["convergence_step"] for r in runs if r["converged"]]
    steps = [r["energy_steps"] for r in runs]
    moves = [r["energy_moves"] for r in runs]
    epc = [r["energy_per_cell"] for r in runs]
    poi = [r["poi_cover_ratio"] for r in runs]
    tail = [r["phi_mean_tail"] for r in runs]
    pall = [r["phi_mean_all"] for r in runs]

    return {
        "n_runs": m,
        "coverage_mean": _mean(cov),
        "coverage_std": _std(cov),
        "coverage_min": min(cov),
        "convergence_step_mean": _mean(conv),
        "convergence_step_mean_converged": _mean(conv_only),
        "converged_ratio": len(conv_only) / m,
        "local_min_trap_ratio": sum(1 for r in runs if r["local_min_trap"]) / m,
        "energy_steps_mean": _mean(steps),
        "energy_moves_mean": _mean(moves),
        "energy_per_cell_mean": _mean(epc),
        "poi_cover_ratio_mean": _mean(poi),
        "phi_mean_tail_mean": _mean(tail),
        "phi_mean_all_mean": _mean(pall),
        "phi_tail_over_all": _mean(tail) / _mean(pall) if _mean(pall) else 0.0,
    }


__all__ = [
    "run_episode", "summarize", "DEFAULT_MAX_STEPS", "CONV_WINDOW",
    "CONV_THRESHOLD_RATIO", "TRAP_COVERAGE_THRESHOLD", "HIST_STRIDE",
    "TAIL_WINDOW",
]
