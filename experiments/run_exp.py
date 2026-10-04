"""
experiments.run_exp —— 一键跑通对比实验
=======================================
    python experiments/run_exp.py --runs 100 --steps 12000

对照智能体
----------
    him           HIM 完整模型（ΔL + 希望 + 区域新奇度）  —— 论文主模型
    him_v1        HIM 原式（ΔL + 希望，无区域新奇度）     —— 论文伪代码直译
    him_nohope    HIM 去掉希望机制（消融）               —— 检验"希望机制"是否关键
    him_nonav     HIM 去掉新奇度的长程导航项（消融）      —— 检验 d 势场是否关键
    him_blocknov  HIM 用「最近访问时刻 + 双线性插值」新奇度（消融）
    ci2018        【A 版 2018】离散性质体状态机           —— 理论母本的工程形态
    fep           标准自由能原理主动推理（最小化惊奇度）
    rl            表格型 Q 学习 + 外部稀疏奖励            —— 论文列出但从未报告结果的基线
    random        均匀随机游走（覆盖率量纲参考）

输出
----
    results/exp_results.json    每个智能体的聚合指标 + 逐轮明细
    results/summary.txt         论文表 1 形态的可读表格
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

# 允许 `python experiments/run_exp.py` 直接运行
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from baseline.fep_model import FEPAgent                       # noqa: E402
from baseline.rl_baseline import RLAgent, RandomAgent         # noqa: E402
from him_core.grid_env import ComfortGrid                     # noqa: E402
from him_core.motivation_engine import HIMAgent               # noqa: E402
from him_core.state_machine import CI2018Agent                # noqa: E402
from experiments.metrics import (                             # noqa: E402
    DEFAULT_MAX_STEPS, HIST_STRIDE, run_episode, summarize,
)

AGENT_ORDER = ["him", "him_v1", "him_nohope", "him_nonav", "him_blocknov",
               "him_wsonly", "him_xonly", "him_xspatial", "him_axiom",
               "him_nosensory", "him_xscramble",
               "ci2018", "fep", "rl", "random"]

AGENT_LABEL = {
    "him": "HIM（ΔL + 希望 + 区域新奇度）",
    "him_v1": "HIM-v1（论文原式：ΔL + 希望）",
    "him_nohope": "HIM 消融：去掉希望机制",
    "him_nonav": "HIM 消融：去掉新奇度长程导航",
    "him_blocknov": "HIM 消融：新奇度用「最近访问时刻」直译",
    "him_wsonly": "公理消融：纯感官（W-only，记忆因素全关）",
    "him_xonly": "公理消融：纯记忆（X-only，β≡1）",
    "him_xspatial": "纯记忆 + 空间记忆判断 X_spatial（A 版 D 体语义）",
    "him_axiom": "A 版语义完整模型（W + X_spatial，β→0.8）",
    "him_nosensory": "奠基检验：无感官需求（感官输入恒定，A 版语义完整模型）",
    "him_xscramble": "奠基检验：记忆与感官脱钩（记忆值=随机数，A 版语义完整模型）",
    "ci2018": "CI-2018（A 版离散性质体状态机）",
    "fep": "FEP（标准主动推理，最小化惊奇度）",
    "rl": "RL（Q 学习 + 外部稀疏奖励）",
    "random": "随机游走（量纲参考）",
}

# 环境参数（论文 §3.1：100×100，随机布局，无外部奖励与预设目标）
ENV_CFG = dict(size=100, n_poi=20, poi_sigma=5.0, poi_amp=0.9,
               background=0.02, obs_noise=0.05)
# 模型超参（论文未给出的部分集中在此，便于复算与扫描；扫描结果见
# results/_sweep.log 与 docs/results_report.md 第 3 节）
MODEL_CFG = dict(
    hope_gain=0.1,            # 【3版】附录值
    beta_max=0.8, beta_tau=300.0,
    boredom_weight=0.0, boredom_ref=0.01,
    # ---- 区域新奇度（论文缺失、本实现补入的算子）----
    curiosity_weight=0.20,    # cw：新奇度的总权重（一步导航梯度 = cw）
    novelty_mode="field",     # field = 距离势场；block = 论文「最近访问时刻」直译
    near_bonus=1.0,           # NB：未访问格的额外满额奖励（单位 D）
    far_cap=96.0,             # D：导航势场的截断距离（格）；扫描见 _validate.log
    far_interval=10,          # d 场重算间隔（步）
    lookahead=1,              # 前瞻深度；1 = 论文原式单步贪心
)



# 「无感官需求」检验臂使用的感官代理：无论智能体走到哪，sense() 恒返回
# 同一常数——身体状态与需求状态的匹配不再随环境变化，即 A 版 §4 语义下
# 「不存在会随环境变化的感官舒服程度」。真实环境保留给指标层：
# run_episode 仍用真实 field 计算 φ，检验的正是"它的行为在真实世界里
# 还有没有意义"。代理只暴露智能体用到的属性。
class _FlatSenseEnv:
    def __init__(self, env: ComfortGrid, flat: float = 0.5):
        import numpy as np
        self._flat = float(flat)
        self.field = [self._flat] * env.n_cells
        self.size = env.size
        self.n_cells = env.n_cells
        self.nbr = env.nbr

    def sense(self, pos: int) -> float:
        return self._flat


# 需要感官代理的臂
FLAT_SENSE_AGENTS = {"him_nosensory"}


def make_agent(name: str, env: ComfortGrid, rng: random.Random):
    common = dict(beta_max=MODEL_CFG["beta_max"], beta_tau=MODEL_CFG["beta_tau"],
                  boredom_weight=MODEL_CFG["boredom_weight"],
                  boredom_ref=MODEL_CFG["boredom_ref"],
                  lookahead=MODEL_CFG["lookahead"])
    nov = dict(novelty_mode=MODEL_CFG["novelty_mode"],
               curiosity_weight=MODEL_CFG["curiosity_weight"],
               near_bonus=MODEL_CFG["near_bonus"],
               far_cap=MODEL_CFG["far_cap"],
               far_interval=MODEL_CFG["far_interval"])
    if name == "him":
        # 主模型：ΔL + 希望 + 区域新奇度（距离势场）
        return HIMAgent(env, rng, hope_gain=MODEL_CFG["hope_gain"],
                        use_hope=True, variant="him", **common, **nov)
    if name == "him_v1":
        # 论文伪代码直译：drive = ΔL + hope_drive，无区域新奇度，单步贪心
        return HIMAgent(env, rng, hope_gain=MODEL_CFG["hope_gain"],
                        use_hope=True, variant="him_v1",
                        beta_max=MODEL_CFG["beta_max"],
                        beta_tau=MODEL_CFG["beta_tau"],
                        boredom_weight=0.0, lookahead=1,
                        curiosity_weight=0.0, novelty_mode="field")
    if name == "him_nohope":
        # 消融：去掉希望机制，其余同 him
        return HIMAgent(env, rng, use_hope=False, variant="him_nohope",
                        **common, **nov)
    if name == "him_nonav":
        # 消融：把导航势场截断到 1 格 ⇒ 只剩「未访问加分」，没有指向性导航
        nanov = dict(nov)
        nanov["far_cap"] = 1.0
        return HIMAgent(env, rng, hope_gain=MODEL_CFG["hope_gain"],
                        use_hope=True, variant="him_nonav", **common, **nanov)
    if name == "him_blocknov":
        # 消融：新奇度用论文「最近访问时刻 + 双线性插值」的直译
        return HIMAgent(env, rng, hope_gain=MODEL_CFG["hope_gain"],
                        use_hope=True, variant="him_blocknov",
                        beta_max=MODEL_CFG["beta_max"],
                        beta_tau=MODEL_CFG["beta_tau"],
                        boredom_weight=MODEL_CFG["boredom_weight"],
                        boredom_ref=MODEL_CFG["boredom_ref"],
                        lookahead=MODEL_CFG["lookahead"],
                        curiosity_weight=MODEL_CFG["curiosity_weight"],
                        novelty_mode="block", block_size=8,
                        novelty_horizon=1500.0)
    if name == "him_wsonly":
        # 公理消融（A 版 2018 §7.2「L 包括 W、X。W、X 同时存在」）：
        # 纯感官臂。β≡0 ⇒ L = W（舒适度只由当下感官决定），
        # 并关闭全部记忆侧驱动（希望、区域新奇度）。
        # 检验命题：只保留非记忆因素，探索行为是否消失。
        return HIMAgent(env, rng, use_hope=False, variant="him_wsonly",
                        beta_max=0.0, beta_tau=MODEL_CFG["beta_tau"],
                        boredom_weight=0.0, lookahead=1,
                        curiosity_weight=0.0, novelty_mode="block")
    if name == "him_xonly":
        # 公理消融：纯记忆臂。β_max=1 ⇒ L = X（舒适度只由记忆参与决定），
        # 保留记忆侧驱动（希望 + 区域新奇度）。
        # 检验命题：只保留记忆因素，探索与回归是否仍成立。
        return HIMAgent(env, rng, hope_gain=MODEL_CFG["hope_gain"],
                        use_hope=True, variant="him_xonly",
                        beta_max=1.0, beta_tau=MODEL_CFG["beta_tau"],
                        boredom_weight=0.0,
                        lookahead=MODEL_CFG["lookahead"], **nov)
    if name == "him_xspatial":
        # X_spatial 检验臂（A 版 D 体的可计算形式，作者 2026-09-24 指正后补入）：
        # 纯记忆臂（β≡1）+ 空间记忆判断 X。
        # X_spatial(j) = max(0.5, max_源 best[k]·max(0, 1−d(j,k)/R))，R=160。
        # 「我记得那里有多舒服（best[k]）+ 过去要走的路（d）」——作者原例：
        # 「还没吃，但知道一分钟后能吃 = 记忆需求已满足」的可计算形式。
        # 预测：xonly 回不了家（末段/全程 φ = 0.810）的根因是 X 对位置不敏感；
        # 换成位置敏感的空间记忆判断后，纯记忆也应当能回家（末段/全程 φ > 1）。
        return HIMAgent(env, rng, hope_gain=MODEL_CFG["hope_gain"],
                        use_hope=True, variant="him_xspatial",
                        beta_max=1.0, beta_tau=MODEL_CFG["beta_tau"],
                        boredom_weight=0.0,
                        lookahead=MODEL_CFG["lookahead"],
                        x_mode="spatial", **nov)
    if name == "him_axiom":
        # A 版语义完整模型：W + X_spatial，β→0.8（B 版 X 公式被按 A 版
        # 底层原理修正后的"应有的主模型"形态）。
        return HIMAgent(env, rng, hope_gain=MODEL_CFG["hope_gain"],
                        use_hope=True, variant="him_axiom",
                        **common, x_mode="spatial", **nov)
    if name == "him_nosensory":
        # 奠基检验臂一（命题：没有感官需求 ⇒ 不存在有内容的记忆需求）。
        # 感官输入恒定（run_agent 传入 _FlatSenseEnv 代理）：无论走到哪，
        # 身体状态都不变 ⇒ 没有任何「舒服程度转变」可供记忆记录（A 版 §4）。
        # 预测：X_spatial 恒为中性 0.5，记忆需求判断永远「无内容」；
        # ΔL≡0、希望恒 0；全图走完后所有动作并列 → 退化为随机漫游，
        # 无家可归（末段 φ ≈ 全场随机水平）。
        return HIMAgent(env, rng, hope_gain=MODEL_CFG["hope_gain"],
                        use_hope=True, variant="him_nosensory",
                        **common, x_mode="spatial", **nov)
    if name == "him_xscramble":
        # 奠基检验臂二（命题：记忆需求由感官需求塑造——记忆的内容必须是
        # 感官舒服度的痕迹）。感官正常，但空间记忆里每格登记的
        # 「那里有多舒服」换成与感官无关的随机数（结构保留、内容脱钩）。
        # 若命题不成立，该臂应与 him_axiom 表现一致；若成立，
        # 记忆判断全部失真 ⇒ 回家回错地方（末段 φ 崩塌到随机水平）。
        return HIMAgent(env, rng, hope_gain=MODEL_CFG["hope_gain"],
                        use_hope=True, variant="him_xscramble",
                        **common, x_mode="spatial", x_scramble=True, **nov)
    if name == "ci2018":
        return CI2018Agent(env, rng)
    if name == "fep":
        return FEPAgent(env, rng)
    if name == "rl":
        return RLAgent(env, rng)
    if name == "random":
        return RandomAgent(env, rng)
    raise ValueError(f"未知智能体：{name}")


def run_agent(name: str, n_runs: int, max_steps: int, seed_base: int = 1000,
              progress: bool = True):
    runs = []
    t0 = time.time()
    for i in range(n_runs):
        seed = seed_base + i
        env = ComfortGrid(seed=seed, **ENV_CFG)
        rng = random.Random(seed * 31 + 7)
        env_for_agent = _FlatSenseEnv(env) if name in FLAT_SENSE_AGENTS else env
        agent = make_agent(name, env_for_agent, rng)
        r = run_episode(agent, env, seed=seed, max_steps=max_steps,
                        rng=random.Random(seed * 17 + 3))
        runs.append(r)
        if progress and (i + 1) % 10 == 0:
            print(f"    [{name}] {i + 1}/{n_runs} 轮  "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return runs


def mean_curve(runs):
    """把多轮的覆盖率历史对齐长度后取均值（供出图用，JSON 里不存逐轮明细）。"""
    hists = [r["coverage_hist"] for r in runs if r.get("coverage_hist")]
    if not hists:
        return []
    m = min(len(h) for h in hists)
    out = [0.0] * m
    for h in hists:
        for k in range(m):
            out[k] += h[k]
    inv = 1.0 / len(hists)
    return [v * inv for v in out]


def main():
    ap = argparse.ArgumentParser(description="HIM 对比实验（论文 §3–§4 复现口径）")
    ap.add_argument("--runs", type=int, default=100, help="每个智能体的独立运行轮数")
    ap.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS, help="每轮最大步数")
    ap.add_argument("--agents", nargs="*", default=AGENT_ORDER, help="要跑的智能体")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--outdir", default=os.path.join(_ROOT, "results"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("HIM / FEP / RL 对比实验")
    print(f"  环境：100×100 离散网格，{ENV_CFG['n_poi']} 个兴趣点，无外部奖励与预设目标")
    print(f"  每轮最大步数：{args.steps}；独立运行轮数：{args.runs}")
    print("=" * 78)

    all_runs = {}
    all_sum = {}
    all_curve = {}
    for name in args.agents:
        print(f"  → {AGENT_LABEL.get(name, name)}")
        runs = run_agent(name, args.runs, args.steps, args.seed_base)
        all_runs[name] = runs
        all_sum[name] = summarize(runs)
        all_curve[name] = mean_curve(runs)

    # ---------------- 可读表格（论文表 1 形态） ----------------
    lines = []
    lines.append("=" * 96)
    lines.append("表 1  HIM 与各基线模型实验指标对比（口径见 experiments/metrics.py）")
    lines.append("=" * 96)
    hdr = (f"{'指标':<22}" + "".join(f"{n:>13}" for n in args.agents))
    lines.append(hdr)
    lines.append("-" * 96)

    def row(label, key, fmt="{:.4f}"):
        vals = []
        for n in args.agents:
            v = all_sum[n].get(key, 0.0)
            vals.append(fmt.format(v))
        return f"{label:<22}" + "".join(f"{v:>13}" for v in vals)

    lines.append(row("平均环境覆盖率", "coverage_mean", "{:.2%}"))
    lines.append(row("  覆盖率标准差", "coverage_std", "{:.2%}"))
    lines.append(row("  覆盖率最小值", "coverage_min", "{:.2%}"))
    lines.append(row("平均收敛步数", "convergence_step_mean", "{:.1f}"))
    lines.append(row("  已收敛轮的收敛步数", "convergence_step_mean_converged", "{:.1f}"))
    lines.append(row("  收敛率", "converged_ratio", "{:.1%}"))
    lines.append(row("陷入局部极小值比例", "local_min_trap_ratio", "{:.1%}"))
    lines.append(row("平均总步数（能耗）", "energy_steps_mean", "{:.1f}"))
    lines.append(row("平均位移步数", "energy_moves_mean", "{:.1f}"))
    lines.append(row("单位覆盖能耗", "energy_per_cell_mean", "{:.4f}"))
    lines.append(row("兴趣区覆盖率", "poi_cover_ratio_mean", "{:.2%}"))
    lines.append(row("末段 φ 均值（最后1000步）", "phi_mean_tail_mean", "{:.4f}"))
    lines.append(row("全程 φ 均值", "phi_mean_all_mean", "{:.4f}"))
    lines.append(row("  末段/全程 φ 之比", "phi_tail_over_all", "{:.3f}"))
    lines.append("=" * 96)
    lines.append("说明：末段 φ 均值显著高于全程 φ 均值 ⇒ 「探索完之后确实回归到较舒适区域」；")
    lines.append("      若两者接近，说明末段只是无差别随机游走。")
    lines.append("      参照系：φ 全场均值 ≈ 0.212，中位数 ≈ 0.086，90 分位 ≈ 0.640，上限 0.900。")
    lines.append("=" * 96)

    # ---------------- 相对能耗（以 FEP 为基准） ----------------
    if "fep" in all_sum:
        base_steps = all_sum["fep"]["energy_steps_mean"]
        base_epc = all_sum["fep"]["energy_per_cell_mean"]
        lines.append("")
        lines.append("相对能耗（基准 = FEP）")
        lines.append("-" * 96)
        lines.append(f"{'模型':<34}{'总步数':>14}{'相对总能耗':>14}{'单位覆盖能耗':>16}{'相对单位能耗':>16}")
        for n in args.agents:
            s = all_sum[n]
            rel = s["energy_steps_mean"] / base_steps if base_steps else 0.0
            re_epc = s["energy_per_cell_mean"] / base_epc if base_epc else 0.0
            lines.append(
                f"{AGENT_LABEL.get(n, n):<34}"
                f"{s['energy_steps_mean']:>14.1f}"
                f"{rel:>13.1%}"
                f"{s['energy_per_cell_mean']:>16.4f}"
                f"{re_epc:>15.1%}"
            )
        lines.append("=" * 96)

    text = "\n".join(lines)
    print()
    print(text)

    summary_path = os.path.join(args.outdir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    # ---------------- JSON ----------------
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "env_cfg": ENV_CFG,
        "model_cfg": MODEL_CFG,
        "n_runs": args.runs,
        "max_steps": args.steps,
        "seed_base": args.seed_base,
        "curve_stride": HIST_STRIDE,
        "agents": {
            n: {
                "label": AGENT_LABEL.get(n, n),
                "summary": all_sum[n],
                # 覆盖率曲线（多轮均值，下采样步长 = metrics.HIST_STRIDE）
                "coverage_curve_mean": all_curve[n],
                "curve_stride": HIST_STRIDE,
                # 逐轮明细不含覆盖率曲线，避免文件过大
                "runs": [
                    {k: v for k, v in r.items() if k != "coverage_hist"}
                    for r in all_runs[n]
                ],
            }
            for n in args.agents
        },
    }
    json_path = os.path.join(args.outdir, "exp_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n结果已写入：\n  {summary_path}\n  {json_path}")
    return payload


if __name__ == "__main__":
    main()
