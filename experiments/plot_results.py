"""
experiments.plot_results —— 实验结果出图
========================================
    python experiments/plot_results.py

读 results/exp_results.json，产出四张图到 results/figures/：

    fig1_coverage_curve.png   各智能体平均覆盖率随步数的增长曲线
    fig2_coverage_bar.png     最终覆盖率 / 兴趣区覆盖率 / 陷入率的对比柱状图
    fig3_trajectory.png       HIM 与论文原式的轨迹对照（同一环境、同一种子）
    fig4_ablation.png         HIM 各消融变体的覆盖率对比

所有图都用中文字体渲染（Windows 上优先 Microsoft YaHei / SimHei）。
"""

from __future__ import annotations

import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.metrics import run_episode, HIST_STRIDE   # noqa: E402
from him_core.grid_env import ComfortGrid                  # noqa: E402
from him_core.motivation_engine import HIMAgent            # noqa: E402

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = os.path.join(_ROOT, "results", "figures")

# 视觉顺序 + 配色（涨/正为暖色、跌/负为冷色的中式习惯只用于第一张柱状图）
ORDER = ["him", "him_v1", "him_nohope", "him_nonav", "him_blocknov",
         "ci2018", "fep", "rl", "random"]
COLOR = {
    "him": "#d62728",
    "him_v1": "#ff7f0e",
    "him_nohope": "#8c564b",
    "him_nonav": "#e377c2",
    "him_blocknov": "#bcbd22",
    "ci2018": "#9467bd",
    "fep": "#1f77b4",
    "rl": "#2ca02c",
    "random": "#7f7f7f",
}
LABEL = {
    "him": "HIM（ΔL+希望+新奇度）",
    "him_v1": "HIM-v1 论文原式",
    "him_nohope": "消融：无希望",
    "him_nonav": "消融：无长程导航",
    "him_blocknov": "消融：新奇度用时间戳直译",
    "ci2018": "CI-2018 状态机",
    "fep": "FEP 主动推理",
    "rl": "RL 外部奖励",
    "random": "随机游走",
}


def _load(path=None):
    path = path or os.path.join(_ROOT, "results", "exp_results.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fig1_curve(payload):
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=140)
    stride = payload.get("curve_stride", HIST_STRIDE)
    for name in ORDER:
        if name not in payload["agents"]:
            continue
        curve = payload["agents"][name].get("coverage_curve_mean") or []
        if not curve:
            continue
        x = np.arange(len(curve)) * stride
        lw = 2.4 if name == "him" else 1.5
        ax.plot(x, np.asarray(curve) * 100.0, label=LABEL.get(name, name),
                color=COLOR.get(name, None), linewidth=lw,
                zorder=5 if name == "him" else 2)
    ax.set_xlabel("步数")
    ax.set_ylabel("环境覆盖率（%）")
    ax.set_title("各智能体覆盖率增长曲线（多轮均值）")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right", ncol=2)
    ax.set_ylim(0, 104)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig1_coverage_curve.png")
    fig.savefig(p)
    plt.close(fig)
    print("  写出", p)


def fig2_bar(payload):
    names = [n for n in ORDER if n in payload["agents"]]
    cov = [payload["agents"][n]["summary"]["coverage_mean"] * 100 for n in names]
    poi = [payload["agents"][n]["summary"]["poi_cover_ratio_mean"] * 100
           for n in names]
    trap = [payload["agents"][n]["summary"]["local_min_trap_ratio"] * 100
            for n in names]
    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=140)
    b1 = ax.bar(x - w, cov, w, label="环境覆盖率", color="#c0392b")
    b2 = ax.bar(x, poi, w, label="兴趣区覆盖率", color="#e59866")
    b3 = ax.bar(x + w, trap, w, label="陷入局部极小值比例", color="#5499c7")
    for bars in (b1, b2, b3):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.1f}", (r.get_x() + r.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(n, n) for n in names], rotation=18,
                       ha="right", fontsize=8.4)
    ax.set_ylabel("百分比（%）")
    ax.set_title("表 1 各指标对比")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 112)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig2_coverage_bar.png")
    fig.savefig(p)
    plt.close(fig)
    print("  写出", p)


def fig3_trajectory(seed=1000, steps=8000):
    """同一环境、同一初始位置下 HIM 与论文原式的轨迹对照。"""
    cfg = [
        ("him", dict(curiosity_weight=0.20, novelty_mode="field",
                     near_bonus=1.0, far_cap=96.0, far_interval=10)),
        ("him_v1", dict(curiosity_weight=0.0, novelty_mode="field")),
    ]
    fig, axes = plt.subplots(1, len(cfg), figsize=(11.6, 5.9), dpi=140)
    for ax, (name, kw) in zip(np.atleast_1d(axes), cfg):
        env = ComfortGrid(seed=seed)
        rng = random.Random(seed * 31 + 7)
        start = env.random_start(random.Random(seed * 17 + 3))
        ag = HIMAgent(env, rng, variant=name, **kw)
        ag.reset(start)
        xs, ys = [start % env.size], [start // env.size]
        for _ in range(steps):
            p = ag.step()
            xs.append(p % env.size)
            ys.append(p // env.size)
        nvis = len(set(x * 1000 + y for x, y in zip(xs, ys)))
        ax.imshow(env.np_field, origin="lower", cmap="magma",
                  extent=(-0.5, env.size - 0.5, -0.5, env.size - 0.5))
        ax.plot(xs, ys, color="#2ecc71", linewidth=0.6, alpha=0.85)
        ax.scatter([xs[0]], [ys[0]], c="#3498db", s=40, marker="o",
                   edgecolor="white", linewidth=0.8, zorder=5, label="起点")
        ax.set_title(f"{LABEL.get(name, name)}\n"
                     f"{steps} 步覆盖 {nvis} 格（{nvis / env.n_cells:.1%}）",
                     fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(f"轨迹对照（环境种子 {seed}，{steps} 步）", fontsize=12)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig3_trajectory.png")
    fig.savefig(p)
    plt.close(fig)
    print("  写出", p)


def fig4_ablation(payload):
    """以 him 为基准，看每个消融项拿掉之后覆盖率掉多少。"""
    base = "him"
    if base not in payload["agents"]:
        return
    items = [n for n in ["him", "him_nohope", "him_nonav", "him_blocknov",
                         "him_v1"] if n in payload["agents"]]
    cov = [payload["agents"][n]["summary"]["coverage_mean"] * 100 for n in items]
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=140)
    bars = ax.barh([LABEL.get(n, n) for n in items], cov,
                   color=[COLOR.get(n, "#888") for n in items])
    for r, v in zip(bars, cov):
        ax.annotate(f"{v:.2f}%", (v, r.get_y() + r.get_height() / 2),
                    va="center", ha="left", fontsize=9,
                    xytext=(4, 0), textcoords="offset points")
    ax.set_xlabel("环境覆盖率（%）")
    ax.set_title("消融：拿掉哪个算子，探索就塌掉")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, 112)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "fig4_ablation.png")
    fig.savefig(p)
    plt.close(fig)
    print("  写出", p)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    payload = _load()
    print("出图中 …")
    fig1_curve(payload)
    fig2_bar(payload)
    fig4_ablation(payload)
    fig3_trajectory()
    print("完成，全部图在", FIG_DIR)


if __name__ == "__main__":
    main()
