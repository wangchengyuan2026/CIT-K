"""
him_core.grid_env —— 100×100 离散随机网格环境
=================================================
对应论文 §3.1 实验环境设置：

    「实验采用 100×100 离散随机网格环境，环境布局随机生成，
      不设置外部奖励与预设目标。」

    （【3版】§4.1 补充：「随机分布高舒适度区域（兴趣点）」）

设计要点
--------
1. 环境本身**不含任何奖励**。它只提供一个「舒适度场」φ(x, y) ∈ [0, 1]，
   代表该位置的感官满足程度。所有"动机"都来自智能体内部，不来自环境。
2. 兴趣点（POI）以高斯凸包形式随机撒布，形成多峰地形 —— 多峰是必要的，
   否则"局部极小值陷阱"问题不成立。
3. 热循环里不使用 numpy：字段被摊平为 Python list，邻接表预计算，
   保证 6 个智能体 × 100 轮 × 上万步能在可接受时间内跑完。
"""

from __future__ import annotations

import math
import random

import numpy as np

# 动作编号：0=原地停留, 1=上, 2=下, 3=左, 4=右
# 与论文「近路原则」「H 控制 G 在 B 中的运动方向」对应：H 只控制方向，不控制位置。
ACTION_STAY, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3, 4
ACTION_NAMES = ("stay", "up", "down", "left", "right")
N_ACTIONS = 5


def build_neighbor_table(size: int):
    """预计算每个格子的 5 个候选落点（含原地）。越界用 -1 标记。"""
    table = [None] * (size * size)
    for y in range(size):
        for x in range(size):
            i = y * size + x
            table[i] = (
                i,                                              # 停留
                i - size if y > 0 else -1,                      # 上
                i + size if y < size - 1 else -1,               # 下
                i - 1 if x > 0 else -1,                         # 左
                i + 1 if x < size - 1 else -1,                  # 右
            )
    return table


class ComfortGrid:
    """舒适度场环境。"""

    def __init__(
        self,
        size: int = 100,
        n_poi: int = 20,
        poi_sigma: float = 5.0,
        poi_amp: float = 0.9,
        poi_threshold_frac: float = 0.5,
        background: float = 0.02,
        obs_noise: float = 0.05,
        seed: int = 0,
    ):
        self.size = size
        self.n_cells = size * size
        self.n_poi = n_poi
        self.obs_noise = obs_noise
        self.seed = seed

        rng = np.random.default_rng(seed)

        # ---- 兴趣点位置（均匀随机，不设外部目标）----
        self.poi_xy = [
            (int(rng.integers(0, size)), int(rng.integers(0, size))) for _ in range(n_poi)
        ]

        # ---- 舒适度场：多峰高斯叠加 ----
        field = np.full((size, size), float(background), dtype=np.float64)
        rad = int(3.0 * poi_sigma)
        two_sig2 = 2.0 * poi_sigma * poi_sigma
        for (px, py) in self.poi_xy:
            x0, x1 = max(0, px - rad), min(size, px + rad + 1)
            y0, y1 = max(0, py - rad), min(size, py + rad + 1)
            # field 以 [y, x] 索引：yy 为行偏移，xx 为列偏移
            yy = np.arange(y0, y1)[:, None] - py
            xx = np.arange(x0, x1)[None, :] - px
            patch = poi_amp * np.exp(-(yy * yy + xx * xx) / two_sig2)
            # 用 maximum 而不是求和：求和在兴趣点密集处会叠加到 1.0 以上并被
            # clip 出一大片等值平台，梯度消失（ΔL ≡ 0），智能体会永远困在
            # 平台上随机游走。maximum 保证场严格 = 单峰高斯，处处有梯度。
            np.maximum(field[y0:y1, x0:x1], patch, out=field[y0:y1, x0:x1])
        # 注意：poi_amp 必须足够小，使叠加后不触顶。一旦触顶就会形成大片
        # φ = 1.0 的等值平台，梯度消失、ΔL ≡ 0，智能体会在平台上随机游走
        # 而无法离开 —— 这是第一版实验里 HIM 覆盖率仅 1.3% 的直接原因。
        np.clip(field, 0.0, 1.0, out=field)

        self.np_field = field
        self.field = field.ravel().tolist()          # 热循环用
        self.prior_mean = float(field.mean())        # 供 FEP 基线当作世界先验
        self.prior_std = float(field.std())
        # 「兴趣区」掩码：φ ≥ poi_threshold_frac × poi_amp 的格子。
        # 仅用于统计「发现了多少兴趣点」，不作为任何智能体的奖励来源。
        thr = poi_threshold_frac * poi_amp
        self.poi_threshold = thr
        self.poi_cells = int((field >= thr).sum())
        self.poi_flags = (field.ravel() >= thr).tolist()

        self.nbr = build_neighbor_table(size)
        self._grng = random.Random(seed * 7919 + 13)

    # ------------------------------------------------------------------
    def sense(self, idx: int) -> float:
        """带观测噪声的感官读数。噪声对智能体不可见，也不参与前瞻。"""
        return self.field[idx] + self._grng.gauss(0.0, self.obs_noise)

    def sense_clean(self, idx: int) -> float:
        """无噪声场值。仅用于前瞻/规划（智能体对候选格子的"预期感官"）。"""
        return self.field[idx]

    def random_start(self, rng: random.Random) -> int:
        return rng.randrange(self.n_cells)

    def describe(self) -> str:
        return (
            f"ComfortGrid(size={self.size}×{self.size}, POI={self.n_poi}, "
            f"兴趣区格子={self.poi_cells} ({self.poi_cells / self.n_cells:.1%}), "
            f"φ均值={self.prior_mean:.4f}, φ标准差={self.prior_std:.4f})"
        )


__all__ = [
    "ComfortGrid",
    "build_neighbor_table",
    "ACTION_STAY",
    "ACTION_UP",
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_NAMES",
    "N_ACTIONS",
]
