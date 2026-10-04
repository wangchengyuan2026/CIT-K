"""环境统计：舒适度场 φ 的均值/分位数（用于把"末段 φ"放到正确参照系里）。"""
import sys

sys.path.insert(0, ".")

import numpy as np                                   # noqa: E402
from him_core.grid_env import ComfortGrid            # noqa: E402

means, meds, p90 = [], [], []
for s in range(1000, 1060):
    env = ComfortGrid(seed=s)
    f = env.np_field
    means.append(f.mean())
    meds.append(float(np.median(f)))
    p90.append(float(np.percentile(f, 90)))
print(f"φ 全场均值   = {np.mean(means):.4f}  (std {np.std(means):.4f})")
print(f"φ 中位数     = {np.mean(meds):.4f}")
print(f"φ 90 分位    = {np.mean(p90):.4f}")
env = ComfortGrid(seed=1000)
print(f"φ 最大/最小  = {env.np_field.max():.4f} / {env.np_field.min():.4f}")
print(f"兴趣区格子占比 = {env.poi_cells / env.n_cells:.2%}  (阈值 {env.poi_threshold:.3f})")
