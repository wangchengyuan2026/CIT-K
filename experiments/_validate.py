"""稳健性验证：多随机种子 × 两个步数预算，确认 field 新奇度不是幸运配置。"""
import random
import statistics
import sys
import time

sys.path.insert(0, ".")

from experiments.metrics import run_episode           # noqa: E402
from him_core.grid_env import ComfortGrid             # noqa: E402
from him_core.motivation_engine import HIMAgent       # noqa: E402

SEED0, N_RUNS = 1000, 20
print(f"{'mode':>10}{'cw':>5}{'cap':>5}{'steps':>7}{'cov_mean':>10}"
      f"{'cov_min':>9}{'poi':>8}{'moves':>8}{'sec':>7}")
for mode, cw, cap, steps in [
    ("block", 0.20, 48, 12000),
    ("field", 0.20, 1, 12000),      # 消融：无导航
    ("field", 0.20, 48, 12000),
    ("field", 0.20, 96, 12000),
    ("field", 0.20, 128, 12000),
    ("field", 0.20, 96, 20000),
]:
    cov, mv, poi = [], [], []
    t0 = time.time()
    for s in range(SEED0, SEED0 + N_RUNS):
        env = ComfortGrid(seed=s)
        ag = HIMAgent(env, random.Random(s * 31 + 7), curiosity_weight=cw,
                      novelty_mode=mode, near_bonus=1.0, far_cap=cap,
                      far_interval=10)
        r = run_episode(ag, env, seed=s, max_steps=steps,
                        rng=random.Random(s * 17 + 3), keep_hist=False)
        cov.append(r["coverage"])
        mv.append(r["moves"])
        poi.append(r["poi_cover_ratio"])
    print(f"{mode:>10}{cw:>5.2f}{cap:>5}{steps:>7}{statistics.mean(cov):>10.2%}"
          f"{min(cov):>9.2%}{statistics.mean(poi):>8.1%}"
          f"{statistics.mean(mv):>8.0f}{time.time() - t0:>7.1f}", flush=True)
