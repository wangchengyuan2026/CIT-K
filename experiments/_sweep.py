"""快速测：field 模式（距离势场新奇度）参数扫描。"""
import random
import sys
import time

sys.path.insert(0, ".")

from experiments.metrics import run_episode           # noqa: E402
from him_core.grid_env import ComfortGrid             # noqa: E402
from him_core.motivation_engine import HIMAgent       # noqa: E402

N_RUNS = 6
STEPS = 20000

print(f"{'mode':>8}{'cw':>6}{'nb':>5}{'cap':>5}{'itv':>5}{'la':>4}"
      f"{'coverage':>11}{'moves':>8}{'poi':>8}{'sec':>7}")
for mode, cw, nb, cap, itv, la in [
    ("block", 0.20, 1.0, 48, 10, 1),
    ("field", 0.05, 1.0, 48, 10, 1),
    ("field", 0.10, 1.0, 48, 10, 1),
    ("field", 0.20, 1.0, 48, 10, 1),
    ("field", 0.50, 1.0, 48, 10, 1),
    ("field", 0.20, 0.0, 48, 10, 1),
    ("field", 0.20, 3.0, 48, 10, 1),
    ("field", 0.20, 1.0, 24, 10, 1),
    ("field", 0.20, 1.0, 96, 10, 1),
    ("field", 0.20, 1.0, 48, 20, 1),
    ("field", 0.20, 1.0, 48, 10, 3),
]:
    cov, mv, poi = [], [], []
    t0 = time.time()
    for s in range(1000, 1000 + N_RUNS):
        env = ComfortGrid(seed=s)
        ag = HIMAgent(env, random.Random(s * 31 + 7), curiosity_weight=cw,
                      novelty_mode=mode, near_bonus=nb, far_cap=cap,
                      far_interval=itv, lookahead=la)
        r = run_episode(ag, env, seed=s, max_steps=STEPS,
                        rng=random.Random(s * 17 + 3), keep_hist=False)
        cov.append(r["coverage"])
        mv.append(r["moves"])
        poi.append(r["poi_cover_ratio"])
    print(f"{mode:>8}{cw:>6.2f}{nb:>5.1f}{cap:>5.0f}{itv:>5}{la:>4}"
          f"{sum(cov) / len(cov):>10.2%}{sum(mv) / len(mv):>8.0f}"
          f"{sum(poi) / len(poi):>8.1%}{time.time() - t0:>7.1f}")
