"""诊断：卡住时智能体到底处在什么状态。"""
import math
import random
import sys

sys.path.insert(0, ".")

from him_core.grid_env import ComfortGrid             # noqa: E402
from him_core.motivation_engine import HIMAgent       # noqa: E402


def run(cw, cap, itv, steps=6000, seed=1000):
    env = ComfortGrid(seed=seed)
    ag = HIMAgent(env, random.Random(seed * 31 + 7), curiosity_weight=cw,
                  novelty_mode="field", near_bonus=1.0, far_weight=1.0,
                  far_cap=cap, far_interval=itv)
    ag.reset(env.random_start(random.Random(seed * 17 + 3)))
    trapped = 0
    for t in range(steps):
        nb = env.nbr[ag.pos]
        if not any(nb[a] >= 0 and not ag.vflat[nb[a]] for a in range(1, 5)):
            trapped += 1
        ag.step()
    h = ag.homeo
    beta = min(1.0, h.beta_max * (1 - math.exp(-h.t / h.beta_tau)))
    nvis = int(ag.vflat.sum())
    print(f"cw={cw} cap={cap} itv={itv}: 覆盖={nvis / env.n_cells:.2%} "
          f"moves={ag.moves} 四邻全访问步数={trapped}/{steps} "
          f"pos={ag.pos}(x={ag.pos % 100},y={ag.pos // 100})")
    print(f"   d[{ag.pos}]={ag.dflat[ag.pos]:.1f}  d.max={ag.d2.max():.1f}  "
          f"d==0 格数={int((ag.d2 == 0).sum())}  未访问={ag._n_unvisited}")
    print(f"   L_prev={h.L_prev:.4f} X={h.predictive_expectation():.4f} "
          f"beta={beta:.3f} peak={h.hope_peak:.4f}")
    for a in range(5):
        j = env.nbr[ag.pos][a]
        if j < 0:
            continue
        v = ag._value(j, h.memory, beta, h.L_prev, h.hope_peak, True,
                      h.hope_threshold, h.hope_gain, cw, 0.0)
        print(f"   act={a} j={j:>5} V={ag.vflat[j]} phi={env.field[j]:.4f} "
              f"d={ag.dflat[j]:.1f} nov={ag._curiosity(j, ag.steps):.5f} "
              f"value={v:+.5f}")
    print()


run(1.0, 96, 20)
run(1.0, 32, 10)
run(0.5, 16, 10)
run(5.0, 96, 20)
