# HIM-Theory — executable reference implementation of *Homeostatic Intrinsic Motivation* (Cat Intelligence)

This repository turns the theory contained in three documents by the same author
(**Chengyuan Wang / 王程远**) into an **experiment you can re-run and re-check**.

| Document | Role | Where it lives here |
|---|---|---|
| *Reflective Intelligence: Design of a Minimal Machine Model with Autonomous Motivation* (2018, "version A") | The theoretical root: a discrete state machine over 17 primitive bodies | `him_core/state_machine.py` (the `ci2018` agent) |
| *Homeostatic Intrinsic Motivation (HIM): An Autonomous Agent Architecture Based on Homeostatic Drives* ("version C") | Makes version A continuous: `L_t` and `dL_t` | `him_core/homeostasis.py` |
| *[Version 3] An Autonomous Agent Architecture Based on Homeostatic Drives* | Adds the three equations and the "three bridges" | `him_core/homeostasis.py` + `docs/results_report.md` |

**Headline result, up front.** The pseudocode in the papers — `drive = dL_t + hope_drive` —
cannot, mathematically, produce the exploratory behaviour the papers claim for it. Transcribed
literally it yields **0.45% environmental coverage**: the agent climbs the first comfort peak it
finds and then stops, moving a total of **93 steps** out of 12,000. The papers report **88.2%**.
Adding one operator that the papers never specify — a computable form for "regional novelty" —
raises coverage to **99.91%**, with **0%** of runs trapped in a local optimum and **99.99%**
of comfort-interest regions discovered. Full argument and every point of divergence:
[`docs/results_report.md`](docs/results_report.md) (in Chinese).

---

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate     # Windows
# source .venv/bin/activate                          # Linux / macOS
pip install -r requirements.txt

# 1) Full comparative experiment (9 agents x N runs), roughly 2-5 minutes
python experiments/run_exp.py --runs 60 --steps 12000

# 2) Figures
python experiments/plot_results.py

# 3) Inspect HIM's trajectory and internal state step by step
python examples/_probe.py
```

Outputs:

```
results/exp_results.json     all aggregate metrics + per-run detail (machine readable)
results/summary.txt          a readable table in the shape of the papers' Table 1
results/figures/*.png        coverage curves / bar comparison / trajectories / ablation
```

Deterministic: environment seed `1000+i`, agent seed `s*31+7`, start-position seed `s*17+3`.
Results are identical across machines.

---

## Layout

```
HIM-Theory/
├── him_core/
│   ├── grid_env.py            100x100 grid; exposes only the comfort field phi, no reward
│   ├── homeostasis.py         Eq1/Eq2/Eq3 + bridge 1 (mapping) + bridge 2 (difference) + hope
│   ├── motivation_engine.py   the HIM agent (action selection, regional novelty, lookahead)
│   └── state_machine.py       [version A, 2018] discrete 17-primitive-body state machine
├── baseline/
│   ├── fep_model.py           standard active inference (minimises expected free energy)
│   └── rl_baseline.py         tabular Q-learning with an external sparse reward; random walk
├── experiments/
│   ├── metrics.py             explicit definitions of all metrics (the papers give none)
│   ├── run_exp.py             one-command batch experiment
│   ├── plot_results.py        figures
│   ├── _sweep.py              parameter sweeps (novelty-operator comparison)
│   └── _validate.py           multi-seed robustness validation
├── examples/_probe.py         single-step drive decomposition probe (for stall diagnosis)
├── docs/results_report.md     results report: itemised divergence from the papers
└── results/                   experiment artefacts
```

---

## The environment (aligned with Section 3.1 of the papers)

| Item | Value | Source |
|---|---|---|
| Grid | 100 x 100, discrete | stated in Sec. 3.1 |
| External reward | **none** | stated in Sec. 3.1 |
| Preset goal | **none** | stated in Sec. 3.1 |
| Layout | randomly generated per seed | stated in Sec. 3.1 |
| Comfort field `phi` | 20 random Gaussian interest points, `phi` in [0.02, 0.9] | version 3, Sec. 4.1 ("randomly distributed high-comfort regions") |
| Observation noise | `sigma = 0.05` (invisible to the agent; lookahead is noise-free) | not given in the papers; registered explicitly here |

`phi` is combined with `np.maximum` rather than summation. Summing saturates where interest points
are dense and `clip` then produces a large plateau of `phi = 1.0`: the gradient vanishes, `dL = 0`,
and the agent random-walks on the plateau. This was the direct cause of 1.3% coverage in the
first implementation.

---

## The nine agents

| Name | One-line description | Needs external reward? |
|---|---|---|
| `him` | full model: `dL` + hope + **regional novelty** | no |
| `him_v1` | literal transcription of the paper's pseudocode, single-step greedy | no |
| `him_nohope` | ablation: hope mechanism removed | no |
| `him_nonav` | ablation: long-range navigation term removed from novelty | no |
| `him_blocknov` | ablation: novelty uses the papers' "last-visit timestamp + bilinear interpolation" | no |
| `ci2018` | version A (2018) discrete state machine | no |
| `fep` | standard active inference, minimal expected free energy | no |
| `rl` | tabular Q-learning | **yes** (+1 on entering an interest region) |
| `random` | uniform random walk (scale reference, not a baseline) | no |

---

## Metric definitions

Sections 3.1 and 4 of the papers use four metrics but define **none** of them. This repository
fixes the definitions in `experiments/metrics.py` and flags every place where a definition may
not match the papers' wording:

| Metric | Definition used here |
|---|---|
| Environmental coverage | visited cells / 100^2 |
| Convergence step | first step at which, over a sliding window of 500 steps, fewer than 10 new cells are added |
| Trapped in a local optimum | converged **and** coverage at convergence < 50% |
| Energy | three numbers reported together: total steps / actual moving steps / steps per covered cell |
| Interest-region coverage | visited interest-region cells / all interest-region cells (a quantifiable proxy for version A's "intelligence test") |

---

## Regional novelty: the operator this repository had to supply

The papers treat "regional novelty / curiosity" as a core axiom (version C, Sec. 6.3) and argue in
version 3, Sec. 5.2 that "`dL` approximately zero produces an internal pressure to actively try new
behaviour" — but they **never give any computable form for it**. The equation
`drive = dL + hope_drive` is identically zero wherever `dL = 0`, so `argmax` always selects
"stay put".

This repository implements and compares two forms:

```
block  (literal translation of the papers' "last-visit timestamp")
    novelty(j) = clip((now - last_visit(block(j))) / horizon, 0, 1)     ->  0.17%

field  (supplied by this repository; the default)
    novelty(j) = cw * [ NB*D*(1 - V[j])  +  (D - d(j)) ]
        V[j] = whether j has been visited
        d(j) = Manhattan distance from j to the nearest unvisited cell (capped at D = 96)
                                                                       ->  99.91%
```

The two terms of `field` play roles that cannot replace each other:

* `NB*D*(1 - V[j])` — the **target signal**. An unvisited cell always outranks a visited one, so
  `stay` is always rejected: the agent cannot stall. This is precisely the "stall pressure" that
  version 3, Sec. 5.2 wants but never writes into an equation.
* `D - d(j)` — the **navigation signal**. An exact per-cell Manhattan distance potential that
  increases by exactly `cw` on every step (a constant gradient, an order of magnitude larger than
  `dL`), and whose maximum can only occur at an unvisited cell. It therefore cannot create a
  spurious local maximum inside already-explored territory. With it, the agent can walk back to the
  frontier even when it has walled itself in.

Both terms are backed by ablation data (see the `curve` and `ablation` figures and
`docs/results_report.md`, Sec. 3).

---

## Correct interpretation of the baselines

* `random` and `rl` serve as **scale references**: they indicate how much of a 100x100 grid a
  naive policy covers in 12,000 steps, so that any coverage figure can be judged against
  "did this agent actually do anything".
* The papers publish no experiment code, no random seeds, no hyperparameters and no metric
  definitions. This repository is therefore **a faithful reconstruction from the papers' text,
  not a reproduction of the original experiments**. Everything that could not be derived from the
  text is registered item by item in `docs/results_report.md`.

---

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this code or the HIM theory, please cite the
2018 article and, once available, the HIM preprint.
