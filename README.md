# TuRBO — Independent Reproduction

A from-scratch, citation-anchored reimplementation of **TuRBO** (Trust Region Bayesian Optimization),
built by reading the paper rather than porting the authors' code.

> **This is not the authors' repository.** It is an independent, unofficial reproduction, not
> affiliated with or endorsed by the authors or Uber. The official implementation lives at
> [uber-research/TuRBO](https://github.com/uber-research/TuRBO). This project consulted that code to
> resolve details the paper leaves unspecified — every such case is tagged `[FROM_OFFICIAL_CODE]` in
> the source and listed in [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) — but no code was copied.

## Paper

> David Eriksson, Michael Pearce, Jacob R. Gardner, Ryan Turner, Matthias Poloczek.
> **Scalable Global Optimization via Local Bayesian Optimization.**
> *Advances in Neural Information Processing Systems 32 (NeurIPS 2019).*
> [arXiv:1910.01739](https://arxiv.org/abs/1910.01739)

**Start here before trusting anything below:**
[REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) — every deviation and undetermined choice ·
[VERIFICATION.md](VERIFICATION.md) — what was checked, what was not, and the audit findings.

---

## What this repository implements

- **TuRBO-1** — a single trust region with restarts (§2).
- **TuRBO-m** — `m` trust regions with the implicit multi-armed bandit across them (§2).
- **Local GP surrogate** — Matérn-5/2 with ARD, constant mean, hyperparameter box constraints from
  App. C, fitted by marginal likelihood before every batch.
- **Trust-region schedule** — lengthscale-anisotropic box, success/failure resizing, restart on
  collapse (§2, App. D).
- **Thompson sampling** — scrambled Sobol candidate sets with coordinate perturbation (App. D), joint
  posterior draws, and the cross-region argmin that allocates the batch (§2, App. E).
- **Synthetic benchmarks** — Ackley, Levy, Rastrigin (10D), Hartmann6, Ackley-200, on the App. A domains.
- **Baselines** — random search, Nelder-Mead, BFGS (SciPy); CMA-ES and BOBYQA behind optional deps.
- **Evaluation harness** — replication driver, best-so-far traces, mean ± one standard error (§3).

## What this repository does NOT implement

| Not implemented | Why |
|---|---|
| **Robot pushing (§3.1), rover trajectory (§3.2), lunar lander (§3.4)** | The reward functions are given (App. F) but the simulators are external (Wang et al. 2018; OpenAI Gym). Lunar lander additionally needs the paper's "fixed constant set of 50 randomly generated terrains", which is unpublished. |
| **Cosmological constants (§3.3)** | Not reproducible from the paper: App. F.3 names "three additional parameters chosen from the many available" without saying which, and §3.3 says "substantially larger parameter bounds" without giving any. |
| **EBO, BOCK, BOHAMIANN, HeSBO-TS baselines** | Third-party methods that App. B says required modification to run in this setting. Not part of the contribution. |
| **§3.6 local-vs-global GP ablation, §3.7 trust-region statistics, §3.8 batch-size scaling** | Not implemented. §3.6 is the paper's central mechanistic evidence and would be the highest-value addition. |
| **GPU execution** | CPU only. App. G ran on an NVIDIA RTX 2080 TI. |

`src/benchmarks.py` declares the unavailable benchmarks and raises `NotImplementedError` naming the
specific reason, rather than silently substituting a stand-in objective.

## Algorithm overview

TuRBO's thesis is that Bayesian optimization fails in high dimensions with large budgets for two
reasons: a global GP implicitly assumes constant lengthscales and signal variance across the space,
and global acquisition over-explores regions of high posterior uncertainty. Its answer is to abandon
the global surrogate and fit **local GPs inside trust regions**, then allocate a batch across those
regions with an implicit bandit.

One batch, in the order the code executes it (`src/local_model.py`):

```
standardize values ──▶ fit local GP ──▶ pick centre x*  ──▶ derive TR box from the
                       (App. C)         (best point, or   CURRENT fitted lengthscales
                                         smallest posterior  L_i = λ_i·L / (Π λ_j)^(1/d)
                                         mean under noise)
                                                    │
        ┌───────────────────────────────────────────┘
        ▼
  scrambled Sobol candidates inside the box ──▶ q joint posterior draws ──▶ select
  (perturb each coord w.p. min{1, 20/d})         (de-standardized)          argmin
        │
        ▼
  evaluate ──▶ update success/failure counters against the pre-batch incumbent
               (double L after τ_succ successes, halve after τ_fail failures,
                restart the region when L < L_min)
```

For TuRBO-m the same per-region step runs for every region, and the batch is selected by a **joint**
argmin over (region, candidate) — the line that makes it a bandit rather than `m` independent runs.

Three orderings are load-bearing and fail *silently* if swapped: the GP must be refit before the box
is derived; counters must update before the batch is appended; and draws must be de-standardized
before regions are compared. See `PAPER_SPEC.md` §5.

## Repository structure

```
├── src/
│   ├── config.py         # every numeric constant, grouped by provenance
│   ├── turbo_1.py        # TuRBO-1: single region + restarts
│   ├── turbo_m.py        # TuRBO-m: m regions + cross-region bandit
│   ├── local_model.py    # the per-region surrogate step shared by both
│   ├── gp.py             # Matérn-5/2 ARD GP, marginal-likelihood fitting
│   ├── trust_region.py   # box geometry, centring, success/failure schedule
│   ├── candidates.py     # scrambled Sobol candidate sets + perturbation mask
│   ├── thompson.py       # posterior draws and batch selection
│   ├── benchmarks.py     # synthetic objectives; unavailable ones raise
│   ├── baselines.py      # random search, NM, BFGS, CMA-ES, BOBYQA
│   ├── evaluate.py       # replication driver, traces, mean ± 1 SE
│   └── utils.py          # domain rescaling, LHD, standardization
├── tests/                # 193 tests, mutation-tested
├── scripts/
│   └── reproduce_minimal.py
├── results/              # committed evidence for the verdict below
├── PAPER_SPEC.md         # the implementation contract
├── REPRODUCTION_NOTES.md # deviations and assumptions
├── VERIFICATION.md       # audit: what is and isn't verified
├── PERFORMANCE.md        # profiling; why no optimization was applied
└── CLEAN_SETUP_CHECKLIST.md
```

## Installation

Python ≥ 3.10. Install the CPU build of PyTorch first so pip does not pull a multi-gigabyte CUDA
build — nothing here requests a GPU.

```bash
git clone <repository-url> paper-reproduction
cd paper-reproduction
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Full walkthrough, optional extras and troubleshooting: [CLEAN_SETUP_CHECKLIST.md](CLEAN_SETUP_CHECKLIST.md).

## 60-second quick start

```bash
pytest tests/test_smoke.py -q
```

Expected: `15 passed` in about 20 seconds. This walks the minimal path in stages — fixture → GP
forward pass → marginal likelihood → one optimizer step → one full TuRBO iteration → complete
`Turbo1` and `TurboM` runs — so a broken install fails at the earliest stage instead of opaquely.

## Minimal example

```python
import numpy as np
from src import Turbo1
from src.benchmarks import get_benchmark

bench = get_benchmark("ackley10")          # App. A domain: [-5, 10]^10

opt = Turbo1(
    f=bench,                               # minimizes; wrap rewards with benchmarks.negate
    lb=bench.lb, ub=bench.ub,
    n_init=20,                             # App. A: 20 initial LHD points
    max_evals=500,                         # App. A: 500 evaluations total
    batch_size=10,                         # App. A: q = 10
    seed=0,                                # a run is reproducible from this seed
).optimize()

print(f"best value {opt.best_value:.4f} after {opt.n_evals} evaluations")
print(f"best point  {np.round(opt.best_point, 3)}")
```

TuRBO-m is the same call with `n_trust_regions`, and `n_init` counted **per region**:

```python
from src import TurboM
opt = TurboM(f=bench, lb=bench.lb, ub=bench.ub,
             n_init=10, max_evals=500, n_trust_regions=5, batch_size=10, seed=0).optimize()
```

## Training

**Not applicable — there is no training phase.** TuRBO is a black-box optimizer, not a learned model.
The only quantity minimized by gradient descent is the GP's negative log marginal likelihood, refit
inside the optimization loop before every batch (App. C), which `Turbo1.optimize()` handles. There is
no dataset, no checkpoint, and no separate training command.

## Evaluation / reproduction

```bash
python scripts/reproduce_minimal.py                    # App. A protocol, ~10 min CPU
python scripts/reproduce_minimal.py --replications 5   # ~2 min; labelled as a deviation
```

Runs Appendix A's pinned protocol on Ackley-10 (500 evaluations, `q=10`, 20 initial LHD points, 30
replications) for TuRBO-1 against random search, Nelder-Mead and BFGS. Logs the environment, seed
range and pinned config; writes per-replication metrics and best-so-far traces to `results/`; and
labels the outcome **MATCH / PARTIAL MATCH / NOT REPRODUCED**.

## Reproduction status

| Paper result | Status | Evidence |
|---|---|---|
| App. A / Fig. 8 — TuRBO "consistently find[s] solutions close to the global optimum" on Ackley, and outperforms the other methods | **MATCH** (consistent with the claim) | `results/`, §Results below |
| App. A — Levy, Rastrigin, Hartmann6 | **NOT RUN** | implemented and testable; simply not executed |
| §3.5 — 200D Ackley, 10 000 evaluations | **NOT RUN** | ~10 min/replication on a GPU per App. G; no GPU here |
| §3.1 robot pushing · §3.2 rover · §3.4 lunar lander | **NOT REPRODUCIBLE** | external simulators / unpublished seeds |
| §3.3 cosmological constants | **NOT REPRODUCIBLE** | paper omits the parameters and the bounds |
| §3.6 local vs global GP (1.284 → 1.174 nats) | **NOT REPRODUCIBLE** | depends on the robot-pushing objective |
| §3.7 trust-region statistics · §3.8 batch-size scaling | **NOT IMPLEMENTED** | depend on rover / robot pushing |
| App. G runtimes | **NOT COMPARABLE** | different hardware (CPU vs RTX 2080 TI) |

**Every precise number the paper reports** — robot pushing's 9.4, rover's ~2, §3.6's 1.284 vs 1.174
nats — **depends on a benchmark that cannot be reproduced from the paper.** No number in this
repository is a match to a published number, because for the parts that *are* reproducible the paper
publishes plots, not numbers.

## Results

Appendix A protocol on **Ackley-10**: 500 evaluations, `q=10`, 20 initial LHD points, 30
replications, seeds 0–29, CPU float64. Final best value, mean ± one standard error:

| Optimizer | Mean | ± 1 SE | Median | Best | Worst |
|---|---:|---:|---:|---:|---:|
| **TuRBO-1** | **0.4587** | 0.0457 | 0.3664 | 0.1602 | 1.3498 |
| BFGS | 5.9916 | 0.3571 | 6.0560 | 2.0133 | 9.6460 |
| Random search | 8.8455 | 0.1587 | 8.8888 | 6.6398 | 10.2274 |
| Nelder-Mead | 9.1224 | 0.2682 | 9.1634 | 6.4742 | 11.7561 |
| *(global optimum)* | *0.0000* | | | | |

**Outcome: MATCH — consistent with the paper's claims.** TuRBO-1 lands within 0.46 of the global
optimum and beats every baseline by far more than one standard error, which is what App. A describes.

**What this does not mean.** App. A reports Figure 8 as a *plot* and states no numeric value for
Ackley-10, so nothing above is matched against a published number. The "close to the global optimum"
threshold used by the verdict logic is this repository's operationalization of the paper's wording,
declared in `scripts/reproduce_minimal.py::CONFIG` so it can be argued with. Baselines are limited to
three of the paper's ten. Raw per-replication values and traces are in `results/`.

## Known deviations and assumptions

Full detail in [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md). The headline: **the paper and the
official implementation disagree in six places.** Where they conflict, this repository follows the
**paper**, which means it will *not* produce trajectories identical to uber-research/TuRBO.

| Item | Paper | Official code | Here |
|---|---|---|---|
| `τ_fail` (TuRBO-1) | `⌈d/q⌉` | `⌈max(4/q, d/q)⌉` | **paper** |
| `τ_fail` (TuRBO-m) | sequential case → `d` | `max(5, d)` | **paper** |
| Noise variance bound | `[5e-4, 0.1]` | `[5e-4, 0.2]` | **paper** |
| "Success" definition | any improvement | requires a `1e-3·|f_best|` margin | code default, `success_tol=0.0` restores the paper |
| Value standardization | "standardized" (implies mean) | centres on the **median** | code default, `center_stat="mean"` available |
| Thompson draws | latent posterior | predictive (adds σ² noise) | code default, `use_predictive=False` available |
| TR centre under noise | smallest posterior mean | never implemented | **both**, via `noisy=True` |

Choices the paper does not determine (Adam lr 0.1, 50 steps, GP initialization, candidate
de-duplication, RNG policy, …) are tagged in the source with what the paper omits, the choice made,
and the alternatives. An upstream off-by-one in the reference implementation's perturbation-mask
repair is deliberately **not** reproduced.

## Tests

```bash
pytest tests/ -q                                              # 193 tests, ~20 s
pytest tests/test_trust_region.py -q                          # one module
pytest tests/test_trust_region.py::test_volume_invariant -q   # a single test
```

Tests are **mutation-tested**: 22 known-wrong implementations were injected into a copy of `src/` and
the suite re-run — all 22 are caught. That sweep found two real gaps (the App. D constants were pinned
by nothing, and one test asserted `TR_DEFAULTS` against `TR_DEFAULTS`), both now closed.

An independent adversarial review of the whole repository is recorded in
[VERIFICATION.md](VERIFICATION.md) §H. It confirmed the core algorithm correct and found seven defects
— all in the *baseline comparison* path, including baselines silently consuming up to 2.4× their
stated evaluation budget. All are fixed; the budget bug is why the BFGS column above is honest.

## Limitations

- **Only the synthetic benchmarks are reproducible from this repository**, and only one of them has
  actually been run (§Results). The paper's headline experiments are not reachable.
- **CPU only.** Runtime is dominated (85%) by GP fitting, which App. C mandates before every batch;
  see [PERFORMANCE.md](PERFORMANCE.md) for why no optimization was applied.
- **Cross-machine numbers will differ slightly.** Runs are reproducible from a seed on a fixed
  machine, but different BLAS builds and thread counts reorder float64 reductions inside the GP fit.
  Statistics should agree within the reported standard error; identical trajectories should not be
  expected.
- **The GPyTorch CG/Lanczos path has never been exercised** — every run so far stays under
  `max_cholesky_size = 2000`, so the scalability claim behind §3.1–3.2 is untested here
  (`VERIFICATION.md` U2).
- **No claim is made about matching the paper's plots.** Only the two quoted claims in
  `scripts/reproduce_minimal.py` were checked, on one function.

## Citation

Cite the paper, not this repository:

```bibtex
@inproceedings{eriksson2019turbo,
  title     = {Scalable Global Optimization via Local Bayesian Optimization},
  author    = {Eriksson, David and Pearce, Michael and Gardner, Jacob R.
               and Turner, Ryan and Poloczek, Matthias},
  booktitle = {Advances in Neural Information Processing Systems 32},
  year      = {2019},
  url       = {https://arxiv.org/abs/1910.01739}
}
```

If you refer to this reproduction specifically, please describe it as an independent reimplementation
and link to the paper above — and note which of its results were actually run (§Reproduction status).

## License

**No license file is present**, so default copyright applies and no permissions are granted. If you
intend to publish or share this repository, add a license explicitly.

Note for anyone considering relicensing: the official TuRBO implementation is distributed under the
**Uber Non-Commercial License**. This code was written from the paper and no source was copied from
it, but it was consulted to resolve unspecified details (each tagged `[FROM_OFFICIAL_CODE]`). Review
that yourself before choosing a license.
