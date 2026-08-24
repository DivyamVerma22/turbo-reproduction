# REPRODUCTION_NOTES.md

Deviations from the paper, and every decision the paper did not determine.
Required by rule 12 (CLAUDE.md). The full evidence table is `PAPER_SPEC.md` §10; this file
records **what the code actually does** and why.

**Paper:** Eriksson et al., *Scalable Global Optimization via Local Bayesian Optimization*,
NeurIPS 2019 ([arXiv 1910.01739](https://arxiv.org/abs/1910.01739)).
**Reference implementation consulted:** https://github.com/uber-research/TuRBO @ `master`.

Status legend: `SPECIFIED` · `PARTIALLY_SPECIFIED` · `UNSPECIFIED` · `ASSUMPTION` ·
`FROM_OFFICIAL_CODE`. The official code is **not** the paper; anything traced to it is
`FROM_OFFICIAL_CODE`, never `SPECIFIED`.

---

## 1. Deviations from the paper as written

These are places where this implementation does something the paper does not literally say.
Each is switchable — the paper-literal behavior is reachable by flag.

| # | What the paper says | What this code does by default | Flag for paper-literal behavior | Status |
|---|---|---|---|---|
| D1 | §2: a success is "a candidate that improves upon x\*" — no tolerance | Requires improvement by `1e-3 · abs(f_best)` | `success_tol=0.0` | PARTIALLY_SPECIFIED → FROM_OFFICIAL_CODE |
| D2 | App. C: "the function values are standardized" (conventionally mean/std) | Centers on the **median** | `center_stat="mean"` | PARTIALLY_SPECIFIED → FROM_OFFICIAL_CODE |
| D3 | §2: `f_l^(i) ~ GP_l(μ_l, k_l)` — the **latent** posterior | Samples the **predictive** distribution (adds σ² noise) | `use_predictive=False` | PARTIALLY_SPECIFIED → FROM_OFFICIAL_CODE |
| D4 | §2: TR is "a hyperrectangle centered at x\*" with `Π L_i = L^d` | Bounds are clipped to `[0,1]^d`, which **breaks the volume invariant near the boundary** | none — App. D requires the intersection with the domain | PARTIALLY_SPECIFIED |

**Note on D4.** The paper states the constant-volume property and separately requires
candidates to lie in "the intersection of the TR and the domain `[0,1]^d`" (App. D). These
two requirements conflict whenever the trust region overlaps a domain face. Neither the
paper nor the official code addresses it; the official code clips, and so do we.
`tests/test_trust_region.py::test_volume_invariant` therefore asserts the invariant on the
**unclipped** weights only.

## 2. Where this code follows the paper and the official code does not

| # | Item | Paper | Official code | This code |
|---|---|---|---|---|
| P1 | `τ_fail`, TuRBO-1 | `⌈d/q⌉` (App. D) | `⌈max(4/q, d/q)⌉` (`turbo_1.py` L106) | **paper** |
| P2 | `τ_fail`, TuRBO-m | "the same tolerances as in the sequential case (q=1)" → `d` (App. D) | `max(5, d)` (`turbo_m.py` L86) | **paper** |
| P3 | Noise-variance upper bound | `σ² ∈ [0.0005, 0.1]` (App. C) | `Interval(5e-4, 0.2)` (`gp.py` L48) | **paper** (0.1) |
| P4 | TR center under noise | "the observation with the smallest posterior mean" (§2) | never implemented; always `argmin(fX)`, with a comment admitting the gap (`turbo_1.py` L152-154) | **both**, via `noisy=True` |

P1–P3 mean this implementation will **not** produce bit-identical trajectories to
uber-research/TuRBO. That is deliberate: rule 2 makes the paper the source of truth.

## 3. Bug in the reference implementation, deliberately not reproduced

`turbo_1.py` L198 repairs an empty perturbation mask with
`np.random.randint(0, self.dim - 1, ...)`. NumPy's `randint` excludes its upper bound, so
the last dimension can never be chosen. This code uses `rng.integers(0, dim)`.

Only reachable when `d > 20` (below that, `min(1, 20/d) = 1` and no mask is ever empty).
Regression test: `tests/test_candidates.py::test_last_dimension_is_reachable_by_the_empty_mask_fix`.

## 4. Choices the paper does not determine

| # | Decision | Chosen | Alternatives | Status |
|---|---|---|---|---|
| C1 | GP hyperparameter optimizer | Adam, lr 0.1, 50 steps (`gp.py` L85-89) | L-BFGS with restarts | FROM_OFFICIAL_CODE |
| C2 | GP hyperparameter init | outputscale 1.0, lengthscale 0.5, noise 0.005 | random restarts; warm start | FROM_OFFICIAL_CODE |
| C3 | Candidate de-duplication within a batch | on (chosen candidate masked to `+inf`) | allow duplicates (closer to the equation as written) | FROM_OFFICIAL_CODE |
| C4 | Empty-mask repair | force one random dimension | resample the mask | FROM_OFFICIAL_CODE |
| C5 | Sobol scramble seed | fresh per batch, drawn from the run's generator | fixed seed; one continuing sequence | FROM_OFFICIAL_CODE |
| C6 | Lengthscale normalization | divide by arithmetic mean, then geometric mean | direct geometric-mean division (overflows at d=200) | FROM_OFFICIAL_CODE |
| C7 | TuRBO-m refit skipping | skip refit for TRs that received no points (cached hypers) | always refit (paper-literal reading of App. C) | FROM_OFFICIAL_CODE |
| C8 | Restart data handling | TuRBO-1 discards local data; TuRBO-m orphans it (`_idx = -1`) but keeps it in the global history | retain for the new TR's GP (contradicts "discard") | PARTIALLY_SPECIFIED |
| C9 | `max_evals` includes the initial design | yes | no — would give TuRBO extra evaluations vs. baselines | FROM_OFFICIAL_CODE |
| C10 | Budget overshoot on restart | **truncate** to the remaining budget | overshoot by up to `n_init`, as the official code does (`turbo_m.py` L221) | UNSPECIFIED → ASSUMPTION |
| C11 | RNG policy | explicit `np.random.Generator` threaded through; torch RNG forked and seeded per draw | global seeding (official code) — not reproducible under parallel replications | ASSUMPTION |
| C12 | `n_init` library default | none — caller must pass it; per-experiment values in `PAPER_SPEC.md` §9 | `2·d`, per the official docstring (no paper backing) | UNSPECIFIED |
| C13 | Synthetic function formulas | standard textbook definitions | — | UNSPECIFIED (App. A names them and gives domains, never formulas) |
| C14 | dtype | float64 throughout | float32 (faster, worse GP conditioning) | UNSPECIFIED → ASSUMPTION |

### Note on C11 (found while testing)

GPyTorch's `.sample()` reads torch's **global** generator and accepts no generator
argument. Threading a NumPy `Generator` through the algorithm is therefore not sufficient
for reproducibility — the initial design and candidate sets were deterministic but the
Thompson draws were not, so identically seeded runs diverged after the first batch. Caught
by `tests/test_turbo.py::test_runs_are_reproducible_from_a_seed`. Fixed by forking the
torch RNG (`torch.random.fork_rng`) and seeding it per draw from the run's generator, which
keeps the caller's global torch state untouched.

## 4b. Guards added for cases the paper does not contemplate

Neither is reachable with the paper's settings; both replace silent wrong behavior with a
loud failure. They do not change the algorithm.

| Guard | Why | Status |
|---|---|---|
| `select_candidates` / `select_candidates_across` reject `q` larger than the candidate set | De-duplication (C3) exhausts the set, after which `argmin` over an all-`inf` column returns index 0 and the same point is proposed repeatedly. App. D fixes `n_cand = min(100d, 5000) ≥ 100`, so the paper never encounters this. | ASSUMPTION |
| `train_gp` requires `train_x` and `train_y` to share dtype and device | GPyTorch takes the model dtype from `train_x` and the likelihood dtype from `train_y`; mixing them silently downcasts the fit to float32, contradicting C14. | ASSUMPTION |

## 5. Not implemented

- **Baselines** EBO, BOCK, BOHAMIANN, HeSBO-TS (`PAPER_SPEC.md` §2). App. B states they
  required modification to run in this setting; they are not part of the contribution.
  `src/baselines.py` covers RS, Nelder-Mead, BFGS, CMA-ES and BOBYQA only.
- **§3.6** local-vs-global GP regression ablation.
- **§3.7** trust-region volume / center-trajectory statistics.
- **§3.8** batch-size scaling study.
- **App. G** runtime table (hardware-specific: NVIDIA RTX 2080 Ti).

## 6. Benchmarks that cannot be reproduced from the paper

`src/benchmarks.py` declares these and raises `NotImplementedError` with the reason rather
than silently substituting a stand-in (rule 11).

| Benchmark | Why not |
|---|---|
| Cosmological constants (§3.3, App. F.3) | App. F.3 says "the nine parameters tuned in previous papers, plus three additional parameters chosen from the many available" without naming the three; §3.3 says "substantially larger parameter bounds" without giving any. |
| Robot pushing (§3.1, App. F.1) | Reward formula is given; the pushing simulator, contact and collision model are external (Wang et al. 2018). |
| Rover trajectory (§3.2, App. F.2) | Reward formula is given; terrain, B-spline fitting and collision geometry are external (Wang et al. 2018). |
| Lunar lander (§3.4, App. F.4) | Needs Gym `LunarLander-v2` **and** the paper's "fixed constant set of 50 randomly generated terrains, initial positions, and velocities" — those seeds are unpublished, so results are not comparable to Fig. 3 even with Gym installed. |

Consequence: **only the synthetic benchmarks** (Ackley, Levy, Rastrigin, Hartmann6,
Ackley-200) can be compared against the paper from this repository alone.
