# PAPER_SPEC.md — TuRBO

**Paper:** Scalable Global Optimization via Local Bayesian Optimization
**Authors:** David Eriksson, Michael Pearce, Jacob R. Gardner, Ryan Turner, Matthias Poloczek
**arXiv:** [1910.01739](https://arxiv.org/abs/1910.01739) (NeurIPS 2019) — categories cs.LG, stat.ML
**Official code:** https://github.com/uber-research/TuRBO (cited in §1, last paragraph)
**Source used for this spec:** full ar5iv HTML text (main text §1–§4 + Supplementary A–G), cross-checked
against the PDF, plus the official repository at `master`.

**Repository status at time of writing:** the repo is empty — `git ls-files` returns 0 files, only `.git/`
exists. There is no generated code to inspect yet, so requirement 2 below ("every module that must exist")
is written as a *target* module list derived from the paper, not as a description of existing code.

**Notation.** `d` = problem dimension, `q` = batch size, `m` = number of trust regions, `n` = number of
evaluations so far, `n_cand` = candidate-set size. TuRBO **minimizes**; the reward-style benchmarks
(robot pushing, rover, lunar lander) are negated before being handed to the optimizer.

**Two structural facts about this paper that shape everything below:**

1. **It contains no numbered equations.** Every formula is inline. Appendix E states it "provide[s]
   details and pseudo-code," but no `Algorithm` float exists in the published text (verified by grepping
   both the ar5iv HTML and the raw PDF: the only "Algorithm N" hit is bibliography entry [58],
   "Algorithm 778: L-BFGS-B"). Equation references below are therefore by section and quoted text.
2. **The official code differs from the paper in at least six places.** These are itemized in §10.A and
   are the single biggest reproduction risk.

---

## 1. Central contribution (5–10 bullets)

- **Reject the global surrogate.** The thesis is that BO fails in high dimensions with large budgets
  because global GP models impose *implicit homogeneity* — "the commonly used global Gaussian process
  (GP) models implicitly suppose that characteristic lengthscales and signal variances of the function
  are constant in the search space" (§1). Heterogeneous objectives violate this.
- **Reject global acquisition.** The second failure mode is over-exploration: "search spaces grow
  considerably faster than sampling budgets due to the curse of dimensionality [which] implies the
  inherent presence of regions with large posterior uncertainty. For common myopic acquisition
  functions, this results in an overemphasized exploration and a failure to exploit promising areas" (§1).
- **Local GPs inside trust regions.** TuRBO fits GP surrogates restricted to a hyperrectangular trust
  region (TR) centered on the best solution found so far, inheriting the noise-robustness and calibrated
  uncertainty of BO while allowing per-region hyperparameters (§2, "Local modeling" / "Trust regions").
- **Lengthscale-anisotropic trust region.** The TR is not a cube: side lengths are rescaled by the GP's
  ARD lengthscales at constant volume, `L_i = λ_i·L / (Π_j λ_j)^(1/d)` (§2, "Trust regions").
- **Classical success/failure TR resizing.** Double `L` after `τ_succ` consecutive successes, halve after
  `τ_fail` consecutive failures, restart the TR when `L < L_min`, cap at `L_max` (§2, "Trust regions").
- **Implicit multi-armed bandit across TRs.** `m` TRs run simultaneously with independent GPs; Thompson
  sampling over the *union* of the TRs decides both which TR to spend on and where inside it — "TS
  provides a principled solution to both the problem of selecting candidates within a single TR, and
  selecting candidates across the set of trust regions simultaneously" (§2, "Trust region Bayesian
  optimization").
- **Batch acquisition that scales linearly.** TS batching cost scales linearly in `q`, unlike other batch
  acquisitions that "do not scale to large batch sizes in practice" (§1.1); §3.8 shows the speed-up from
  large batches is "essentially linear."
- **Empirical claim.** TuRBO outperforms BO (EBO, BOCK, BOHAMIANN, GP-TS, HeSBO-TS), evolutionary
  (CMA-ES), and local/OR methods (BOBYQA, Nelder-Mead, BFGS) on 14D robot pushing, 60D rover, 12D
  cosmological constants, 12D lunar lander, and 200D Ackley (§3).
- **Mechanistic evidence, not just benchmark wins.** §3.6 shows local GPs beat a global GP on held-out log
  loss on *every one of 50 trials* (1.174 vs 1.284 nats, `p < 1e-4`), and that learned hyperparameters
  vary substantially across regions — the paper's direct evidence for the heterogeneity claim.
- **Cheap.** Appendix G / Table 1: TuRBO runs in minutes on every benchmark and is ">2000× faster than
  the slowest BO method."

---

## 2. Modules that must exist in code

Target layout. Nothing here exists yet.

| Module | Responsibility | Paper anchor |
|---|---|---|
| `src/utils.py` | `to_unit_cube`, `from_unit_cube`, `latin_hypercube` initial design | App. C ("domain is rescaled to [0,1]^d"); App. A ("Latin hypercube design [29]") |
| `src/gp.py` | Matérn-5/2 ARD GP + constant mean + Gaussian likelihood; `train_gp` fitting the log-marginal likelihood under hyperparameter box constraints | App. C |
| `src/trust_region.py` | TR state (`L`, `succcount`, `failcount`), success/failure update rule, `L`-rescaled bounds, restart trigger | §2 "Trust regions"; App. D |
| `src/candidates.py` | Scrambled Sobol candidate set inside TR ∩ [0,1]^d, coordinate-perturbation mask | App. D |
| `src/thompson.py` | Draw `q` posterior realizations on the candidate set; greedy argmin selection with de-duplication | §2 "TRBO"; App. E |
| `src/turbo_1.py` | TuRBO-1: single TR, restart-on-collapse outer loop | §2; App. D |
| `src/turbo_m.py` | TuRBO-m: `m` TRs, per-TR counters, cross-TR bandit selection, per-TR restart | §2; App. D |
| `src/benchmarks.py` | Ackley/Levy/Rastrigin/Hartmann6 synthetics; wrappers for robot pushing, rover, lunar lander, cosmological constants | §3; App. A; App. F |
| `src/evaluate.py` | Replication driver, best-so-far traces, mean ± one standard error curves | §3 ("mean performances with one standard error") |
| `src/baselines.py` *(optional)* | CMA-ES / BOBYQA / NM / BFGS / GP-TS wrappers for the comparison plots | App. B |

**Deliberately out of scope for a core reproduction:** EBO, BOCK, BOHAMIANN, HeSBO-TS. These are
third-party baselines that App. B says required modification to run at all; they are not part of the
contribution.

---

## 3. Inputs, outputs, tensor shapes, dtypes

Dtype is `float64` throughout unless stated. The official code runs `float64` on CPU and only switches to
the configured dtype above `min_cuda = 1024` points (`turbo_1.py` L120–L123, L163–L167) — that switch is
not in the paper.

### `utils`

| Function | Input | Output |
|---|---|---|
| `to_unit_cube(x, lb, ub)` | `x (n,d) f64`, `lb (d,) f64`, `ub (d,) f64` | `(n,d) f64` in `[0,1]^d` |
| `from_unit_cube(x, lb, ub)` | `x (n,d) f64` in `[0,1]^d` | `(n,d) f64` in original bounds |
| `latin_hypercube(n_pts, dim)` | ints | `(n_pts, dim) f64` in `[0,1]^d` |

### `gp.train_gp`

| Item | Shape / dtype |
|---|---|
| `train_x` | `(n, d)` f64, **must already be in `[0,1]^d`** |
| `train_y` | `(n,)` f64, **must already be standardized** |
| returns | fitted GP in eval mode |
| `gp.covar_module.base_kernel.lengthscale` | `(1, d)` f64 → raveled to `(d,)` |
| `gp.covar_module.outputscale` | scalar |
| `gp.likelihood.noise` | scalar |

### `trust_region`

| Item | Shape / dtype |
|---|---|
| `L` (base side length) | scalar f64 — TuRBO-1; `(m,)` f64 — TuRBO-m |
| `succcount`, `failcount` | int — TuRBO-1; `(m,)` int — TuRBO-m |
| `x_center` | `(1, d)` f64 in `[0,1]^d` |
| `weights` (normalized lengthscales) | `(d,)` f64, `Π weights = 1` |
| `lb_tr`, `ub_tr` | `(1, d)` f64, clipped to `[0,1]` |

### `candidates` / `thompson`

| Item | Shape / dtype |
|---|---|
| `n_cand` | `min(100·d, 5000)` |
| `X_cand` | `(n_cand, d)` f64 — TuRBO-1; `(m, n_cand, d)` — TuRBO-m |
| `mask` | `(n_cand, d)` bool |
| `y_cand` (TS draws) | `(n_cand, q)` f64 — TuRBO-1; `(m, n_cand, q)` — TuRBO-m |
| `X_next` | `(q, d)` f64 |
| `idx_next` (owning TR per selected point) | `(q, 1)` int — TuRBO-m only |

### Optimizer state

| Item | Shape / dtype |
|---|---|
| `X` (global history) | `(n, d)` f64, original bounds |
| `fX` (global history) | `(n, 1)` f64 |
| `_idx` (TR ownership, TuRBO-m) | `(n, 1)` int; `-1` marks points orphaned by a restart |
| `f(x)` | `x (d,) f64` → scalar f64 |

---

## 4. Implementation-relevant equations → pseudocode

The paper has no numbered equations; each block below quotes its source text.

### E1 — Problem statement (§2)

> "Find x\* ∈ Ω such that f(x\*) ≤ f(x), ∀x ∈ Ω, where f: Ω → ℝ and Ω = [0,1]^d. We observe potentially
> noisy values y(x) = f(x) + ε, where ε ~ N(0, σ²)."

```
minimize f over Omega = [0,1]^d
observed:  y(x) = f(x) + eps,  eps ~ Normal(0, sigma^2)   # homoscedastic Gaussian noise
```

### E2 — Anisotropic trust-region side lengths (§2, "Trust regions")

> "The actual side length for each dimension is obtained from this base side length by rescaling
> according to its lengthscale λ_i in the GP model while maintaining a total volume of L^d.
> That is, L_i = λ_i L / (Π_{j=1..d} λ_j)^{1/d}."

```
lambda = gp.lengthscales                      # (d,)
w      = lambda / geometric_mean(lambda)      # prod(w) == 1 exactly
L_i    = w * L                                # (d,), prod(L_i) == L^d
lb_tr  = clip(x_center - L_i / 2, 0, 1)       # [FROM_OFFICIAL_CODE] turbo_1.py L185
ub_tr  = clip(x_center + L_i / 2, 0, 1)       # [FROM_OFFICIAL_CODE] turbo_1.py L186
```

*Note:* the paper gives the side lengths and says the TR is "a hyperrectangle centered at the best
solution found so far," which implies the `±L_i/2` extents. The **clipping** to `[0,1]^d` is only implied
("the intersection of the TR and the domain [0,1]^d", App. D) and is explicit in the code. Clipping breaks
the exact `Π L_i = L^d` volume invariant near the boundary — the paper does not discuss this.

### E3 — TR center (§2, "Trust regions")

> "In the noise-free case, we set x\* to the location of the best observation so far. In the presence of
> noise, we use the observation with the smallest posterior mean under the surrogate model."

```
if noise_free:
    x_center = X[argmin(fX)]
else:
    x_center = X[argmin(gp.posterior_mean(X))]      # NOT implemented in official code — see §10 A3
```

### E4 — TR resizing (§2, "Trust regions"; App. D)

> "After τ_succ consecutive successes, we double the size of the TR, i.e., L ← min{L_max, 2L}. After
> τ_fail consecutive failures, we halve the size of the TR: L ← L/2. We reset the success and failure
> counters to zero after we change the size of the TR. Whenever L falls below a given minimum threshold
> L_min, we discard the respective TR and initialize a new one with side length L_init."

TuRBO-1 (App. D: "we consider an improvement from at least one evaluation in the batch a success"):

```
if min(fX_batch) improves on best_in_TR:
    succcount += 1;  failcount  = 0
else:
    succcount  = 0;  failcount += 1
if succcount == tau_succ:  L = min(2*L, L_max);  succcount = 0
if failcount == tau_fail:  L = L / 2;            failcount = 0
if L < L_min:              restart_TR()
```

TuRBO-m (App. D: per-TR counters; a failure adds the batch size, not 1):

```
for each TR_l that received q_l > 0 points this batch:
    if min(fX_batch_l) < best_in_TR_l:
        succcount[l] += 1;    failcount[l]  = 0
    else:
        succcount[l]  = 0;    failcount[l] += q_l     # "add q_l to the failure counter"
    if succcount[l] == tau_succ:  L[l] = min(2*L[l], L_max);  succcount[l] = 0
    if failcount[l] >= tau_fail:  L[l] = L[l] / 2;            failcount[l] = 0
# App. D: "The failure counter is set to tau_fail if we increment past this tolerance,
# which will trigger a halving of its side length."  -> the >= comparison is required, not ==.
```

### E5 — Thompson-sampling acquisition across TRs (§2, "Trust region Bayesian optimization")

> "x_i^(t) ∈ argmin_ℓ argmin_{x ∈ TR_ℓ} f_ℓ^(i) where f_ℓ^(i) ~ GP_ℓ^(t)(μ_ℓ(x), k_ℓ(x,x'))."

```
for l in 1..m:
    X_cand[l] = sobol_candidates(TR_l)                     # (n_cand, d)
    y_cand[l] = gp_l.sample_posterior(X_cand[l], n=q)      # (n_cand, q)  q independent realizations
for i in 1..q:
    (l*, j*) = argmin over (l, j) of y_cand[l][j][i]       # joint argmin: picks TR and point at once
    x_i      = X_cand[l*][j*]
    owner[i] = l*
    y_cand[l*][j*][:] = +inf                               # never select the same candidate twice
```

The de-duplication line is not in the paper; it is in the code (`turbo_1.py` L233, `turbo_m.py` L156).

### E6 — Candidate-set perturbation (App. D)

> "In order to not perturb all coordinates at once, we use the value in the Sobol sequence with
> probability min{1, 20/d} for a given candidate and dimension, and the value of the center otherwise."

```
p_perturb    = min(1, 20 / d)
pert         = lb_tr + (ub_tr - lb_tr) * scrambled_sobol(n_cand, d)   # (n_cand, d)
mask         = uniform(n_cand, d) <= p_perturb                        # bool
# [FROM_OFFICIAL_CODE] rows with an all-False mask get one random dim forced True (turbo_1.py L197-198)
X_cand       = tile(x_center, n_cand)
X_cand[mask] = pert[mask]
```

For `d <= 20` this is a no-op (`p_perturb = 1`, every coordinate perturbed).

### E7 — Value standardization before GP fitting (App. C)

> "The domain is rescaled to [0,1]^d and the function values are standardized before fitting the GP."

```
# Paper text implies mean/std. Official code uses MEDIAN/std -- see §10 A4.
mu, sigma = median(fX), std(fX)
sigma     = 1.0 if sigma < 1e-6 else sigma
fX_std    = (fX - mu) / sigma
...
y_cand    = mu + sigma * y_cand_std          # de-standardize TS draws before comparing
```

De-standardizing is order-preserving within one TR, but in **TuRBO-m it is load-bearing**: the bandit
compares draws across TRs that each have their own `(mu, sigma)`. Skipping it silently breaks the
cross-TR comparison in E5.

### E8 — GP model and objective (App. C)

> "the GP is parameterized using a Matérn-5/2 kernel with ARD and a constant mean function for all
> experiments. The GP hyperparameters are fitted before proposing a new batch by optimizing the
> log-marginal likelihood."

```
r        = sqrt( sum_i ( (x_i - x'_i)^2 / lambda_i^2 ) )            # ARD scaled distance
k(x,x')  = s^2 * (1 + sqrt(5)*r + (5/3)*r^2) * exp(-sqrt(5)*r)      # Matern-5/2
mean(x)  = c                                                        # ConstantMean, c learned
K_y      = K + sigma^2 * I
log p(y|X, theta) = -0.5*(y-c)^T K_y^-1 (y-c) - 0.5*log|K_y| - (n/2)*log(2*pi)
theta    = argmax log p(y|X, theta)   s.t. the box constraints in section 9
```

The Matérn-5/2 closed form is the standard one (Rasmussen & Williams); the paper names the kernel but does
not write it out. App. C also specifies **GPyTorch** with **CG solves + Lanczos log-determinant**
(following Dong et al. 2017a) for scalability — an exact-Cholesky implementation is numerically different
at large `n` but not conceptually.

### E9 — Benchmark objectives (App. F)

Robot pushing (F.1), `d = 14`:

```
f(x) = sum_{i=1..2} ( ||x_gi - x_si|| - ||x_gi - x_fi|| )    # reward, MAXIMIZED -> negate for TuRBO
```

Rover (F.2), `d = 60` (30 points in a 2D plane, B-spline fit):

```
f(x) = c(x) - 10*( ||x_{1,2} - x_s||_1 + ||x_{59,60} - x_g||_1 ) + 5
# c(x) penalizes each collision along the trajectory by -20
```

Lunar lander (F.4), `d = 12`: mean final reward over a **fixed** set of 50 random terrains/initial
conditions, episodes capped at 1000 steps, "after which failure to land was scored as a crash."

---

## 5. Architecture and execution order

There is no neural architecture. The "architecture" is the per-batch control flow.

### TuRBO-1 (§2; App. D)

```
outer restart loop:  while n_evals < max_evals:
  1. reset L = L_init, succcount = 0, failcount = 0, discard TR data
  2. X_init = latin_hypercube(n_init, d) -> from_unit_cube -> evaluate f      # n_init evals
  3. inner loop: while n_evals < max_evals AND L >= L_min:
       a. X_unit   = to_unit_cube(TR data)
       b. fX_std   = standardize(fX)                                (E7)
       c. gp       = train_gp(X_unit, fX_std)                       (E8)  <- refit EVERY batch
       d. x_center = best point in TR                               (E3)
       e. lb_tr, ub_tr from ARD lengthscales                        (E2)
       f. X_cand   = sobol + perturbation mask                      (E6)
       g. y_cand   = q posterior draws on X_cand, de-standardized   (E5, E7)
       h. X_next   = q greedy argmins with de-duplication           (E5)
       i. evaluate f on X_next                                      # q evals
       j. update L, counters                                        (E4)
       k. append to TR data and global history
```

### TuRBO-m (§2; App. D)

```
1. for l in 1..m: X_init_l = latin_hypercube(n_init, d), evaluate      # m * n_init evals total
2. while n_evals < max_evals:
     a. for l in 1..m:  fit GP_l on TR_l's own data; build X_cand[l], y_cand[l]
     b. joint TS selection of q points across all m TRs -> X_next, idx_next   (E5)
     c. evaluate f on X_next (one batch of q)                                # q evals
     d. for each l with q_l > 0: update L[l] and counters                    (E4)
     e. append to global history with TR ownership
     f. for each l with L[l] < L_min: restart TR_l -- reset L and counters,
        ORPHAN its old points (idx <- -1), draw a fresh n_init LHD, evaluate  # n_init extra evals
```

**Ordering constraints that matter and are easy to get wrong:**

- The GP must be refit **before** `x_center` / `lb_tr` / `ub_tr` are computed — the TR bounds depend on the
  *current* fitted lengthscales (E2). Reversing this silently uses stale geometry.
- TR updates (step d) must be computed against each TR's best value **before** the new batch is appended
  to the history, or every batch trivially counts as a non-improvement.
- In TuRBO-m, restarts (step f) consume budget *outside* the `q`-per-iteration accounting, so total
  evaluations can overshoot `max_evals` by up to `n_init`.

---

## 6. Loss functions and objective terms

TuRBO has **no training loss** in the deep-learning sense. There are exactly two objective terms:

1. **The black-box objective `f`** — minimized by the optimizer, never differentiated. Gradient-free by
   assumption: "closed form expressions and derivatives are unavailable" (§1).
2. **Negative log marginal likelihood of the GP** — the only quantity minimized by gradient descent, used
   solely to fit GP hyperparameters `θ = {c, λ_{1..d}, s², σ²}` per TR per batch (App. C; formula in E8).
   Optimized under the box constraints in §9.

There is no regularization term, no auxiliary loss, and no gradient-based acquisition optimization — TS is
discretized over the Sobol candidate set (App. E: "we cannot sample an entire function f from the GP
posterior in practice. We therefore work in a discretized setting").

---

## 7. Training procedure, optimizer, scheduler

"Training" = GP hyperparameter fitting. Frequency and objective are specified; the optimizer is not.

| Item | Value | Status | Source |
|---|---|---|---|
| Fit frequency | before every proposed batch | SPECIFIED | App. C |
| Objective | log-marginal likelihood | SPECIFIED | App. C |
| GP library | GPyTorch, CG + Lanczos | SPECIFIED | App. C |
| Optimizer | Adam | FROM_OFFICIAL_CODE | `gp.py` L85 |
| Learning rate | 0.1 | FROM_OFFICIAL_CODE | `gp.py` L85 |
| Steps | 50 | FROM_OFFICIAL_CODE | `turbo_1.py` L61 |
| Init: outputscale / lengthscale / noise | 1.0 / 0.5 / 0.005 | FROM_OFFICIAL_CODE | `gp.py` L79–81 |
| Hyperparameter warm-start | TuRBO-m caches hypers and skips refitting for TRs that received no points this batch (`n_training_steps = 0`) | FROM_OFFICIAL_CODE | `turbo_m.py` L165 |
| Cholesky↔CG switch | `max_cholesky_size = 2000` | FROM_OFFICIAL_CODE | `turbo_1.py` L60 |

No learning-rate schedule exists — 50 fixed Adam steps at lr 0.1 per fit. The paper states none.

---

## 8. Dataset preprocessing and evaluation procedure

**Preprocessing (App. C):**

1. Rescale the domain to `[0,1]^d` (affine, per-dimension, using the known box bounds `lb`/`ub`).
2. Standardize function values before fitting the GP (see E7 and §10 A4).
3. Initial design: Latin hypercube (App. A cites LHD [29]; the main-text experiments say only "initial
   points"). The official `latin_hypercube` uses stratified centers plus a uniform jitter of ±1/(2n).

**Evaluation procedure (§3):**

- Metric = best objective value found so far as a function of the number of evaluations.
- "Performance plots show the mean performances with one standard error" (§3).
- App. A: 30 replications for the synthetic suite. Main-text replication counts are not stated in the
  extracted text.
- Reward-style problems (robot pushing, rover, lunar lander) are plotted as reward (higher is better)
  while TuRBO minimizes — negate on the way in and on the way out.
- Ablations to reproduce: §3.6 local-vs-global GP log loss (20 hypercubes of side 0.4, 200 training points
  each, 4000 total, isotropic kernel, 50 trials, paired `t`-test); §3.7 TR volume / center-trajectory
  statistics over 50 restarts on the 60D rover; §3.8 batch-size scaling `q ∈ {1,2,4,…,64}` with
  `max{200q, 6400}` samples, 30 replications.
- Runtime is reported excluding objective-function evaluation time (App. G).

---

## 9. Stated hyperparameters with exact source

### TuRBO algorithm

All from App. D, first sentence — "In all experiments, we use the following hyperparameters for TuRBO-1".

| Symbol | Value | Source |
|---|---|---|
| `τ_succ` | 3 | App. D |
| `τ_fail` | `⌈d/q⌉` | App. D |
| `L_min` | `2^-7` = 0.0078125 | App. D |
| `L_max` | 1.6 | App. D |
| `L_init` | 0.8 | App. D |
| `n_cand` | `min(100d, 5000)` | App. D |
| perturbation probability | `min(1, 20/d)` | App. D |
| candidate sequence | scrambled Sobol, regenerated per batch, inside TR ∩ `[0,1]^d` | App. D |
| TuRBO-m tolerances | "the same tolerances as in the sequential case (q=1)" | App. D |

### GP (App. C)

| Symbol | Value | Source |
|---|---|---|
| kernel | Matérn-5/2 with ARD | App. C |
| mean | constant | App. C |
| lengthscale bound `λ_i` | `[0.005, 2.0]` | App. C |
| signal variance bound `s²` | `[0.05, 20.0]` | App. C |
| noise variance bound `σ²` | `[0.0005, 0.1]` | App. C |

### Per-experiment budgets

| Experiment | `d` | budget `n` | `q` | `n_init` | Source |
|---|---|---|---|---|---|
| Synthetic — Ackley `[-5,10]^10`, Levy `[-5,10]^10`, Rastrigin `[-3,4]^10`, Hartmann6 `[0,1]^6` | 10/10/10/6 | 500 (50 batches) | 10 | 20 (TuRBO-5: 10 per TR); 30 runs | App. A |
| Robot pushing | 14 | 10 000 | 50 | 100 (TuRBO-20: 50 per TR) | §3.1 |
| Rover trajectory | 60 | 20 000 (200 steps) | 100 | 200 (TuRBO-20: 100 per TR) | §3.2 |
| Cosmological constants | 12 | 2 000 | 50 | 50 (TuRBO-5: 20 per TR) | §3.3 |
| Lunar lander | 12 | 1 500 | 50 | 50 (TuRBO-5: 20 per TR) | §3.4 |
| Ackley-200 `[-5,10]^200` | 200 | 10 000 | 100 | 200 | §3.5 |
| Batch-size study (robot pushing) | 14 | `max{200q, 6400}` | 1…64 | not stated | §3.8 |

### Baseline settings worth recording (App. B)

CMA-ES: pycma, default settings, population size = batch size, initialized from the best of a few initial
points. NM/BFGS: SciPy; BOBYQA: nlopt; all with multiple restarts from the best of a few initial points.
GP-TS / BOCK / BOHAMIANN: 5000-point scrambled Sobol candidate set per batch.
HeSBO-TS target dimensions: 8 (robot pushing, `p=5`), 10 (rover, `p=15`), 8 (cosmology, `p=4`),
8 (lunar, `p=3`), 20 (Ackley-200, `p=5`), 4 (Hartmann6), 6 (other synthetics).

### Hardware (App. G)

NVIDIA RTX 2080 Ti for the scalable-GP baselines. TuRBO runtimes: <1 min (synthetic, lunar, cosmology),
8 min (robot pushing), 22 min (rover), 10 min (Ackley-200).

---

## 10. Ambiguity table

Legend: **SPECIFIED** = stated in the paper. **PARTIALLY_SPECIFIED** = named but underdetermined.
**UNSPECIFIED** = absent. **ASSUMPTION** = my choice, no source. **FROM_OFFICIAL_CODE** = resolved from
uber-research/TuRBO@master, which is *not* the paper and may deviate from it.

### A. Paper ↔ official code conflicts (highest reproduction risk)

| # | Decision | Status | Evidence | Chosen value | Alternatives |
|---|---|---|---|---|---|
| A1 | Definition of a "success" | PARTIALLY_SPECIFIED → FROM_OFFICIAL_CODE | §2: "a 'success' [is] a candidate that improves upon x\*" — no tolerance given. Code: `min(fX_next) < min(fX) - 1e-3*abs(min(fX))` (`turbo_1.py` L138, `turbo_m.py` L109) | Relative margin `1e-3·|f_best|`, per code | Strict `<` per the paper text; an absolute epsilon. **Strict `<` makes TRs expand more readily on noisy or plateaued objectives — this materially changes behavior** |
| A2 | `τ_fail` | SPECIFIED (conflicting) | Paper App. D: `⌈d/q⌉`. Code TuRBO-1: `⌈max(4/q, d/q)⌉` (`turbo_1.py` L106); code TuRBO-m: `max(5, d)` (`turbo_m.py` L86) | Paper value `⌈d/q⌉` for TuRBO-1; flag the discrepancy | Code values. Differs whenever `d < 4` (TuRBO-1) and for every TuRBO-m run (`max(5,d)` vs `⌈d/1⌉ = d`) |
| A3 | TR center under noise | SPECIFIED but unimplemented upstream | §2: "In the presence of noise, we use the observation with the smallest posterior mean." Code always uses `X[fX.argmin()]` and carries the comment "NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead" (`turbo_1.py` L152–154, L181) | Implement **both**; default to the paper rule (posterior mean) when a `noisy=True` flag is set | Always-argmin (code behavior). Robot pushing and lunar lander are explicitly noisy problems, so this path is exercised during reproduction |
| A4 | Centering statistic for standardization | PARTIALLY_SPECIFIED | App. C says "standardized" (conventionally mean/std). Code: `mu, sigma = np.median(fX), fX.std()` (`turbo_1.py` L159) | Median/std, per code | Mean/std, per the ordinary reading of "standardized". Affects the GP fit through the constant mean and through outlier robustness |
| A5 | Noise-variance upper bound | SPECIFIED (conflicting) | Paper App. C: `σ² ∈ [0.0005, 0.1]`. Code: `Interval(5e-4, 0.2)` (`gp.py` L48) | Paper value 0.1; flag | Code value 0.2. Matters on high-noise objectives where the fit saturates the bound |
| A6 | Non-ARD lengthscale bound | UNSPECIFIED in paper | The paper gives only the ARD bound. Code: `[0.005, sqrt(d)]` when `use_ard=False` (`gp.py` L52) | ARD always on (paper: "Matérn-5/2 kernel with ARD … for all experiments") | Isotropic kernel — used by the paper *only* for the §3.6 regression ablation ("For the sake of illustration, we used an isotropic kernel") |

### B. Underspecified in the paper, resolvable from official code

| # | Decision | Status | Evidence | Chosen value | Alternatives |
|---|---|---|---|---|---|
| B1 | GP hyperparameter optimizer | FROM_OFFICIAL_CODE | Paper: "optimizing the log-marginal likelihood" (App. C), no optimizer named. Code: Adam, lr 0.1, 50 steps (`gp.py` L85–89) | Adam(lr=0.1), 50 steps | L-BFGS with restarts (the more common GP-fitting choice; gives different fits) |
| B2 | GP hyperparameter initialization | FROM_OFFICIAL_CODE | `gp.py` L79–81: outputscale 1.0, lengthscale 0.5, noise 0.005 | Those values | Random restarts; previous-batch warm start |
| B3 | TR bound construction (`±L_i/2` + clip) | PARTIALLY_SPECIFIED → FROM_OFFICIAL_CODE | Paper gives `L_i` and "hyperrectangle centered at x\*"; App. D says "intersection of the TR and the domain". Code `turbo_1.py` L185–186 | `clip(x_center ± w·L/2, 0, 1)` | Reflecting or shifting the box to preserve volume at the boundary — the paper's stated `Π L_i = L^d` invariant is violated by clipping and neither source addresses it |
| B4 | Lengthscale normalization arithmetic | PARTIALLY_SPECIFIED → FROM_OFFICIAL_CODE | Paper: `L_i = λ_i L/(Π λ_j)^{1/d}`. Code divides by the mean first for numerical stability, then by the geometric mean (`turbo_1.py` L183–184) | Two-step normalization per code (mathematically identical, numerically safer in 200D) | Direct geometric-mean division — can overflow/underflow for large `d` |
| B5 | Candidate de-duplication in TS | UNSPECIFIED → FROM_OFFICIAL_CODE | Not in §2 or App. E. Code sets the chosen row to `inf` across all `q` columns (`turbo_1.py` L233, `turbo_m.py` L156) | De-duplicate | Allow duplicates (pure independent TS — arguably closer to E5 as literally written) |
| B6 | Empty-perturbation-mask handling | UNSPECIFIED → FROM_OFFICIAL_CODE | App. D describes the mask but not the all-false case. Code forces one random dimension true (`turbo_1.py` L197–198) | Force one dimension | Resample the mask. Only relevant for `d > 20`. **Note the code uses `randint(0, d-1)`, which can never select the last dimension — an off-by-one in the reference implementation. Prefer `randint(0, d)`** |
| B7 | Sampling from the posterior vs the predictive distribution | PARTIALLY_SPECIFIED → FROM_OFFICIAL_CODE | §2 / E5 says `f ~ GP(μ, k)` (latent posterior). Code samples `gp.likelihood(gp(X_cand))` — the **predictive** distribution, i.e. with observation noise added (`turbo_1.py` L216) | Predictive, per code | Latent posterior, per the paper's equation. Adds `σ²` jitter to every draw, which increases TS exploration |
| B8 | TuRBO-m: skip refit for TRs with no new data | UNSPECIFIED → FROM_OFFICIAL_CODE | `turbo_m.py` L165: `n_training_steps = 0 if self.hypers[i] else self.n_training_steps` | Cache and skip | Refit every TR every batch (slower; App. C's "fitted before proposing a new batch" arguably requires it) |
| B9 | Restart data handling | PARTIALLY_SPECIFIED | §2: "we discard the respective TR and initialize a new one." Code TuRBO-1 discards the local data entirely; TuRBO-m orphans it via `_idx = -1` but keeps it in the global history (`turbo_m.py` L192) | Orphan; keep in global history for reporting | Retain the data for the new TR's GP — would contradict "discard" |
| B10 | Number of Sobol draws / scrambling seed | UNSPECIFIED → FROM_OFFICIAL_CODE | App. D says "scrambled Sobol" with no seeding scheme. Code draws a fresh `seed = randint(1e6)` per batch (`turbo_1.py` L191–192) | Fresh random scramble seed per batch | Fixed seed; a continuing Sobol sequence across batches |

### C. Unspecified in both paper and code

| # | Decision | Status | Evidence | Chosen value | Alternatives |
|---|---|---|---|---|---|
| C1 | `n_init` default | UNSPECIFIED | Per-experiment values only (§3.1–3.5, App. A). The code docstring recommends `2·d` with no paper backing | `2·d` as the library default; the exact per-experiment values from §9 for reproduction runs | `d+1`; a fixed 20 |
| C2 | Replication counts for main-text experiments | UNSPECIFIED | App. A gives 30 runs for the synthetics; §3 says only "mean … with one standard error" | 30, matching App. A | 10 or 20 (common); the paper's actual number is unrecoverable from the text |
| C3 | Whether `max_evals` includes initial points | UNSPECIFIED → FROM_OFFICIAL_CODE | Code counts `n_init` against the budget (`turbo_1.py` L253) | Inclusive | Exclusive — would give TuRBO extra evaluations relative to the baselines |
| C4 | Budget overshoot on restart | UNSPECIFIED | Neither source discusses it; `turbo_m.py` L221 adds `n_init` evaluations after the budget check | Allow the overshoot; record the actual `n_evals` | Truncate the restart design to the remaining budget |
| C5 | Global RNG seeding / reproducibility policy | UNSPECIFIED | Code uses the global NumPy RNG with no seed parameter | Thread an explicit `seed` / `Generator` through | Global seeding (not reproducible under parallel replications) |
| C6 | Behavior when a TR has fewer points than dimensions | UNSPECIFIED | Not discussed. Matters at `d=200`, `n_init=200`, where the GP is fit on ~200 points in 200D with 200 ARD lengthscales | Fit anyway (what the code does) | Fall back to an isotropic kernel until `n > d` |
| C7 | Objective-value sign convention | ASSUMPTION | §2 states minimization; §3 plots rewards for pushing / rover / lunar | Optimizer minimizes; benchmark wrappers negate rewards internally and report reward | A maximization-native API |
| C8 | float32 vs float64 policy | UNSPECIFIED → FROM_OFFICIAL_CODE | App. C mentions GPyTorch/CG but no precision. Code: float64 on CPU below `min_cuda=1024` points, configurable dtype above (`turbo_1.py` L120–123) | float64 throughout; float32 only as an explicit opt-in | float32 everywhere (faster, meaningfully worse GP conditioning) |
| C9 | Cosmological-constants objective | PARTIALLY_SPECIFIED | App. F.3: "nine parameters tuned in previous papers, plus three additional parameters chosen from the many available"; §3.3: "substantially larger parameter bounds" — neither the 3 extra parameters nor any bounds are given | Not reproducible as stated; requires the toolbox linked in §3.3 | Omit this benchmark. **This experiment cannot be reproduced from the paper alone** |
| C10 | Robot pushing / rover simulators | PARTIALLY_SPECIFIED | App. F.1–F.2 give the reward formulas and cite Wang et al. 2018 for the simulators; the simulators themselves are external | Vendor them from the Wang et al. 2018 release | Reimplement from the formulas — the collision model and B-spline details are not in this paper |

---

## Reproduction risk summary

**Reproducible from the paper alone:** the TuRBO-1 and TuRBO-m algorithms, the GP model, and the synthetic
benchmarks (Ackley, Levy, Rastrigin, Hartmann6, Ackley-200) — modulo the ambiguity items above.

**Not reproducible from the paper alone:** cosmological constants (C9); robot pushing and rover (C10,
external simulators); lunar lander (needs the exact fixed set of 50 seeds, which is not published).

**Items most likely to cause a silent mismatch, in order:** A1 (success tolerance), A3 (noisy TR center),
B7 (predictive vs latent sampling), A2 (`τ_fail` for TuRBO-m), E7 (de-standardization in the cross-TR
bandit comparison).
