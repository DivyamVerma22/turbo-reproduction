# PERFORMANCE.md

Profiling of the TuRBO reproduction, and the optimization decisions that follow from it.

**Headline: no optimization was applied. The profile does not justify one.** 85% of runtime
is spent in work the paper explicitly specifies, and every remaining candidate measured
either below 1% of runtime or required changing the algorithm. The evidence is below so the
decision can be re-examined rather than taken on faith.

Environment: Python 3.11.0 · torch 2.13.0+cpu · gpytorch 1.15.2 · numpy 2.2.6 · CPU only,
float64. Objective-evaluation time is excluded, matching App. G ("the total runtime for one
optimization run, excluding the time spent evaluating the objective function").

---

## 1. Representative workloads

| Workload | Settings | Wall time |
|---|---|---|
| TuRBO-1, `ackley10` | 220 evals, `q=10`, `n_init=20`, 50 Adam steps | 16.5 s |
| TuRBO-5, `ackley10` | 400 evals, `q=10`, `n_init=20`/TR, 50 Adam steps | 32.6 s |

Both use the App. A domain and the App. D schedule, scaled down in budget so the profile is
fast to reproduce. `n_cand = min(100d, 5000) = 1000` at `d=10`.

## 2. Where the time goes

TuRBO-1, `ackley10`, 220 evaluations (cProfile, cumulative):

| Component | Time | Share | Fixed by |
|---|---:|---:|---|
| `gp.train_gp` | 13.95 s | **85%** | App. C: "The GP hyperparameters are fitted before proposing a new batch by optimizing the log-marginal likelihood" × 50 Adam steps (FROM_OFFICIAL_CODE) |
| ↳ `run_backward` | 4.29 s | 26% | — |
| ↳ `MultivariateNormal.log_prob` | 5.50 s | 34% | — |
| `thompson.thompson_draws` | 1.68 s | 10% | App. D fixes `n_cand`; App. E requires a joint draw |
| `candidates.create_candidates` | 0.05 s | 0.3% | — |
| everything else | ~0.7 s | 4% | — |

TuRBO-5 shifts slightly (`train_gp` 21.5 s of 32.6 s = 66%) because five GPs are fitted per
batch on smaller per-region datasets.

**The dominant cost is mandated, not incidental.** The paper requires a refit before *every*
batch; the step count and optimizer come from the official code. Reducing either changes
fitted hyperparameters and therefore results.

## 3. Scaling behavior at paper-scale settings

Measured directly, since the representative workloads are small:

| Measurement | Result |
|---|---|
| `thompson_draws`, `d=60`, `n_cand=5000`, `q=100` (rover, §3.2) | **2.6–3.3 s per draw**, 200 MB covariance |
| `thompson_draws`, `d=200`, `n_cand=5000`, `q=100` (§3.5) | 1.35 s per draw |
| History `np.vstack` accumulation to 20 000 evals, `d=60` | 0.41 s **total** |
| `TurboM._tr_data`, 50 batches at `n=10 000`, `m=20` | 0.14 s total |
| TuRBO-m candidate arrays, `m=20`, `d=60`, `q=100` | `X_cand` 48 MB + `y_cand` 80 MB = **128 MB** |

Peak memory is dominated by two things: the `n_cand × n_cand` posterior covariance during
sampling (200 MB at `n_cand=5000`) and the TuRBO-m candidate block (128 MB at rover
settings). Both sizes are set by App. D's `n_cand = min(100d, 5000)`.

## 4. Optimization candidates

Each was measured before being accepted or rejected. None was applied.

| # | Optimization | Expected benefit | Correctness risk | Measured | Verdict |
|---|---|---|---|---|---|
| O1 | Skip the unused `to_unit_cube` in `TurboM`'s incumbent loop (it needs only `fX`) | 71% of the data-gathering step | **None** — bit-identical, no RNG or numerics touched | `_tr_data` is **0.08%** of runtime (0.025 s of 31.6 s over 280 calls) | **Rejected.** Real waste, irrelevant magnitude. Optimizing it would add churn to `TurboM` for an unmeasurable gain. |
| O2 | Group history rows once per batch instead of `2m` `np.where` scans | O(n) instead of O(m·n) per batch | None — pure bookkeeping | Folded into O1's 0.08% | **Rejected**, same reason. |
| O3 | Pre-allocate history instead of `np.vstack` per batch | Removes O(n²/q) copying | None — same values | 0.41 s across a **full 20 000-evaluation run**, against hours of GP fitting | **Rejected.** Below noise. |
| O4 | Skip the `state_dict` clone in `propose_from_trust_region` when the caller is TuRBO-1 | Avoids one clone per batch | None | 199 µs/batch → 4 ms per 220-eval run | **Rejected.** Below noise; a flag would cost more clarity than it buys. |
| O5 | Raise `max_cholesky_size` so `n_cand=5000` draws use exact Cholesky | Hypothesis: exact may beat the iterative path, *and* be more accurate | Changes draw values (approximate → exact) | Exact is **slower**: 3.03 s vs 2.64 s (`d=60`), 3.52 s vs 1.35 s (`d=200`) | **Rejected.** The current default of 2000 is already the faster path. Hypothesis tested and disproved. |
| O6 | float32 instead of float64 | ~2× on the GP fit | **High** — changes every numeric result and worsens GP conditioning | not benchmarked | **Rejected on semantics.** `REPRODUCTION_NOTES.md` C14 commits to float64. |
| O7 | Sample independent per-point marginals instead of the joint candidate-set Gaussian | **3× on Thompson draws** (2.45 s → 0.73 s), and turns an O(n_cand³) root into O(n_cand) | **Disqualifying** | measured 3× | **Declined — this is an algorithm change.** See below. |

### On O7, the tempting one

This is the largest available speedup and it must not be taken. App. E: *"we sample the GP
marginal on the candidate set, and then apply regular Thompson sampling."* Read carelessly,
"marginal" invites per-point independent sampling. It means the joint distribution of the GP
restricted to the finite candidate set — the `n_cand`-dimensional Gaussian — which is what
the official code draws via `gp(X_cand).sample()`.

Independent per-point draws discard the spatial correlation between candidates. A Thompson
sample would no longer be a realization of a *function*, and `argmin` over it would reduce to
picking the most optimistic isolated point. That breaks the exploration behavior Sect. 2
relies on. It is a different algorithm that happens to run faster.

## 5. Conclusion

The implementation is close to the floor set by the paper's own specification:

- **85% of runtime** is GP hyperparameter fitting, required before every batch by App. C.
- **~10%** is Thompson sampling, whose cost is set by App. D's candidate-set size and App.
  E's joint-draw requirement, and which already uses the faster of the two available root
  decompositions.
- **Everything else is under 1%**, including every candidate that would have been safe.

Making this reproduction meaningfully faster requires either changing what the paper
specifies or moving to a GPU (App. G reports the paper's own runs on an RTX 2080 Ti, where
TuRBO completes every benchmark "in minutes"). Neither is an optimization of this code.

**If runtime becomes a blocker**, the honest lever is hardware or budget, not code: the
`n_training_steps` parameter is already exposed and reducing it trades fidelity for speed
explicitly, at the caller's discretion, rather than silently.
