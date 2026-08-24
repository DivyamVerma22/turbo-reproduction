# VERIFICATION.md

Audit of the repository against `PAPER_SPEC.md` and the original paper
(Eriksson et al., NeurIPS 2019, [arXiv 1910.01739](https://arxiv.org/abs/1910.01739)).
Required by rule 13 (CLAUDE.md). Sorted by severity: **FAIL → UNVERIFIED → PASS → N/A**.

**Rule 11 applies throughout: nothing here claims a paper result is reproduced.** The
paper's protocol is 30 replications (App. A); the runs in §D use 2–3.

Last run: 2026-08-24 · Python 3.11.0 · torch 2.13.0+cpu · gpytorch 1.15.2 · numpy 2.2.6 ·
scipy 1.13.1 · pytest 9.1.1 · CPU only, float64.
Suite: **193 tests, 193 passed** (13.0 s). Reproduce: `pytest tests/ -q`

**Mutation-tested.** 22 known-wrong implementations were injected into a throwaway copy of
`src/` and the suite re-run; **22/22 are now caught** (17 initially, of which 2 escaped; a 22nd gap surfaced
when configuration was centralized — see §G). Mutations covered: trust-region half-width, inverted lengthscale weighting,
Sobol→uniform candidates, `ddof` in standardization and in the standard error, dropped
de-standardization, growth factor 2→1.5, ARD disabled, every App. D constant
(`τ_succ`, `τ_fail` rounding, `L_init`, `L_max`, `L_min`, `n_cand` cap, perturbation
probability), `argmin`→`argmax` in centering and in Thompson selection, the App. C noise
bound, and removal of the torch RNG fork.

Status counts: **0 FAIL · 11 UNVERIFIED · 51 PASS · 5 N/A** (3 FAIL in §A, 7 review findings in §H — all fixed)

---

## A. FAIL — found by this audit, now fixed

All three were found by the audit, reproduced by a test that failed first, then fixed.
Re-audited status is PASS; the original finding is kept for the record.

| # | Requirement | Paper reference | Code location | Test that verifies it | Status | Notes |
|---|---|---|---|---|---|---|
| F1 | BFGS baseline must respect its evaluation budget; "BFGS approximates the gradient via finite differences and thus requires d+1 evaluations for each step" | §3 ¶4; App. B ("Scipy implementations of NM and BFGS", "with multiple restarts") | `src/baselines.py:65-73` | `test_scipy_budget_option_is_accepted_by_the_solver` | **PASS** (was FAIL) | **Mismatch:** `options={"maxfev": remaining}` was passed to **both** solvers, but L-BFGS-B has no `maxfev` option — it takes `maxfun`. SciPy emitted `OptimizeWarning: Unknown solver options: maxfev` and **silently ignored the budget**, so one `minimize` call could run past the remaining evaluations; the outer loop then truncated the trace, making the reported `n_evals` look correct while the evaluations were misattributed. **Fix:** select the option key per solver (`maxfun` for L-BFGS-B, `maxfev` for Nelder-Mead). **Class: environment/API compatibility fix.** Severity was medium-high — a silent budget violation in a comparison baseline. |
| F2 | Reward-style problems are negated for the minimizer | §2 (minimization); §3.1–3.4 (rewards plotted); `PAPER_SPEC.md` §10 C7 | `src/benchmarks.py:negate` | `test_negate_converts_a_reward_into_a_minimization_objective`, `test_negate_round_trips`, `test_negated_benchmark_minimum_is_the_reward_maximum` | **PASS** (was FAIL) | **Mismatch:** the module docstring claimed rewards "are negated inside their wrappers and reported as reward by evaluate.py" — neither existed. **Fix:** implemented the `negate` wrapper the convention requires, and corrected the docstring to state that `evaluate.py` reports the minimized value as-is, so a caller comparing against Fig. 2–3 must negate it. **Class: implementation bug (documented behavior absent from code).** Severity was medium — a rule-11 hazard. |
| F3 | TuRBO-m must use `τ_fail` for the sequential case (`q=1`) | App. D ("we use the same tolerances as in the sequential case (q = 1)") | `src/turbo_m.py:62-67` | `test_turbo_m_inherited_state_is_not_stale`, `test_turbo_m_every_trust_region_state_carries_the_sequential_failtol` | **PASS** (was FAIL) | **Mismatch:** `TurboM` inherited `self.state` from `Turbo1.__init__`, built **before** `self.failtol` was overwritten, so it carried the TuRBO-1 `⌈d/q⌉` (observed: 1 instead of 4) while `self.states` correctly carried `d`. Unread by the TuRBO-m path, but `evaluate.py:137` reads `state.succtol`. **Fix:** rebuild `self.state` after the tolerance is set. **Class: implementation bug (latent trap; no behavioral change today).** Severity was low. |

## B. UNVERIFIED — not checked, or not checkable from this repository

| # | Requirement | Paper reference | Code location | Test | Status | Notes |
|---|---|---|---|---|---|---|
| U1 | 200D Ackley, 10 000 evaluations, `q=100`, 200 initial points | §3.5; Fig. 4 | `src/benchmarks.py` (`ackley200`) | none | UNVERIFIED | Never run. App. G reports ~10 min/replication on an RTX 2080 Ti; no GPU here. Domain and dimension are verified; optimization behavior is not. |
| U2 | CG + Lanczos solves for large `n` | App. C ("GPyTorch follows Dong et al. 2017a … conjugate gradient … Lanczos") | `src/gp.py` (`max_cholesky_size=2000`) | none | UNVERIFIED | Every run so far stays under 2000 points, so only the exact-Cholesky path has executed. The scalability claim underpinning §3.1–3.2 is untested. |
| U3 | CMA-ES baseline: pycma, default settings, popsize = batch size | App. B | `src/baselines.py:100-137` | none | UNVERIFIED | `cma` not installed; raises a clear `ImportError`. Never executed. |
| U4 | BOBYQA baseline via nlopt | App. B | `src/baselines.py:140-170` | none | UNVERIFIED | `nlopt` not installed; raises a clear `ImportError`. Never executed. |
| U5 | Nelder-Mead / BFGS / random-search baselines | App. B | `src/baselines.py:74-103` | `tests/test_baselines.py` (7 tests) | **PASS** (was UNVERIFIED) | Now covered: budget respected, trace/best-value consistency, solver option accepted, and BFGS beats random search on Hartmann6 at equal budget. Optional-dependency baselines still U3/U4. |
| U6 | Local GPs beat a global GP (0.110 nats, `p < 1e-4`, 50 trials) | §3.6; Fig. 5 | not implemented | none | UNVERIFIED | Ablation not implemented. This is the paper's central mechanistic evidence. |
| U7 | TR volume shrinks; restarts land far apart | §3.7; Fig. 6 | not implemented | none | UNVERIFIED | Ablation not implemented. |
| U8 | Large batches give near-linear speed-up, `q ∈ {1..64}`, `max{200q, 6400}` samples | §3.8; Fig. 7 | not implemented | none | UNVERIFIED | Ablation not implemented. |
| U9 | Per-experiment budgets for the five main-text benchmarks | §3.1–3.5 (table in `PAPER_SPEC.md` §9) | `src/evaluate.py:24-32` holds **only** `SYNTHETIC_PROTOCOL` | none | UNVERIFIED | The App. A synthetic protocol is encoded; the robot-pushing / rover / cosmology / lunar / Ackley-200 settings exist only as prose in `PAPER_SPEC.md` §9. There are no config files in the repo. |
| U10 | GP hyperparameter initialization (outputscale 1.0, lengthscale 0.5, noise 0.005) | not in paper; `gp.py` L79-81 of official code | `src/gp.py:150-157` | `test_gp_hyperparameter_initialization_values` | **PASS** (was UNVERIFIED) | Values now asserted against literals, and each checked to sit inside its App. C box constraint. |
| U11 | Numerical agreement with uber-research/TuRBO | — | — | none | UNVERIFIED | Not attempted and **not expected**: this code follows the paper over the official code on `τ_fail` (×2) and the noise bound. See `REPRODUCTION_NOTES.md` §2. |
| U12 | Robot pushing, rover, lunar lander, cosmological constants | §3.1–3.4; App. F | `src/benchmarks.py:200-232` | `test_unreproducible_benchmarks_fail_loudly` | UNVERIFIED | Not reproducible from the paper — see `REPRODUCTION_NOTES.md` §6. Only the *refusal* is tested, which is the correct behavior under rule 11. |

## C. PASS — audited and verified by an executing test

### 1. Equations and mathematical operations

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| `L_i = λ_i·L / (Π λ_j)^(1/d)`, constant volume `Π L_i = L^d` | §2 "Trust regions" (E2) | `trust_region.py:trust_region_weights` | `test_volume_invariant`, `test_weights_have_unit_product` | PASS |
| Side lengths stay proportional to lengthscales; scale-invariant | §2 (E2) | `trust_region.py:trust_region_weights` | `test_weights_preserve_lengthscale_ratios`, `test_weights_are_scale_invariant` | PASS |
| `L ← min(2L, L_max)` after `τ_succ` successes | §2 (E4) | `trust_region.py:TrustRegionState.update` | `test_expand_after_tau_succ_consecutive_successes`, `test_expansion_is_capped_at_length_max` | PASS |
| `L ← L/2` after `τ_fail` failures; counters reset on resize | §2 (E4) | `trust_region.py:TrustRegionState.update` | `test_shrink_after_tau_fail_consecutive_failures`, `test_success_resets_the_failure_counter` | PASS |
| TuRBO-m adds `q_l` to the failure counter; halves on `>=` | App. D (E4) | `trust_region.py:TrustRegionState.update` | `test_failure_counter_accumulates_batch_size_for_turbo_m` | PASS |
| Terminate TR when `L < L_min` | App. D | `trust_region.py:is_converged` | `test_convergence_threshold` | PASS |
| `x_i ∈ argmin_l argmin_x f_l^(i)` (cross-TR bandit) | §2 (E5) | `thompson.py:select_candidates_across` | `test_bandit_allocates_to_the_trust_region_with_better_draws`, `test_bandit_can_split_a_batch_across_trust_regions` | PASS |
| Matérn-5/2 ARD kernel, constant mean | App. C (E8) | `gp.py:GP` | `test_ard_gives_one_lengthscale_per_dimension` | PASS |
| Known global minima of the synthetic functions | App. A | `benchmarks.py` | `test_ackley_minimum_at_origin`, `test_levy_minimum_at_ones`, `test_rastrigin_minimum_at_origin`, `test_hartmann6_known_optimum` | PASS |

### 2. Tensor dimensions and broadcasting

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| `X (n,d)`, `fX (n,1)`, GP targets `(n,)` | `PAPER_SPEC.md` §3 | `turbo_1.py`, `gp.py` | `test_stage0_fixture_shapes_dtypes_and_ranges`, `test_history_shapes`, `test_train_gp_rejects_2d_targets` | PASS |
| `X_cand (n_cand,d)`, `y_cand (n_cand,q)` | `PAPER_SPEC.md` §3 | `candidates.py`, `thompson.py` | `test_candidates_shape_and_dtype`, `test_draw_shape_is_n_cand_by_q` | PASS |
| TuRBO-m `(m,n_cand,d)` / `(m,n_cand,q)`; `idx_next (q,1)` | `PAPER_SPEC.md` §3 | `turbo_m.py:_propose_all` | `test_select_across_shapes`, `test_stage4_cross_tr_selection_executes` | PASS |
| Bounds `(1,d)`; center `(1,d)` | `PAPER_SPEC.md` §3 | `trust_region.py` | `test_bounds_are_centered_and_clipped` | PASS |
| History and TR-ownership arrays stay aligned | — | `turbo_m.py` | `test_turbo_m_history_and_ownership_stay_aligned` | PASS |
| Mismatched `train_x`/`train_y` rows rejected | — | `gp.py` | `test_train_gp_rejects_mismatched_shapes` | PASS |

### 3. Ordering of operations

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| GP refit **precedes** TR bound computation (bounds use current lengthscales) | `PAPER_SPEC.md` §5 | `turbo_1.py:_propose` (refit → center → bounds) | `test_stage4_one_turbo_iteration_executes` | PASS |
| TR counters update **before** the batch is appended | `PAPER_SPEC.md` §5; App. D | `turbo_1.py:optimize`; `turbo_m.py:optimize` | `test_optimization_improves_on_the_initial_design`, `test_trust_region_shrinks_on_a_flat_objective` | PASS |
| Only TRs with `q_l > 0` are updated | App. D | `turbo_m.py:174-183` | `test_turbo_m_improves_on_its_initial_designs` | PASS |
| Restart resets `L`, counters, and cached hypers | §2; App. D | `trust_region.py:reset` | `test_reset_restores_initial_length_and_counters` | PASS |

### 4. Normalization placement

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| Domain rescaled to `[0,1]^d` before GP fitting | App. C | `utils.py:to_unit_cube`; `turbo_1.py:_propose` | `test_unit_cube_roundtrip_is_identity`, `test_to_unit_cube_maps_bounds_to_corners` | PASS |
| Values standardized before GP fitting | App. C (E7) | `utils.py:standardize` | `test_standardize_median_matches_paper_spec_e7` | PASS |
| Draws **de-standardized** before comparison (load-bearing across TRs) | E7 | `thompson.py:thompson_draws` | `test_destandardization_is_applied`, `test_cross_tr_comparison_needs_destandardized_draws` | PASS |

### 6. Masking and boundary conditions

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| Perturbation probability `min(1, 20/d)` | App. D (E6) | `candidates.py:perturbation_probability` | `test_perturbation_probability_matches_appendix_d`, `test_high_dimension_perturbs_only_a_subset` | PASS |
| Every candidate perturbs ≥1 coordinate (empty-mask edge case) | App. D (implied) | `candidates.py:79-82` | `test_every_candidate_perturbs_at_least_one_coordinate` | PASS |
| Last dimension reachable (upstream off-by-one not reproduced) | — | `candidates.py:82` | `test_last_dimension_is_reachable_by_the_empty_mask_fix` | PASS |
| TR clipped to the domain; candidates never leave `[0,1]^d` | App. D | `trust_region.py:trust_region_bounds` | `test_bounds_clip_at_the_domain_boundary`, `test_candidates_stay_in_the_unit_cube_at_the_boundary` | PASS |
| Batch exceeding the candidate set rejected (no duplicate proposals) | — | `thompson.py` | `test_batch_larger_than_candidate_set_is_rejected`, `test_pooled_batch_larger_than_all_candidates_is_rejected` | PASS |
| All evaluated points lie inside the domain | §2 | `turbo_1.py` | `test_all_points_lie_inside_the_domain` | PASS |

### 7. Loss terms and weighting

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| Log-marginal likelihood is the only differentiated objective | App. C; `PAPER_SPEC.md` §6 (E8) | `gp.py:train_gp` | `test_stage2_marginal_log_likelihood_computes`, `test_stage2_loss_backward_populates_gradients` | PASS |
| Fitting improves the marginal likelihood | App. C | `gp.py:train_gp` | `test_fitting_improves_the_marginal_likelihood`, `test_stage3_fifty_steps_reduce_the_loss` | PASS |
| Black-box objective is never differentiated | §1 | `turbo_1.py:_evaluate` | `test_stage4_one_turbo_iteration_executes` | PASS |

### 8. Initialization

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| Latin hypercube initial design | App. A | `utils.py:latin_hypercube` | `test_latin_hypercube_is_stratified`, `test_latin_hypercube_shape_and_range` | PASS |
| TR initialized to `L_init = 0.8` | App. D | `trust_region.py:TR_DEFAULTS` | `test_reset_restores_initial_length_and_counters` | PASS |
| Every TR gets its own initial design | §3.1; App. D | `turbo_m.py:_init_tr` | `test_turbo_m_initializes_every_trust_region` | PASS |

### 9. Optimizer and scheduler

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| Hyperparameters fitted before every proposed batch | App. C | `turbo_1.py:_propose` | `test_stage4_one_turbo_iteration_executes` | PASS |
| One Adam step at lr 0.1 moves the hyperparameters | not in paper (official code) | `gp.py:159` | `test_stage3_single_adam_step_changes_hyperparameters` | PASS |
| Warm start reproduces cached hyperparameters (`num_steps=0`) | not in paper (official code) | `gp.py:150-153` | `test_warm_start_from_cached_hypers_reproduces_the_model` | PASS |
| No LR schedule exists | — | `gp.py` | — | PASS (absence is correct; paper states none) |

### 10. Data preprocessing

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| Degenerate σ does not produce NaNs | — | `utils.py:standardize` | `test_standardize_constant_values_do_not_produce_nan` | PASS |
| Mixed dtype/device rejected (float64 policy) | `REPRODUCTION_NOTES` C14 | `gp.py:119-129` | `test_mixed_dtype_training_inputs_are_rejected` | PASS |
| Domains match the paper for all five synthetics | App. A; §3.5 | `benchmarks.py:SYNTHETIC_SUITE` | `test_domains_match_the_paper` | PASS |

### 12. Train/eval behavior

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| GP in `train()` during fitting, `eval()` for sampling | — | `gp.py:169-171` | `test_posterior_shapes`, `test_stage1_forward_pass_on_unseen_candidates` | PASS |
| Posterior interpolates its training data | — | `gp.py` | `test_posterior_interpolates_training_data` | PASS |

### 13. Metrics

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| Best-so-far trace is monotone non-increasing | §3 (convergence plots) | `evaluate.py:best_so_far` | `test_best_so_far_is_monotone_non_increasing` | PASS |
| Mean ± **one standard error** across replications | §3 ("mean performances with one standard error") | `evaluate.py:mean_standard_error` | `test_mean_standard_error_matches_the_definition`, `test_mean_standard_error_single_replication_has_zero_sem` | PASS |
| `best_point` is consistent with `best_value` | — | `turbo_1.py` | `test_best_point_matches_best_value` | PASS |
| Settings recorded alongside results (rule 11 checkability) | — | `evaluate.py:RunResult` | `test_run_replications_records_the_settings_used` | PASS |

### 14. Random seeds and determinism

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| A run is fully reproducible from one seed (NumPy **and** forked torch RNG) | not in paper (`PAPER_SPEC.md` §10 C5) | `utils.py:as_generator`; `thompson.py:_draw` | `test_runs_are_reproducible_from_a_seed`, `test_stage5_run_is_deterministic_under_a_fixed_seed` | PASS |
| Different seeds give different trajectories | — | — | `test_different_seeds_give_different_runs` | PASS |
| Initial design reproducible from a seed | — | `utils.py:latin_hypercube` | `test_latin_hypercube_is_reproducible_from_seed` | PASS |
| A fresh candidate set is drawn per batch | App. D | `candidates.py` | `test_a_new_candidate_set_is_drawn_each_call` | PASS |

### 15. Hyperparameters

| Requirement | Paper reference | Code location | Test | Status |
|---|---|---|---|---|
| `τ_succ = 3`, `L_min = 2^-7`, `L_max = 1.6`, `L_init = 0.8` | App. D | `trust_region.py:TR_DEFAULTS` | `test_convergence_threshold`, `test_expansion_is_capped_at_length_max` | PASS |
| `τ_fail = ⌈d/q⌉` (TuRBO-1); sequential case for TuRBO-m | App. D | `trust_region.py:default_failtol` | `test_failtol_follows_the_paper_for_turbo_1`, `test_failtol_for_turbo_m_uses_the_sequential_case` | PASS |
| `n_cand = min(100d, 5000)` | App. D | `candidates.py:n_candidates` | `test_candidate_set_size_matches_appendix_d`, `test_candidate_set_size_follows_appendix_d` | PASS |
| λ∈[0.005,2.0], s²∈[0.05,20.0], σ²∈[0.0005,0.1] | App. C | `gp.py:HYPER_BOUNDS` | `test_hyper_bounds_match_appendix_c`, `test_fitted_hyperparameters_respect_the_box_constraints` | PASS |
| App. A protocol: 500 evals, `q=10`, 20 init (10/TR for TuRBO-5), 30 reps | App. A | `evaluate.py:SYNTHETIC_PROTOCOL` | `test_run_replications_records_the_settings_used` | PASS |
| `max_evals` inclusive of the initial design | not in paper (official code) | `turbo_1.py` | `test_budget_is_never_exceeded`, `test_turbo_m_respects_the_budget` | PASS |

## D. Partial experimental runs — NOT reproductions

App. A settings but **2–3 replications instead of 30**. No comparison to Fig. 8 is claimed.

| Benchmark | Algorithm | Reps | Mean best | ± 1 SE | Known optimum | Status |
|---|---|---:|---:|---:|---:|---|
| Hartmann6 | TuRBO-1 | 3 | −3.3208 | 0.0005 | −3.32237 | ⚠️ partial — within 4e-4 of the optimum |
| Ackley-10 | TuRBO-1 | 3 | 0.3651 | 0.1637 | 0.0 | ⚠️ partial — consistent with App. A prose |
| Ackley-10 | TuRBO-5 | 2 | 0.4385 | 0.1735 | 0.0 | ⚠️ partial — TuRBO-m path runs at protocol scale |

Runtimes 36–57 s on CPU, in line with App. G's "<1 min" for the synthetic suite (different
hardware, and App. G used the full 30-replication protocol).

**What would make these reproductions:** 30 replications on all four App. A synthetics plus
Ackley-200 (§3.5), plotted as mean ± one standard error against Fig. 8 / Fig. 4.

## E. N/A — categories that do not apply to this paper

TuRBO is a Bayesian-optimization algorithm; it has no neural network.

| Category | Why N/A |
|---|---|
| 5. Residual connections | No network. The only "skip"-like structure is the GP's constant mean, which is additive by construction (`gp.py:GP.forward`). |
| 11. Augmentation | No training dataset. Data arrives from sequential objective evaluations; the nearest analogue is candidate-set perturbation (§6 above), which is an acquisition device, not augmentation. |
| 4. Normalization *layers* | No BatchNorm/LayerNorm. Normalization here is input rescaling and target standardization, audited in §4 above. |
| 3. Layer ordering | No layers. Operation ordering is audited in §3 above. |
| 7. Loss weighting | Exactly one loss term (marginal likelihood); nothing to weight. |

## F. Defects found and fixed in earlier passes

| # | Defect | Found by | Resolution |
|---|---|---|---|
| 1 | Identically seeded runs diverged — GPyTorch `.sample()` reads torch's **global** RNG, unreached by the threaded NumPy generator | `test_runs_are_reproducible_from_a_seed` | `thompson_draws` forks and seeds the torch RNG per draw |
| 2 | Bare `pytest tests/…` failed with `ModuleNotFoundError: No module named 'src'` — the rule-9 command documented in CLAUDE.md | running the documented command | Added root `conftest.py` |
| 3 | Test expectation wrong, not the code: flat objective at `d=3, q=5` needs exactly 40 evals to collapse `L` | `test_trust_region_shrinks_on_a_flat_objective` | Raised the test budget to 70 |
| 4 | Silent duplicate proposals when `q > n_cand` (all-`inf` argmin returns index 0) | edge-configuration probe | `select_candidates`/`_across` now raise `ValueError` |
| 5 | Silent dtype downcast in `train_gp` (model follows `train_x`, likelihood follows `train_y`) | dtype-mismatch probe | dtype/device equality assertions |
| 6 | **F1** — L-BFGS-B budget silently ignored (`maxfev` vs `maxfun`) | this audit; `test_scipy_budget_option_is_accepted_by_the_solver` | per-solver option key (`src/baselines.py:70`) |
| 7 | **F2** — docstring promised a reward path with no implementation | this audit; `test_negate_*` | added `negate`; corrected the docstring |
| 8 | **F3** — `TurboM` carried a stale single-TR state with the TuRBO-1 `failtol` | this audit; `test_turbo_m_inherited_state_is_not_stale` | rebuild `self.state` after the tolerance is set (`src/turbo_m.py:66`) |

## H. Independent adversarial review (findings, all fixed)

A reviewer pass assuming the implementation was subtly wrong despite passing tests. The
**core TuRBO algorithm was confirmed correct** — `Turbo1._propose` fits on trust-region-local
data (`turbo_1.py:145`), and trust-region geometry, the success/failure schedule, per-TR
counters and the cross-TR bandit all match §2/App. D. No CRITICAL finding. Every defect was
in the baseline comparison path — the module that had zero test coverage until recently.

| # | Sev | Finding | Paper ref | Fix | Test |
|---|---|---|---|---|---|
| H1 | **HIGH** | `_initial_best` evaluated the objective directly, bypassing the counter, and was redrawn per restart. Measured: BFGS made **52 real calls for a stated budget of 30** (71 at `n_init=20`). Since §3 compares per evaluation, this biased the comparison toward the baselines — BFGS scored −3.08 with 52 evals vs **−2.05** with an honest 30. | §3; App. B ("initialized from the best of a few initial points") | Charge the design through the counted objective, cap by remaining budget, enforce hard via `_BudgetExhausted` (solvers check caps only between iterations, and a finite-difference gradient is atomic) | `test_baseline_objective_calls_match_the_stated_budget`, `test_restart_initial_designs_do_not_inflate_the_budget` |
| H2 | MEDIUM | The budget test asserted `result["n_evals"]` — the *truncated* trace length, which cannot exceed the budget by construction. A tautology, and the reason H1 survived 184 tests and a 22-mutation sweep. | — | Assert real objective calls via a counting benchmark | as above |
| H3 | MEDIUM | Nelder-Mead ran with `bounds=None` and relied on clipping inside the objective, so the simplex moved through out-of-box space while being scored on clipped points; distinct vertices collapse to the same point, wasting budget and creating artificial flat regions. | §2 (domain `Ω`); App. B | Pass the box to both solvers | `test_nelder_mead_searches_only_inside_the_domain` (0 out-of-bounds, was non-zero) |
| H4 | MEDIUM | `select_candidates`/`_across` mutated the caller's `y_cand` in place, while `LocalProposal` advertises `frozen=True`. Later inspection would read `+inf` garbage that does not look obviously wrong. | — | Copy on entry (~2 MB even at rover settings) | `test_selection_does_not_corrupt_the_callers_draws` (+ cross-TR variant) |
| H5 | LOW | The noisy centering path emitted `GPInputWarning` on every batch — values correct, but it trains reviewers to ignore warnings. | §2 (noisy centering) | Silence that one warning at the call site, with the reason | warning count 8 → 7 |
| H6 | LOW | A restart near the end of the budget draws a truncated Latin hypercube, losing stratification (degenerate at `n_init=1`). | App. A (LHD) | Documented as `REPRODUCTION_NOTES.md` C10a | — |
| H7 | LOW | `summarize` deep-copied every trace via `asdict` to read one small dict. | — | `dict(results[0].settings)` | — |

**Core untouched:** golden trajectories captured before the refactor replay **bit-identical**
after all seven fixes, across TuRBO-1 (default), TuRBO-1 (all four behavioral flags flipped)
and TuRBO-m (m=3), comparing `X`, `fX` and `_idx`.

## G2. Behavior-preservation check for the readability refactor

The refactor (config centralization, extraction of the shared per-region surrogate step
into `src/local_model.py`, type hints) was verified beyond "tests still pass": golden
trajectories captured **before** the change were replayed after it.

| Configuration | Arrays compared | Result |
|---|---|---|
| TuRBO-1, default flags | `X`, `fX` (46 evals) | bit-identical |
| TuRBO-1, all four behavioral flags flipped (`noisy`, `center_stat="mean"`, `use_predictive=False`, `success_tol=0.0`) | `X`, `fX` (34 evals) | bit-identical |
| TuRBO-m, `m=3` | `X`, `fX`, `_idx` (54 evals) | bit-identical |

One behavior-changing slip was caught during the refactor and fixed before it landed: the
Thompson draw seed is drawn as `rng.integers(0, 2**31 - 1)`, and an early draft routed it
through the Sobol constant (`1e6`). A different range yields a different seed and therefore
different trajectories — the same algorithm, but not the same run. `config.TORCH_SEED_MAX`
now holds `2**31 - 1` separately from `config.SOBOL_SEED_MAX`.

## G. Test-suite gaps found by mutation testing

Two injected mutations initially escaped a 138-test suite. Both were **test gaps, not code
defects** — the code was correct, but nothing would have noticed if it stopped being so.

| Escaped mutation | Why it escaped | Closed by |
|---|---|---|
| `τ_succ: 3 → 2` (App. D) | Every trust-region test passes `succtol` in explicitly, so the default was never exercised | `test_trust_region_defaults_match_appendix_d_literally`, `test_a_default_trust_region_state_starts_at_appendix_d_values`, `test_optimizer_adopts_appendix_d_defaults_without_overrides` |
| `L_init: 0.8 → 0.5` (App. D) | Same, plus `test_convergence_threshold` reads `TR_DEFAULTS["length_min"]` to assert `TR_DEFAULTS["length_min"]` — a self-referential assertion that holds under any value | same three tests, which assert paper literals (`0.8`, `1.6`, `0.0078125`) rather than reading the constants back |

`L_max` and `L_min` were unpinned for the same reason and are now covered; all four App. D
constants are confirmed caught on re-run.

| `ADAM_LR: 0.1 → 0.5` (FROM_OFFICIAL_CODE) | Surfaced when configuration was centralized into `src/config.py`. Nothing pinned `train_gp`'s `lr` default: the stage-3 smoke test builds its own `torch.optim.Adam` with an explicit learning rate, so the default was never exercised | `test_gp_fitting_constants_match_the_official_code` |

That row is a pre-existing gap made *visible* by centralization, not caused by it.
