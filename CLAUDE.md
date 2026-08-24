# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A from-scratch reproduction of **TuRBO** — *Scalable Global Optimization via Local Bayesian
Optimization* (Eriksson et al., NeurIPS 2019, [arXiv 1910.01739](https://arxiv.org/abs/1910.01739)).

The package lives in `src/` (module map: `PAPER_SPEC.md` §2), tests in `tests/`. TuRBO-1, TuRBO-m and
the synthetic benchmarks are implemented; the four real-world benchmarks from §3 are declared but raise
`NotImplementedError`, because the paper does not specify them (`REPRODUCTION_NOTES.md` §6).

## Research reproduction rules

These govern all work in this repository.

1. `PAPER_SPEC.md` is the implementation contract.
2. The research paper is the source of truth.
3. Never silently invent a hyperparameter or algorithmic detail.
4. Mark unresolved choices as `[UNSPECIFIED]`, `[PARTIALLY_SPECIFIED]`, or `[ASSUMPTION]`.
5. Keep the relevant paper section/equation reference next to the corresponding code.
6. Before changing behavior, explain which paper requirement the change satisfies.
7. Keep functions small and testable.
8. Add tests for tensor shapes, invariants, losses, and edge cases.
9. Run the smallest relevant test after every meaningful change.
10. Do not optimize performance until correctness tests pass.
11. Do not claim a paper result is reproduced unless the exact experiment was actually run.
12. Maintain `REPRODUCTION_NOTES.md` with deviations from the paper.
13. Maintain `VERIFICATION.md` with checks performed and their status.

**One reconciliation you need to know about.** Rule 4 names three inline tags, but `PAPER_SPEC.md` §10
classifies decisions with **five** statuses — the three above plus `SPECIFIED` and
`FROM_OFFICIAL_CODE`. The fifth is not optional here: a large share of this reproduction's concrete
choices come from the official implementation rather than the paper, and collapsing them into
`[ASSUMPTION]` would lose the distinction between "the authors' code does this" and "I picked this."
Use `[FROM_OFFICIAL_CODE]` inline as well, and never let it become `SPECIFIED` — the official code is
not the paper, and it contradicts the paper in six known places (below).

An `[UNSPECIFIED]` comment carries three parts: what the paper omits, the choice made, and the
alternatives. "Standard practice" is not a justification.

Rules 12 and 13 are served by `REPRODUCTION_NOTES.md` (every deviation and undetermined choice, with the
flag that restores paper-literal behavior) and `VERIFICATION.md` (checks run, their status, and what is
explicitly *not* verified). Update both in the same commit as any behavior change — a deviation that
only exists in a code comment is not recorded.

## PAPER_SPEC.md is the contract

Derived from a full read of the paper (main text §1–§4 plus appendices A–G) and of the official
reference implementation. Read it before writing or changing implementation code — in particular §4
(equations → pseudocode), §5 (execution order), §9 (hyperparameters with per-value sources), and §10
(the ambiguity table).

Two facts about the paper that shape the work: it contains **no numbered equations and no algorithm
box** — every formula is inline prose, so §4 quotes source text instead of citing equation numbers that
do not exist. Cite sections the same way (e.g. `# §2 "Trust regions"` / `# App. D`) per rule 5.

## The six paper ↔ official-code conflicts

The highest-risk items in the reproduction and the most likely source of a silent mismatch. Evidence and
line citations in `PAPER_SPEC.md` §10.A; short form:

| Item | Paper | Official code |
|---|---|---|
| "Success" definition | any improvement on `x*` | requires a relative margin of `1e-3·abs(f_best)` |
| `τ_fail` | `⌈d/q⌉` | `⌈max(4/q, d/q)⌉` (TuRBO-1); `max(5, d)` (TuRBO-m) |
| TR center under noise | smallest posterior mean | always `argmin(fX)` — the noisy rule is never implemented |
| Value standardization | "standardized" (implies mean) | centers on the **median** |
| Noise variance bound | `[5e-4, 0.1]` | `[5e-4, 0.2]` |
| Thompson draws | latent posterior `f ~ GP(μ, k)` | the **predictive** distribution, i.e. with σ² noise added |

Each of these is a rule-6 decision point: whichever side you implement, state which paper requirement it
satisfies and record it in `REPRODUCTION_NOTES.md`.

Also: the reference implementation's empty-perturbation-mask fallback uses `randint(0, d-1)`, which can
never select the last dimension. That is an upstream off-by-one — do not copy it.

## Architecture: the parts that need multiple files to understand

TuRBO has no neural network. The "architecture" is a per-batch control flow whose ordering constraints
are load-bearing and easy to violate **without any error being raised**. `PAPER_SPEC.md` §5 has the full
loops; these are the cross-module couplings, and each is a rule-8 invariant test worth writing:

- **The GP must be refit before the trust-region bounds are computed.** TR side lengths derive from the
  *current* fitted ARD lengthscales (`L_i = λ_i·L / (Π λ_j)^(1/d)`). Computing bounds first silently
  reuses stale geometry from the previous batch. Testable invariant: `Π L_i == L^d` before clipping.
- **TR success/failure counters must update against each TR's best value before the new batch is
  appended to history.** Append first and every batch trivially reads as a non-improvement, so trust
  regions only ever shrink.
- **De-standardizing Thompson draws is load-bearing in TuRBO-m, not cosmetic.** Within one trust region
  the affine transform is order-preserving, so omitting it looks harmless. But the cross-TR bandit
  compares draws from `m` GPs that each fitted their own `(mu, sigma)` — un-de-standardized draws make
  that comparison meaningless while still running cleanly.
- **TuRBO-1 and TuRBO-m are not two algorithms.** TuRBO-m subclasses TuRBO-1 upstream and overrides only
  the counters, the candidate selection (joint argmin across TRs), and the restart path. Keep the shared
  candidate-generation path genuinely shared.
- **Restarts consume budget outside the `q`-per-iteration accounting**, so total evaluations can
  overshoot `max_evals` by up to `n_init`. Whatever the eval driver reports must be actual `n_evals`.

Planned module layout is `PAPER_SPEC.md` §2; per-module tensor shapes and dtypes are §3. Everything is
`float64`.

## Environment

Setup from a clean clone is in `CLEAN_SETUP_CHECKLIST.md`. Summary:

- Python >= 3.10; versions are pinned in `requirements.txt` (CPU-only torch is sufficient — nothing
  in `src/` requests a GPU).
- `cma` and `nlopt` are **optional** and not installed by default; `src/baselines.py` imports them
  lazily, so CMA-ES and BOBYQA raise a clear `ImportError` naming the package rather than failing at
  import time.
- The paper specifies GPyTorch with CG solves + Lanczos log-determinants (App. C). An exact-Cholesky GP
  is conceptually equivalent but numerically different at large `n` — a rule-12 deviation if you use one.
- No environment variables, no network access, and no external data files are needed to run the tests
  or the reproduction script.
- Default branch is `master`.

## Testing and verification

```bash
pytest tests/ -q                                              # full suite (193 tests, ~13 s)
pytest tests/test_trust_region.py -q                          # one module
pytest tests/test_trust_region.py::test_volume_invariant -q   # rule 9: the smallest relevant test
pytest tests/ -q -k "thompson or bandit"                      # by keyword
```

A root `conftest.py` puts the repo root on `sys.path`; without it a bare `pytest` fails to import `src`.
There is no lint or build tooling — the package is pure source, installed only via `requirements.txt`.

Record each check and its status in `VERIFICATION.md` (rule 13), including what you did *not* verify.

Rule 11 has teeth in this repo. Four of the paper's benchmarks are **not reproducible from the paper
alone**: cosmological constants (the extra parameters and bounds are never stated), robot pushing and
rover (external simulators from Wang et al. 2018), and lunar lander (the fixed set of 50 seeds is
unpublished). The reproducible targets are the synthetics in `PAPER_SPEC.md` §8–§9 — Ackley, Levy,
Rastrigin (10D) and Hartmann6, 500 evaluations at `q=10` with 20 initial points, 30 replications,
reported as mean ± one standard error.

## Reproducing against the paper

```bash
python scripts/reproduce_minimal.py          # App. A protocol: 30 replications, ~10 min CPU
python scripts/reproduce_minimal.py --replications 5   # faster; flagged as a deviation
```

The cheapest experiment that meaningfully validates the implementation: TuRBO-1 vs. random
search / Nelder-Mead / BFGS on Ackley-10 under App. A's pinned protocol (500 evaluations,
`q=10`, 20 initial LHD points, 30 runs, mean ± one standard error). It logs seed and
environment, writes raw per-replication metrics and best-so-far traces to `results/`, and
labels the outcome **MATCH / PARTIAL MATCH / NOT REPRODUCED**.

Read the verdict narrowly. App. A reports Figure 8 as a *plot* and states no numeric value
for Ackley-10, so a MATCH means "consistent with the paper's claims", never "reproduces the
paper's numbers". The "close to the global optimum" threshold is this repo's
operationalization of App. A's wording, declared in `CONFIG`, not a paper-stated quantity.

Every precise number the paper does report — robot pushing's 9.4 (§3.1), rover's ~2 (§3.2),
§3.6's 1.284 vs 1.174 nats — depends on external simulators the paper does not specify, so
none of them is checkable here. See `REPRODUCTION_NOTES.md` §6.

## Regenerating the paper artifacts

`.paper2code_work/` is untracked scratch that held the parsed paper and a copy of the official
reference source. **It is not in the repository** and was produced by a local authoring tool that a
cloner will not have. Nothing in `src/`, `tests/` or `scripts/` depends on it — it exists only to
re-check quotations.

To recreate the equivalent by hand: download `https://arxiv.org/abs/1910.01739` and, for the official
code referenced throughout `REPRODUCTION_NOTES.md`, `https://github.com/uber-research/TuRBO`. Every
citation in `PAPER_SPEC.md` names its section or `file:line`, so quotations can be verified directly
against those two sources without any tooling.

PDF extraction of this paper fails its quality check — `pdfplumber` loses inter-word spacing — and the
fetcher falls back to ar5iv HTML. That fallback is the good text. When verifying a quote against the
PDF, expect words to be run together.
