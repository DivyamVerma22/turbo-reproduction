"""Optional baseline wrappers for the comparison plots (App. B).

App. B, verbatim:
  "For local optimization, we use the popular NM, BOBYQA, and BFGS methods with multiple
   restarts. They are all initialized from the best of a few initial points. We use the
   Scipy [24] implementations of NM and BFGS and the nlopt [23] implementation of BOBYQA."
  "We compare to CMA-ES [19] ... We use the pycma implementation with the default settings
   and a population size equal to the batch size. The population is initialized from the
   best of a few initial points."

Scope note (PAPER_SPEC.md §2): EBO, BOCK, BOHAMIANN and HeSBO-TS are deliberately NOT
implemented here. They are third-party baselines that App. B says required modification to
run in this setting, and they are not part of the contribution.

Every wrapper below MINIMIZES, matching src.benchmarks. Optional dependencies (`cma`,
`nlopt`) are imported lazily so the core package works without them.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .benchmarks import Benchmark
from .utils import as_generator, from_unit_cube, latin_hypercube

__all__ = ["random_search", "nelder_mead", "bfgs", "cma_es", "bobyqa"]


class _BudgetExhausted(Exception):
    """Raised inside a counted objective once the evaluation budget is spent.

    Solvers check their own evaluation caps only between iterations, and a
    finite-difference gradient is atomic (Sect. 3: "BFGS approximates the gradient via
    finite differences and thus requires d+1 evaluations for each step"), so `maxfun`
    alone lets a solver overshoot by up to d+1. §3 compares methods per evaluation, so the
    budget is enforced hard: the objective refuses the call that would exceed it and the
    caller unwinds.
    """


def _counted(bench: Benchmark, history: list[float], max_evals: int, clip: bool = False):
    """Wrap `bench` so every call is recorded and the budget is strictly enforced."""

    def objective(x):
        if len(history) >= max_evals:
            raise _BudgetExhausted
        x = np.asarray(x, dtype=np.float64)
        if clip:
            x = np.clip(x, bench.lb, bench.ub)
        v = bench(x)
        history.append(v)
        return v

    return objective


def _initial_best(bench: Benchmark, n_init: int, rng, objective=None) -> np.ndarray:
    """App. B: baselines are "initialized from the best of a few initial points".

    Args:
        objective: the counted callable the caller uses to record evaluations. These
            initial points are real objective calls and MUST be charged to the budget --
            §3 compares every method per evaluation (Figs. 2-4). Evaluating `bench`
            directly here would hand the local baselines free evaluations, which measured
            at 1.7-2.4x their stated budget for BFGS.
    """
    objective = bench if objective is None else objective
    X = from_unit_cube(latin_hypercube(n_init, bench.dim, rng), bench.lb, bench.ub)
    fX = np.array([objective(x) for x in X])
    return X[int(np.argmin(fX))]


def random_search(bench: Benchmark, max_evals: int, seed=None) -> dict:
    """Uniform random search (RS baseline, Sect. 3)."""
    rng = as_generator(seed)
    X = from_unit_cube(rng.random((max_evals, bench.dim)), bench.lb, bench.ub)
    fX = np.array([bench(x) for x in X])
    return {"best_value": float(fX.min()), "n_evals": max_evals, "fX": fX}


def _scipy_with_restarts(bench, max_evals, method, seed, n_init, **opts) -> dict:
    """App. B: "with multiple restarts", each "initialized from the best of a few initial points"."""
    rng = as_generator(seed)
    history: list[float] = []
    wrapped = _counted(bench, history, max_evals, clip=True)

    best = np.inf
    while len(history) < max_evals:
        # The initial design is charged to the budget via `wrapped`, and capped by what
        # remains so a restart can never overshoot.
        n_start = min(n_init, max_evals - len(history))
        if n_start <= 0:
            break
        try:
            x0 = _initial_best(bench, n_start, rng, objective=wrapped)
        except _BudgetExhausted:
            break
        if len(history) >= max_evals:
            break
        remaining = max_evals - len(history)
        # SciPy spells the function-evaluation cap differently per solver: Nelder-Mead
        # takes `maxfev`, L-BFGS-B takes `maxfun`. Passing the wrong key makes SciPy emit
        # OptimizeWarning and silently ignore the budget, so the solver runs past the
        # remaining evaluations (§3: the comparison is per evaluation).
        budget_key = "maxfun" if method == "L-BFGS-B" else "maxfev"
        try:
            res = minimize(
                wrapped,
                x0,
                method=method,
                bounds=list(zip(bench.lb, bench.ub)),
                options={budget_key: remaining, **opts},
            )
            best = min(best, float(res.fun))
        except _BudgetExhausted:
            break
    fX = np.array(history[:max_evals], dtype=np.float64)
    return {"best_value": float(fX.min()), "n_evals": len(fX), "fX": fX}


def nelder_mead(bench: Benchmark, max_evals: int, seed=None, n_init: int = 5) -> dict:
    """Nelder-Mead with restarts (App. B, SciPy implementation)."""
    return _scipy_with_restarts(bench, max_evals, "Nelder-Mead", seed, n_init)


def bfgs(bench: Benchmark, max_evals: int, seed=None, n_init: int = 5) -> dict:
    """L-BFGS-B with restarts (App. B).

    Sect. 3: "BFGS approximates the gradient via finite differences and thus requires d+1
    evaluations for each step" -- SciPy does exactly this when no jac is supplied.
    """
    return _scipy_with_restarts(bench, max_evals, "L-BFGS-B", seed, n_init)


def cma_es(bench: Benchmark, max_evals: int, batch_size: int, seed=None, n_init: int = 5) -> dict:
    """CMA-ES via pycma (App. B: "default settings and a population size equal to the batch size").

    # [UNSPECIFIED] App. B does not state the initial step size sigma0. Using 0.3 of the
    # domain width. Alternatives: 0.2 or 0.5 of the width; pycma's own default.
    """
    try:
        import cma  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "cma_es requires pycma (App. B uses https://github.com/CMA-ES/pycma):\n"
            "    pip install cma"
        ) from exc

    rng = as_generator(seed)
    history: list[float] = []
    counted = _counted(bench, history, max_evals)
    x0 = _initial_best(bench, min(n_init, max_evals), rng, objective=counted)
    sigma0 = 0.3 * float(np.mean(bench.ub - bench.lb))
    es = cma.CMAEvolutionStrategy(
        list(x0),
        sigma0,
        {
            "popsize": batch_size,  # App. B
            "bounds": [list(bench.lb), list(bench.ub)],
            "seed": int(rng.integers(1, 2**31 - 1)),
            "verbose": -9,
        },
    )
    try:
        while len(history) < max_evals and not es.stop():
            solutions = es.ask()
            values = [counted(np.asarray(s)) for s in solutions]
            es.tell(solutions, values)
    except _BudgetExhausted:
        pass
    fX = np.array(history[:max_evals], dtype=np.float64)
    return {"best_value": float(fX.min()), "n_evals": len(fX), "fX": fX}


def bobyqa(bench: Benchmark, max_evals: int, seed=None, n_init: int = 5) -> dict:
    """BOBYQA via nlopt (App. B).

    nlopt is an optional dependency and is not installed by default.
    """
    try:
        import nlopt  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "bobyqa requires nlopt (App. B uses the nlopt implementation):\n"
            "    pip install nlopt"
        ) from exc

    rng = as_generator(seed)
    history: list[float] = []

    counted = _counted(bench, history, max_evals)

    def objective(x, _grad):
        return counted(x)

    opt = nlopt.opt(nlopt.LN_BOBYQA, bench.dim)
    opt.set_lower_bounds(bench.lb)
    opt.set_upper_bounds(bench.ub)
    opt.set_min_objective(objective)
    try:
        x0 = _initial_best(bench, min(n_init, max_evals), rng, objective=counted)
        opt.set_maxeval(max(0, max_evals - len(history)))
        opt.optimize(list(x0))
    except _BudgetExhausted:
        pass
    fX = np.array(history[:max_evals], dtype=np.float64)
    return {"best_value": float(fX.min()), "n_evals": len(fX), "fX": fX}
