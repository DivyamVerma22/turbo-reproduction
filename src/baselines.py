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


def _initial_best(bench: Benchmark, n_init: int, rng) -> np.ndarray:
    """App. B: baselines are "initialized from the best of a few initial points"."""
    X = from_unit_cube(latin_hypercube(n_init, bench.dim, rng), bench.lb, bench.ub)
    fX = np.array([bench(x) for x in X])
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

    def wrapped(x):
        x = np.clip(x, bench.lb, bench.ub)
        v = bench(x)
        history.append(v)
        return v

    best = np.inf
    while len(history) < max_evals:
        x0 = _initial_best(bench, min(n_init, max(1, max_evals - len(history))), rng)
        if len(history) >= max_evals:
            break
        remaining = max_evals - len(history)
        # SciPy spells the function-evaluation cap differently per solver: Nelder-Mead
        # takes `maxfev`, L-BFGS-B takes `maxfun`. Passing the wrong key makes SciPy emit
        # OptimizeWarning and silently ignore the budget, so the solver runs past the
        # remaining evaluations (§3: the comparison is per evaluation).
        budget_key = "maxfun" if method == "L-BFGS-B" else "maxfev"
        res = minimize(
            wrapped,
            x0,
            method=method,
            bounds=list(zip(bench.lb, bench.ub)) if method != "Nelder-Mead" else None,
            options={budget_key: remaining, **opts},
        )
        best = min(best, float(res.fun))
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
    x0 = _initial_best(bench, n_init, rng)
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
    history: list[float] = []
    while len(history) < max_evals and not es.stop():
        solutions = es.ask()
        values = [bench(np.asarray(s)) for s in solutions]
        es.tell(solutions, values)
        history.extend(values)
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

    def objective(x, _grad):
        v = bench(np.asarray(x))
        history.append(v)
        return v

    opt = nlopt.opt(nlopt.LN_BOBYQA, bench.dim)
    opt.set_lower_bounds(bench.lb)
    opt.set_upper_bounds(bench.ub)
    opt.set_min_objective(objective)
    opt.set_maxeval(max_evals)
    opt.optimize(list(_initial_best(bench, n_init, rng)))
    fX = np.array(history[:max_evals], dtype=np.float64)
    return {"best_value": float(fX.min()), "n_evals": len(fX), "fX": fX}
