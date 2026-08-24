"""Baseline optimizers (App. B).

App. B: "For local optimization, we use the popular NM, BOBYQA, and BFGS methods with
multiple restarts. They are all initialized from the best of a few initial points. We use
the Scipy [24] implementations of NM and BFGS and the nlopt [23] implementation of BOBYQA."

§3: "BFGS approximates the gradient via finite differences and thus requires d+1
evaluations for each step" -- the comparison in Fig. 2-4 is per evaluation, so each
baseline must respect its evaluation budget.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeWarning

from src import baselines as B
from src.benchmarks import Benchmark, get_benchmark

MAX_EVALS = 30


@pytest.fixture(scope="module")
def bench():
    return get_benchmark("hartmann6")


def counting_benchmark(dim: int = 6):
    """A benchmark that records every objective call, to audit true budget use."""
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return float(np.sum((np.asarray(x, dtype=float) - 0.3) ** 2))

    return Benchmark("counted", f, np.zeros(dim), np.ones(dim), dim, 0.0), calls


@pytest.mark.parametrize("name", ["random_search", "nelder_mead", "bfgs"])
def test_baseline_runs_and_reports_a_consistent_trace(bench, name):
    result = getattr(B, name)(bench, MAX_EVALS, seed=0)
    assert np.isfinite(result["best_value"])
    assert result["fX"].shape == (result["n_evals"],)
    assert result["best_value"] == pytest.approx(result["fX"].min())


@pytest.mark.parametrize("name", ["random_search", "nelder_mead", "bfgs"])
def test_baseline_objective_calls_match_the_stated_budget(name):
    """Protects: §3 compares every method PER EVALUATION (Figs. 2-4).

    `result["n_evals"]` is the truncated length of the recorded trace, so asserting on it
    is a tautology -- it cannot exceed the budget by construction. The quantity that
    matters is how many times the objective was actually called: App. B initializes the
    local methods "from the best of a few initial points", and those points cost budget.
    """
    b, calls = counting_benchmark()
    getattr(B, name)(b, MAX_EVALS, seed=0)
    assert calls["n"] <= MAX_EVALS, (
        f"{name} used {calls['n']} objective calls for a stated budget of {MAX_EVALS}"
    )


@pytest.mark.parametrize("n_init", [5, 10, 20])
def test_restart_initial_designs_do_not_inflate_the_budget(n_init):
    """The initial design is redrawn on every restart, so an uncounted design inflates the
    true budget without bound as restarts accumulate."""
    b, calls = counting_benchmark()
    B.bfgs(b, MAX_EVALS, seed=0, n_init=n_init)
    assert calls["n"] <= MAX_EVALS, (
        f"n_init={n_init}: {calls['n']} calls for a budget of {MAX_EVALS}"
    )


@pytest.mark.parametrize("method,name", [("Nelder-Mead", "nelder_mead"), ("L-BFGS-B", "bfgs")])
def test_scipy_budget_option_is_accepted_by_the_solver(bench, method, name):
    """The evaluation budget must actually reach SciPy.

    L-BFGS-B has no `maxfev` option -- it takes `maxfun`. Passing the wrong key makes SciPy
    emit OptimizeWarning and silently ignore the budget, so one `minimize` call can run
    past the remaining evaluations while the outer loop truncates the trace afterwards.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", OptimizeWarning)
        result = getattr(B, name)(bench, MAX_EVALS, seed=0)
    assert result["n_evals"] <= MAX_EVALS


def test_nelder_mead_searches_only_inside_the_domain():
    """Protects: §2 defines the problem on the box Omega; App. B uses SciPy's NM.

    NM was previously called with `bounds=None` and the objective clipped instead, so the
    simplex moved through out-of-box space while being scored on clipped points. Distinct
    vertices then collapse onto the same clipped point, wasting budget and creating
    artificial flat regions that stall the simplex.
    """
    seen_outside = []

    def f(x):
        x = np.asarray(x, dtype=float)
        if np.any(x < -1e-12) or np.any(x > 1 + 1e-12):
            seen_outside.append(x.copy())
        return float(np.sum((x - 0.3) ** 2))

    b = Benchmark("boxed", f, np.zeros(6), np.ones(6), 6, 0.0)
    B.nelder_mead(b, 40, seed=0)
    assert not seen_outside, f"{len(seen_outside)} evaluations fell outside the domain"


def test_local_baselines_beat_random_search(bench):
    """App. B calls NM/BFGS "popular local optimization" methods; on the easy Hartmann6
    benchmark (App. A: "much easier and most methods converge quickly") they should at
    least beat uniform random search from the same budget."""
    rs = B.random_search(bench, MAX_EVALS, seed=0)["best_value"]
    bf = B.bfgs(bench, MAX_EVALS, seed=0)["best_value"]
    assert bf < rs


def test_optional_dependency_baselines_fail_with_a_clear_message(bench):
    """cma and nlopt are optional (App. B names pycma and nlopt); absence must be explicit."""
    for fn, dep in ((lambda: B.cma_es(bench, MAX_EVALS, batch_size=5, seed=0), "cma"),
                    (lambda: B.bobyqa(bench, MAX_EVALS, seed=0), "nlopt")):
        try:
            fn()
        except ImportError as exc:
            assert dep in str(exc)
        except Exception as exc:  # pragma: no cover - only if the dep is installed
            pytest.fail(f"unexpected error for {dep}: {type(exc).__name__}: {exc}")
