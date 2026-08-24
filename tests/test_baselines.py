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
from src.benchmarks import get_benchmark

MAX_EVALS = 30


@pytest.fixture(scope="module")
def bench():
    return get_benchmark("hartmann6")


@pytest.mark.parametrize("name", ["random_search", "nelder_mead", "bfgs"])
def test_baseline_runs_and_respects_its_budget(bench, name):
    result = getattr(B, name)(bench, MAX_EVALS, seed=0)
    assert result["n_evals"] <= MAX_EVALS
    assert np.isfinite(result["best_value"])
    assert result["fX"].shape == (result["n_evals"],)
    assert result["best_value"] == pytest.approx(result["fX"].min())


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
