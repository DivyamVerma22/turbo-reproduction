"""Replication driver: best-so-far traces and mean +/- one standard error.

Paper anchors:
  - Sect. 3: "Performance plots show the mean performances with one standard error."
  - App. A: "The optimizers are given a budget of 50 batches of size q = 10 which results
    in a total of n = 500 function evaluations. All methods use 20 initial points from a
    Latin hypercube design (LHD) [29] except for TuRBO-5, where we use 10 initial points in
    each local region. To compute confidence intervals on the results, we use 30 runs."

Rule 11 (CLAUDE.md): a result is only "reproduced" if the exact experiment was run. This
module records the settings actually used alongside the trace so that claim is checkable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .benchmarks import Benchmark, get_benchmark
from .turbo_1 import Turbo1
from .turbo_m import TurboM

__all__ = ["best_so_far", "mean_standard_error", "RunResult", "run_replications", "SYNTHETIC_PROTOCOL"]


# App. A settings for the synthetic suite, defined in config.py with the verbatim quote;
# re-exported so `from src.evaluate import SYNTHETIC_PROTOCOL` keeps working.
SYNTHETIC_PROTOCOL = config.SYNTHETIC_PROTOCOL


def best_so_far(fX: np.ndarray) -> np.ndarray:
    """Running minimum of the evaluation sequence. (n,) -> (n,)."""
    return np.minimum.accumulate(np.asarray(fX, dtype=np.float64).ravel())


def mean_standard_error(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and ONE standard error across replications.  (Sect. 3)

    Args:
        traces: (n_reps, n_evals) float64.

    Returns (mean, sem), each (n_evals,). SEM uses the sample standard deviation
    (ddof=1) divided by sqrt(n_reps), which is what "one standard error" means.
    """
    traces = np.atleast_2d(np.asarray(traces, dtype=np.float64))
    n_reps = traces.shape[0]
    mean = traces.mean(axis=0)
    if n_reps < 2:
        return mean, np.zeros_like(mean)
    return mean, traces.std(axis=0, ddof=1) / np.sqrt(n_reps)


@dataclass
class RunResult:
    """One replication, plus the settings it was actually run with (rule 11)."""

    benchmark: str
    algorithm: str
    seed: int
    best_value: float
    n_evals: int
    n_restarts: int
    trace: np.ndarray = field(repr=False)
    settings: dict = field(default_factory=dict)


def _make_optimizer(algorithm: str, bench: Benchmark, seed: int, **kwargs):
    if algorithm == "turbo-1":
        return Turbo1(
            f=bench,
            lb=bench.lb,
            ub=bench.ub,
            n_init=kwargs.pop("n_init", SYNTHETIC_PROTOCOL["n_init"]),
            max_evals=kwargs.pop("max_evals", SYNTHETIC_PROTOCOL["max_evals"]),
            batch_size=kwargs.pop("batch_size", SYNTHETIC_PROTOCOL["batch_size"]),
            seed=seed,
            **kwargs,
        )
    if algorithm.startswith("turbo-"):
        m = int(algorithm.split("-", 1)[1])
        return TurboM(
            f=bench,
            lb=bench.lb,
            ub=bench.ub,
            n_init=kwargs.pop("n_init", SYNTHETIC_PROTOCOL["n_init_per_tr"]),
            max_evals=kwargs.pop("max_evals", SYNTHETIC_PROTOCOL["max_evals"]),
            n_trust_regions=m,
            batch_size=kwargs.pop("batch_size", SYNTHETIC_PROTOCOL["batch_size"]),
            seed=seed,
            **kwargs,
        )
    raise ValueError(f"unknown algorithm {algorithm!r}; expected 'turbo-1' or 'turbo-<m>'")


def run_replications(
    benchmark: str,
    algorithm: str = "turbo-1",
    n_replications: int | None = None,
    base_seed: int = 0,
    **kwargs,
) -> list[RunResult]:
    """Run `n_replications` independent optimizations and collect best-so-far traces.

    Replications differ only by seed (base_seed + i), so a run is reproducible from
    (benchmark, algorithm, base_seed, settings).
    """
    bench = get_benchmark(benchmark)
    if n_replications is None:
        n_replications = SYNTHETIC_PROTOCOL["n_replications"]

    results: list[RunResult] = []
    for i in range(n_replications):
        seed = base_seed + i
        opt = _make_optimizer(algorithm, bench, seed, **dict(kwargs))
        opt.optimize()
        results.append(
            RunResult(
                benchmark=benchmark,
                algorithm=algorithm,
                seed=seed,
                best_value=opt.best_value,
                n_evals=opt.n_evals,
                n_restarts=opt.n_restarts,
                trace=best_so_far(opt.fX),
                settings={
                    "n_init": opt.n_init,
                    "max_evals": opt.max_evals,
                    "batch_size": opt.batch_size,
                    "failtol": opt.failtol,
                    "succtol": opt.state.succtol if hasattr(opt, "state") else None,
                    "n_cand": opt.n_cand,
                    "success_tol": opt.success_tol,
                    "center_stat": opt.center_stat,
                    "use_predictive": opt.use_predictive,
                    "noisy": opt.noisy,
                },
            )
        )
    return results


def summarize(results: list[RunResult]) -> dict:
    """Mean +/- one standard error of the final best value across replications."""
    finals = np.array([r.best_value for r in results], dtype=np.float64)
    n = len(finals)
    sem = float(finals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "benchmark": results[0].benchmark,
        "algorithm": results[0].algorithm,
        "n_replications": n,
        "mean_best": float(finals.mean()),
        "standard_error": sem,
        "median_best": float(np.median(finals)),
        "min_best": float(finals.min()),
        "max_best": float(finals.max()),
        "settings": dict(results[0].settings),
    }
