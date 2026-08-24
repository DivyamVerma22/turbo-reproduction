"""Benchmark objectives.

All callables here MINIMIZE, matching Sect. 2 ("Find x* in Omega such that f(x*) <= f(x)").
The Sect. 3 problems (robot pushing, rover, lunar lander) report REWARD, where higher is
better; wrap such an objective with `negate` before handing it to the optimizer. Note that
`evaluate.py` reports the minimized value as-is -- it does not convert back to reward, so a
caller comparing against Fig. 2-3 must negate the reported value itself.

Domains are from the paper (App. A; Sect. 3.5):
  Ackley    [-5, 10]^10      Levy      [-5, 10]^10
  Rastrigin [-3,  4]^10      Hartmann6 [ 0,  1]^6
  Ackley-200 [-5, 10]^200 (Sect. 3.5)

# [UNSPECIFIED] (PAPER_SPEC.md §10) The paper names these four synthetic functions and
# gives their domains (App. A) but never writes out their formulas.
# Using: the standard textbook definitions (Surjanovic & Bingham, "Virtual Library of
# Simulation Experiments"), which are what the BO literature universally means by these
# names, with the global minima noted per function.
# Alternatives: none plausible -- but note that a different Rastrigin amplitude or Ackley
# (a, b, c) would change absolute objective values and make numbers incomparable to Fig. 8.

Three of the paper's benchmarks are NOT reproducible from the paper alone. They are
declared here so that calling them fails loudly instead of silently substituting something
else. See PAPER_SPEC.md §10 C9/C10 and REPRODUCTION_NOTES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "Benchmark",
    "negate",
    "ackley",
    "levy",
    "rastrigin",
    "hartmann6",
    "SYNTHETIC_SUITE",
    "get_benchmark",
    "robot_pushing",
    "rover",
    "lunar_lander",
    "cosmological_constants",
]


@dataclass(frozen=True)
class Benchmark:
    """A minimization problem with a known box domain."""

    name: str
    f: Callable[[np.ndarray], float]
    lb: np.ndarray
    ub: np.ndarray
    dim: int
    global_min: float | None = None

    def __call__(self, x: np.ndarray) -> float:
        return float(self.f(np.asarray(x, dtype=np.float64).ravel()))


def negate(f: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
    """Turn a reward (higher is better) into a cost for the minimizer.  (PAPER_SPEC.md §10 C7)

    Sect. 2 states the problem as minimization, while the Sect. 3 benchmarks report reward:
    e.g. App. F.1 gives robot pushing as f(x) = sum_i ||x_gi - x_si|| - ||x_gi - x_fi||,
    which the paper maximizes. Wrap such an objective with `negate` before optimizing, and
    negate the reported best value again to read it as reward.
    """

    def cost(x):
        return -float(f(x))

    cost.__doc__ = f"Negated (minimization) form of {getattr(f, '__name__', 'reward')}."
    return cost


# --- synthetic functions -----------------------------------------------------------
def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi) -> float:
    """Ackley. Global minimum 0 at the origin."""
    x = np.asarray(x, dtype=np.float64).ravel()
    d = x.size
    return float(
        -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
        - np.exp(np.sum(np.cos(c * x)) / d)
        + a
        + np.e
    )


def levy(x: np.ndarray) -> float:
    """Levy. Global minimum 0 at x = (1, ..., 1)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    term2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    return float(term1 + term2 + term3)


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin. Global minimum 0 at the origin."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(10.0 * x.size + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


_H6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H6_A = np.array(
    [
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
    ]
)
_H6_P = 1e-4 * np.array(
    [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]
)


def hartmann6(x: np.ndarray) -> float:
    """Hartmann-6. Global minimum approx -3.32237 on [0,1]^6."""
    x = np.asarray(x, dtype=np.float64).ravel()
    assert x.size == 6, "hartmann6 is defined on [0,1]^6"
    inner = np.sum(_H6_A * (x - _H6_P) ** 2, axis=1)
    return float(-np.sum(_H6_ALPHA * np.exp(-inner)))


def _bench(name, f, lo, hi, dim, gmin) -> Benchmark:
    return Benchmark(
        name=name,
        f=f,
        lb=np.full(dim, lo, dtype=np.float64),
        ub=np.full(dim, hi, dtype=np.float64),
        dim=dim,
        global_min=gmin,
    )


# App. A: "Ackley with domain [-5,10]^10, Levy with domain [-5,10]^10, Rastrigin with
# domain [-3,4]^10, and the 6D Hartmann function with domain [0,1]^6."
SYNTHETIC_SUITE = {
    "ackley10": _bench("ackley10", ackley, -5.0, 10.0, 10, 0.0),
    "levy10": _bench("levy10", levy, -5.0, 10.0, 10, 0.0),
    "rastrigin10": _bench("rastrigin10", rastrigin, -3.0, 4.0, 10, 0.0),
    "hartmann6": _bench("hartmann6", hartmann6, 0.0, 1.0, 6, -3.32237),
    # Sect. 3.5: "the 200-dimensional Ackley function in the domain [-5,10]^200"
    "ackley200": _bench("ackley200", ackley, -5.0, 10.0, 200, 0.0),
}


def get_benchmark(name: str) -> Benchmark:
    if name not in SYNTHETIC_SUITE:
        raise KeyError(f"unknown benchmark {name!r}; available: {sorted(SYNTHETIC_SUITE)}")
    return SYNTHETIC_SUITE[name]


# --- benchmarks that cannot be reproduced from the paper ---------------------------
_UNAVAILABLE = {
    "robot_pushing": (
        "14D robot pushing (Sect. 3.1, App. F.1). App. F.1 gives the reward formula\n"
        "  f(x) = sum_i ||x_gi - x_si|| - ||x_gi - x_fi||\n"
        "but the pushing simulator itself is external (Wang et al. 2018). The collision\n"
        "and contact model is not described in this paper. Vendor the simulator from the\n"
        "Wang et al. 2018 release to run this benchmark. See PAPER_SPEC.md §10 C10."
    ),
    "rover": (
        "60D rover trajectory planning (Sect. 3.2, App. F.2). App. F.2 gives the reward\n"
        "  f(x) = c(x) - 10*(||x_{1,2} - x_s||_1 + ||x_{59,60} - x_g||_1) + 5\n"
        "with c(x) penalizing each collision by -20, but the terrain, the B-spline fitting\n"
        "details and the collision geometry are external (Wang et al. 2018).\n"
        "See PAPER_SPEC.md §10 C10."
    ),
    "lunar_lander": (
        "12D lunar lander (Sect. 3.4, App. F.4). Requires OpenAI Gym LunarLander-v2 AND\n"
        "the paper's 'fixed constant set of 50 randomly generated terrains, initial\n"
        "positions, and velocities' -- those 50 seeds are not published, so results are\n"
        "not comparable to Fig. 3 even with Gym installed. See PAPER_SPEC.md §10 C10."
    ),
    "cosmological_constants": (
        "12D cosmological constant learning (Sect. 3.3, App. F.3). NOT reproducible:\n"
        "App. F.3 says 'the nine parameters tuned in previous papers, plus three additional\n"
        "parameters chosen from the many available to the simulator' without naming the\n"
        "three, and Sect. 3.3 says 'substantially larger parameter bounds' without giving\n"
        "any bounds. See PAPER_SPEC.md §10 C9."
    ),
}


def _unavailable(key: str):
    def _raise(*_args, **_kwargs):
        raise NotImplementedError(_UNAVAILABLE[key])

    _raise.__doc__ = _UNAVAILABLE[key]
    return _raise


robot_pushing = _unavailable("robot_pushing")
rover = _unavailable("rover")
lunar_lander = _unavailable("lunar_lander")
cosmological_constants = _unavailable("cosmological_constants")
