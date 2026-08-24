"""Domain rescaling and the initial design.

Paper anchors:
  - App. C: "The domain is rescaled to [0,1]^d and the function values are
    standardized before fitting the GP."
  - App. A: "All methods use 20 initial points from a Latin hypercube design (LHD) [29]"

See PAPER_SPEC.md §3 (`utils`) for shapes and dtypes.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_unit_cube",
    "from_unit_cube",
    "latin_hypercube",
    "standardize",
    "as_generator",
]


def as_generator(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Normalize a seed / Generator / None into a Generator.

    # [UNSPECIFIED] (PAPER_SPEC.md §10 C5) Neither the paper nor the official code
    # specifies a seeding policy; the official code uses the global NumPy RNG with no
    # seed parameter, which is not reproducible under parallel replications.
    # Using: an explicit Generator threaded through every stochastic call site.
    # Alternatives: global np.random seeding; per-module RNGs.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def to_unit_cube(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Project from the box [lb, ub] into [0,1]^d.  (App. C)

    x: (n, d) float64 -> (n, d) float64 in [0,1]^d
    """
    assert x.ndim == 2, "x must be (n, d)"
    assert lb.ndim == 1 and ub.ndim == 1, "bounds must be (d,)"
    assert np.all(lb < ub), "require lb < ub elementwise"
    return (x - lb) / (ub - lb)


def from_unit_cube(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Project from [0,1]^d back into the box [lb, ub].  (App. C)

    x: (n, d) float64 in [0,1]^d -> (n, d) float64
    """
    assert x.ndim == 2, "x must be (n, d)"
    assert lb.ndim == 1 and ub.ndim == 1, "bounds must be (d,)"
    assert np.all(lb < ub), "require lb < ub elementwise"
    return x * (ub - lb) + lb


def latin_hypercube(n_pts: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube design in [0,1]^d.  (App. A, "Latin hypercube design (LHD) [29]")

    Stratified centers, independently permuted per dimension, plus a uniform jitter of
    +/- 1/(2n) inside each stratum.

    # [FROM_OFFICIAL_CODE] uber-research/TuRBO utils.py `latin_hypercube`.
    # The paper cites LHD [29] but does not state the variant. Using: centered LHD with
    # per-box uniform perturbation, matching the official implementation.
    # Alternatives: maximin-optimized LHD (scipy.stats.qmc.LatinHypercube(optimization=...)),
    # plain stratified sampling without the jitter.

    Returns (n_pts, dim) float64 in [0,1]^d.
    """
    assert n_pts > 0 and dim > 0
    X = np.zeros((n_pts, dim), dtype=np.float64)
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):
        X[:, i] = centers[rng.permutation(n_pts)]
    X += rng.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    return X


def standardize(fX: np.ndarray, center: str = "median") -> tuple[np.ndarray, float, float]:
    """Standardize objective values before fitting the GP.  (App. C; PAPER_SPEC.md E7)

    # [PARTIALLY_SPECIFIED] (PAPER_SPEC.md §10 A4) App. C says only that "the function
    # values are standardized", which conventionally means mean/std. The official code
    # centers on the MEDIAN (turbo_1.py L159).
    # Using: median (official-code behavior), configurable via `center`.
    # Alternatives: "mean", the ordinary reading of "standardized".

    Returns (fX_standardized, mu, sigma). `sigma` is clamped to 1.0 when degenerate so a
    constant batch of values does not produce NaNs.
    """
    fX = np.asarray(fX, dtype=np.float64).ravel()
    if center == "median":
        mu = float(np.median(fX))
    elif center == "mean":
        mu = float(np.mean(fX))
    else:
        raise ValueError(f"center must be 'median' or 'mean', got {center!r}")
    sigma = float(fX.std())
    sigma = 1.0 if sigma < 1e-6 else sigma
    return (fX - mu) / sigma, mu, sigma
