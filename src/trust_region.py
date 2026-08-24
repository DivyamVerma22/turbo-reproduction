"""Trust-region geometry, centering, and the success/failure resize rule.

Paper anchors (Sect. 2, "Trust regions"; App. D). Verbatim:
  - "We choose our TR to be a hyperrectangle centered at the best solution found so far,
     denoted by x*."
  - "In the noise-free case, we set x* to the location of the best observation so far. In
     the presence of noise, we use the observation with the smallest posterior mean under
     the surrogate model."
  - "The actual side length for each dimension is obtained from this base side length by
     rescaling according to its lengthscale lambda_i in the GP model while maintaining a
     total volume of L^d. That is, L_i = lambda_i L / (prod_j lambda_j)^(1/d)."
  - "After tau_succ consecutive successes, we double the size of the TR, i.e.,
     L <- min{L_max, 2L}. After tau_fail consecutive failures, we halve the size of the
     TR: L <- L/2. We reset the success and failure counters to zero after we change the
     size of the TR. Whenever L falls below a given minimum threshold L_min, we discard
     the respective TR and initialize a new one with side length L_init."

See PAPER_SPEC.md E2, E3, E4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config

__all__ = [
    "TR_DEFAULTS",
    "TrustRegionState",
    "trust_region_weights",
    "trust_region_bounds",
    "select_center",
    "is_success",
    "default_failtol",
]


# App. D: "In all experiments, we use the following hyperparameters for TuRBO-1:
# tau_succ = 3, tau_fail = ceil(d/q), L_min = 2^-7, L_max = 1.6, and L_init = 0.8".
# Defined in config.py alongside their provenance; re-exported here so
# `from src.trust_region import TR_DEFAULTS` keeps working.
TR_DEFAULTS = config.TR_DEFAULTS


def default_failtol(dim: int, batch_size: int, n_trust_regions: int = 1) -> int:
    """tau_fail.

    App. D (TuRBO-1): "tau_fail = ceil(d/q)".
    App. D (TuRBO-m): "we use the same tolerances as in the sequential case (q = 1) as the
    number of evaluations allocated by each TR may differ in each batch" -> ceil(d/1) = d.

    # [PARTIALLY_SPECIFIED] (PAPER_SPEC.md §10 A2) The official code disagrees with the
    # paper here, in both variants:
    #   TuRBO-1: ceil(max(4/q, d/q))  (turbo_1.py L106) -- adds a floor of 4/q
    #   TuRBO-m: max(5, d)            (turbo_m.py L86)  -- not ceil(d/q) at all
    # Using: the paper's values.
    # Alternatives: the official-code values (differ whenever d < 4 for TuRBO-1, and for
    # every TuRBO-m run).
    """
    if n_trust_regions > 1:
        return int(math.ceil(dim / 1))
    return int(math.ceil(dim / batch_size))


@dataclass
class TrustRegionState:
    """Mutable state of a single trust region.  (Sect. 2, "Trust regions"; App. D)"""

    length: float = TR_DEFAULTS["length_init"]
    succcount: int = 0
    failcount: int = 0
    succtol: int = TR_DEFAULTS["succtol"]
    failtol: int = 4
    length_min: float = TR_DEFAULTS["length_min"]
    length_max: float = TR_DEFAULTS["length_max"]
    length_init: float = TR_DEFAULTS["length_init"]
    # TuRBO-m caches fitted hypers so a TR that received no points this batch is not
    # refit. [FROM_OFFICIAL_CODE] PAPER_SPEC.md §10 B8; turbo_m.py L165.
    hypers: dict = field(default_factory=dict)

    @property
    def is_converged(self) -> bool:
        """App. D: "terminate the TR when L < L_min"."""
        return self.length < self.length_min

    def reset(self) -> None:
        """App. D: "For each TR, we initialize L <- L_init"."""
        self.length = self.length_init
        self.succcount = 0
        self.failcount = 0
        self.hypers = {}

    def update(self, success: bool, n_failures: int = 1) -> None:
        """Apply the success/failure counters and resize rule.  (Sect. 2; App. D)

        Args:
            success: whether this batch improved on the TR's incumbent (see `is_success`).
            n_failures: amount to add to the failure counter on a failure. This is 1 for
                TuRBO-1 (App. D: "we set the success counter to zero and increment the
                failure counter") and q_l for TuRBO-m (App. D: "If all q_l > 0 evaluations
                are worse than the current best solution we consider this a failure and
                set the success counter to zero and add q_l to the failure counter").
        """
        if success:
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += n_failures

        if self.succcount == self.succtol:
            self.length = min(2.0 * self.length, self.length_max)
            self.succcount = 0
        elif self.failcount >= self.failtol:
            # App. D: "The failure counter is set to tau_fail if we increment past this
            # tolerance, which will trigger a halving of its side length." -> the
            # comparison must be >=, not ==, because TuRBO-m adds q_l at a time.
            self.length /= 2.0
            self.failcount = 0


def is_success(
    f_batch: np.ndarray, f_incumbent: float, success_tol: float = config.SUCCESS_TOL
) -> bool:
    """Did this batch improve on the trust region's incumbent?

    Sect. 2: "We define a 'success' as a candidate that improves upon x*, and a 'failure'
    as a candidate that does not."
    App. D: "When using TuRBO-1, we consider an improvement from at least one evaluation
    in the batch a success."

    # [PARTIALLY_SPECIFIED] (PAPER_SPEC.md §10 A1) The paper states no tolerance -- read
    # literally, any strict improvement is a success. The official code requires a
    # relative margin: min(fX_next) < min(fX) - 1e-3 * abs(min(fX))
    # (turbo_1.py L138, turbo_m.py L109).
    # Using: the official-code margin, `success_tol = 1e-3`.
    # Alternatives: success_tol = 0.0 recovers the paper's literal rule. This materially
    # changes behavior -- with 0.0, trust regions expand far more readily on noisy or
    # plateaued objectives.
    """
    return bool(np.min(f_batch) < f_incumbent - success_tol * abs(f_incumbent))


def trust_region_weights(lengthscales: np.ndarray) -> np.ndarray:
    """Per-dimension side-length weights with unit product.  (Sect. 2, "Trust regions")

    Implements L_i = lambda_i * L / (prod_j lambda_j)^(1/d) as w_i * L, where
    prod_i w_i == 1 so that prod_i L_i == L^d exactly (the paper's stated volume
    invariant).

    # [FROM_OFFICIAL_CODE] (PAPER_SPEC.md §10 B4) The official code divides by the
    # arithmetic mean first, then by the geometric mean (turbo_1.py L183-L184). That is
    # mathematically identical to a single geometric-mean division but avoids
    # overflow/underflow of prod(lambda) at d = 200.
    # Alternatives: direct division by prod(lambda)^(1/d).

    Returns (d,) float64 with prod == 1.
    """
    w = np.asarray(lengthscales, dtype=np.float64).ravel()
    assert w.ndim == 1 and np.all(w > 0), "lengthscales must be positive (d,)"
    w = w / w.mean()
    w = w / np.prod(np.power(w, 1.0 / len(w)))
    return w


def trust_region_bounds(
    x_center: np.ndarray, lengthscales: np.ndarray, length: float
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned TR bounds, clipped to the unit cube.  (Sect. 2; App. D)

    Args:
        x_center: (1, d) float64 in [0,1]^d.
        lengthscales: (d,) float64, the GP's *current* fitted ARD lengthscales.
        length: base side length L.

    Returns (lb, ub), each (1, d) float64 clipped to [0, 1].

    # [PARTIALLY_SPECIFIED] (PAPER_SPEC.md §10 B3) The paper gives L_i and says the TR is
    # "a hyperrectangle centered at the best solution found so far", which implies the
    # +/- L_i/2 extents; App. D refers to "the intersection of the TR and the domain
    # [0,1]^d". The clipping is explicit in the official code (turbo_1.py L185-L186).
    # NOTE: clipping breaks the exact prod(L_i) == L^d volume invariant near the domain
    # boundary. Neither the paper nor the official code addresses this.
    # Alternatives: reflect or shift the box to preserve volume at the boundary.
    """
    x_center = np.atleast_2d(np.asarray(x_center, dtype=np.float64))
    assert x_center.shape[0] == 1, "x_center must be (1, d)"
    w = trust_region_weights(lengthscales)
    lb = np.clip(x_center - w * length / 2.0, 0.0, 1.0)
    ub = np.clip(x_center + w * length / 2.0, 0.0, 1.0)
    return lb, ub


def select_center(
    X: np.ndarray,
    fX: np.ndarray,
    posterior_mean: np.ndarray | None = None,
    noisy: bool = False,
) -> np.ndarray:
    """Choose the trust-region center x*.  (Sect. 2, "Trust regions")

    Args:
        X: (n, d) float64 in [0,1]^d.
        fX: (n,) float64 observed values.
        posterior_mean: (n,) float64 GP posterior mean at X. Required when noisy=True.
        noisy: select by smallest posterior mean instead of smallest observation.

    Returns (1, d) float64.

    # [SPECIFIED but unimplemented upstream] (PAPER_SPEC.md §10 A3) The paper specifies
    # BOTH rules. The official code only ever implements the noise-free one and says so in
    # a comment: "NOTE: This may not be robust to noise, in which case the posterior mean
    # of the GP can be used instead" (turbo_1.py L152-L154, L181).
    # Using: both, selected by `noisy`. Default False = official-code behavior.
    # NOTE: robot pushing (Sect. 3.1) and lunar lander (Sect. 3.4) are explicitly noisy
    # problems, so this path is exercised during reproduction.
    """
    X = np.asarray(X, dtype=np.float64)
    fX = np.asarray(fX, dtype=np.float64).ravel()
    assert X.ndim == 2 and X.shape[0] == fX.shape[0]

    if noisy:
        if posterior_mean is None:
            raise ValueError(
                "noisy=True requires posterior_mean (Sect. 2: 'we use the observation "
                "with the smallest posterior mean under the surrogate model')"
            )
        idx = int(np.argmin(np.asarray(posterior_mean, dtype=np.float64).ravel()))
    else:
        idx = int(np.argmin(fX))
    return X[idx, :][None, :]
