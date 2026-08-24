"""Discretized candidate set inside a trust region.

Paper anchor (App. D, verbatim):
  "Each TR in TuRBO uses a candidate set of size min{100d, 5000} on which we generate each
   Thompson sample. We create each candidate set by first generating a scrambled Sobol
   sequence within the intersection of the TR and the domain [0,1]^d. A new candidate set
   is generated for each batch. In order to not perturb all coordinates at once, we use
   the value in the Sobol sequence with probability min{1, 20/d} for a given candidate and
   dimension, and the value of the center otherwise."

App. E explains why a discrete set is used at all: "we cannot sample an entire function f
from the GP posterior in practice. We therefore work in a discretized setting by first
drawing a finite candidate set".

See PAPER_SPEC.md E6.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from . import config

__all__ = ["n_candidates", "perturbation_probability", "create_candidates"]


def n_candidates(dim: int) -> int:
    """App. D: "a candidate set of size min{100d, 5000}"."""
    return int(min(config.N_CAND_PER_DIM * dim, config.N_CAND_MAX))


def perturbation_probability(dim: int) -> float:
    """App. D: "with probability min{1, 20/d}".

    Note this is a no-op for d <= 20 (every coordinate is perturbed).
    """
    return float(min(1.0, config.PERTURB_NUMERATOR / dim))


def create_candidates(
    x_center: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    n_cand: int | None = None,
) -> np.ndarray:
    """Build one candidate set inside [lb, ub].  (App. D)

    Args:
        x_center: (1, d) float64 in [0,1]^d, the TR center.
        lb, ub: (1, d) float64, TR bounds already clipped to the unit cube.
        rng: Generator (also used to seed the Sobol scramble).
        n_cand: defaults to min(100d, 5000).

    Returns X_cand: (n_cand, d) float64 in [0,1]^d.
    """
    x_center = np.atleast_2d(np.asarray(x_center, dtype=np.float64))
    lb = np.atleast_2d(np.asarray(lb, dtype=np.float64))
    ub = np.atleast_2d(np.asarray(ub, dtype=np.float64))
    assert x_center.shape == lb.shape == ub.shape and x_center.shape[0] == 1
    dim = x_center.shape[1]
    if n_cand is None:
        n_cand = n_candidates(dim)

    # App. D: "a scrambled Sobol sequence within the intersection of the TR and the domain".
    # [FROM_OFFICIAL_CODE] (PAPER_SPEC.md §10 B10) The paper does not state a seeding
    # scheme; the official code draws a fresh random scramble seed per batch
    # (turbo_1.py L191-L192). Using: fresh seed per call, drawn from `rng` so the whole
    # run stays reproducible from one seed.
    # Alternatives: a fixed seed; one continuing Sobol sequence across batches.
    seed = int(rng.integers(0, config.SOBOL_SEED_MAX))
    sobol = SobolEngine(dim, scramble=True, seed=seed)
    pert = sobol.draw(n_cand).to(dtype=torch.float64).numpy()
    pert = lb + (ub - lb) * pert

    # App. D: perturb each coordinate with probability min{1, 20/d}, else keep the center.
    prob_perturb = perturbation_probability(dim)
    mask = rng.random((n_cand, dim)) <= prob_perturb

    # [FROM_OFFICIAL_CODE] (PAPER_SPEC.md §10 B6) App. D does not say what happens when a
    # candidate's mask is empty (it would duplicate the center). The official code forces
    # one random dimension on (turbo_1.py L197-L198).
    # Using: that fix, but over the full range [0, d).
    # NOTE: the official line is `np.random.randint(0, self.dim - 1, ...)`, which can never
    # select the last dimension -- an upstream off-by-one. Deliberately not reproduced;
    # see REPRODUCTION_NOTES.md.
    # Alternatives: resample the mask until non-empty.
    empty = np.where(~mask.any(axis=1))[0]
    if empty.size > 0:
        mask[empty, rng.integers(0, dim, size=empty.size)] = True

    X_cand = np.tile(x_center, (n_cand, 1))
    X_cand[mask] = pert[mask]
    return X_cand
