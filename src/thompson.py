"""Thompson sampling: posterior draws on the candidate set, and batch selection.

Paper anchors:
  Sect. 2, "Trust region Bayesian optimization" (verbatim):
    "To select the i-th candidate from across the trust regions, we draw a realization of
     the posterior function from the local GP within each TR: f_l^(i) ~ GP_l^(t)(mu_l(x),
     k_l(x,x')). We then select the i-th candidate such that it minimizes the function
     value across all m samples and all trust regions:
        x_i^(t) in argmin_l argmin_{x in TR_l} f_l^(i)
     That is, we select as point with the smallest function value after concatenating a
     Thompson sample from each TR for i = 1, ..., q."

  App. E (verbatim):
    "Independent TS for parallel batches is exactly equivalent to conditioning on imputed
     values for unobserved suggestions."

See PAPER_SPEC.md E5.
"""

from __future__ import annotations

import gpytorch
import numpy as np
import torch

from .gp import GP

__all__ = ["thompson_draws", "select_candidates", "select_candidates_across"]


def thompson_draws(
    gp: GP,
    X_cand: np.ndarray,
    batch_size: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    use_predictive: bool = True,
    max_cholesky_size: int = 2000,
    seed: int | None = None,
) -> np.ndarray:
    """Draw `batch_size` independent posterior realizations on the candidate set.

    Args:
        gp: fitted GP (eval mode) trained on standardized values.
        X_cand: (n_cand, d) float64 in [0,1]^d.
        batch_size: q, the number of independent realizations.
        mu, sigma: the standardization constants used when fitting (see utils.standardize).
        use_predictive: sample the predictive distribution (with observation noise) rather
            than the latent posterior.
        seed: if given, the torch RNG is forked and seeded for this draw so a whole run is
            reproducible from a single numpy seed. GPyTorch's `.sample()` reads torch's
            GLOBAL generator and takes no generator argument, so threading the run's
            np.random.Generator alone does NOT make sampling deterministic. Forking keeps
            the caller's global torch RNG state untouched. (PAPER_SPEC.md §10 C5)

    Returns y_cand: (n_cand, q) float64, **de-standardized**.

    # [PARTIALLY_SPECIFIED] (PAPER_SPEC.md §10 B7) Sect. 2 writes f ~ GP(mu, k), i.e. the
    # LATENT posterior. The official code samples gp.likelihood(gp(X_cand)) -- the
    # PREDICTIVE distribution, which adds sigma^2 observation noise to every draw
    # (turbo_1.py L216) and therefore increases TS exploration.
    # Using: predictive (official-code behavior), configurable.
    # Alternatives: use_predictive=False for the paper's equation as literally written.

    # NOTE (PAPER_SPEC.md E7): de-standardizing is order-preserving within a single TR, so
    # it looks optional -- but it is load-bearing for TuRBO-m, where the bandit compares
    # draws from m GPs that each fitted their own (mu, sigma). Omitting it silently makes
    # that cross-TR comparison meaningless while still running cleanly.
    """
    X_torch = torch.as_tensor(np.asarray(X_cand, dtype=np.float64), dtype=torch.float64)

    def _draw():
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(max_cholesky_size):
            dist = gp(X_torch)
            if use_predictive:
                dist = gp.likelihood(dist)
            return dist.sample(torch.Size([batch_size])).t().cpu().numpy()

    if seed is None:
        y = _draw()
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            y = _draw()
    return mu + sigma * np.asarray(y, dtype=np.float64)


def select_candidates(X_cand: np.ndarray, y_cand: np.ndarray) -> np.ndarray:
    """Select q points from ONE trust region.  (Sect. 2; TuRBO-1)

    Args:
        X_cand: (n_cand, d) float64.
        y_cand: (n_cand, q) float64. Mutated: selected rows are set to +inf.

    Returns X_next: (q, d) float64.

    # [FROM_OFFICIAL_CODE] (PAPER_SPEC.md §10 B5) The de-duplication -- masking a chosen
    # candidate out of every column so it can never be picked twice within a batch -- is
    # not in the paper. It is in turbo_1.py L233.
    # Alternatives: allow duplicates (pure independent TS, arguably closer to the
    # equation as literally written).
    """
    X_cand = np.asarray(X_cand, dtype=np.float64)
    y_cand = np.asarray(y_cand, dtype=np.float64)
    n_cand, dim = X_cand.shape
    batch_size = y_cand.shape[1]
    assert y_cand.shape[0] == n_cand

    # De-duplication masks a chosen candidate to +inf, so a batch larger than the candidate
    # set exhausts it; argmin over an all-inf column then silently returns index 0 and the
    # same point is proposed repeatedly, wasting evaluations. Unreachable under the paper's
    # settings (App. D gives n_cand = min(100d, 5000) >= 100), so fail loudly instead.
    if batch_size > n_cand:
        raise ValueError(
            f"batch_size q={batch_size} exceeds the candidate set n_cand={n_cand}; "
            "App. D sets n_cand = min(100d, 5000), so this cannot happen with paper settings"
        )

    X_next = np.zeros((batch_size, dim), dtype=np.float64)
    for i in range(batch_size):
        idx = int(np.argmin(y_cand[:, i]))
        X_next[i, :] = X_cand[idx, :]
        y_cand[idx, :] = np.inf
    return X_next


def select_candidates_across(
    X_cand: np.ndarray, y_cand: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select q points jointly across m trust regions -- the implicit bandit.  (Sect. 2)

    This is the single line that makes TuRBO-m more than m independent TuRBO-1 runs:
    the argmin is taken over (trust region, candidate) jointly, so the number of points a
    TR receives this batch is decided by how good its posterior draws are.

    Args:
        X_cand: (m, n_cand, d) float64.
        y_cand: (m, n_cand, q) float64, de-standardized. Mutated: selected entries -> +inf.

    Returns:
        X_next: (q, d) float64.
        idx_next: (q, 1) int, the owning trust region of each selected point.
    """
    X_cand = np.asarray(X_cand, dtype=np.float64)
    y_cand = np.asarray(y_cand, dtype=np.float64)
    m, n_cand, dim = X_cand.shape
    batch_size = y_cand.shape[2]
    assert y_cand.shape[:2] == (m, n_cand)
    assert np.all(np.isfinite(y_cand)), "posterior draws must be finite"
    if batch_size > m * n_cand:  # see the note in select_candidates
        raise ValueError(
            f"batch_size q={batch_size} exceeds the pooled candidate set m*n_cand={m * n_cand}"
        )

    X_next = np.zeros((batch_size, dim), dtype=np.float64)
    idx_next = np.zeros((batch_size, 1), dtype=int)
    for k in range(batch_size):
        i, j = np.unravel_index(np.argmin(y_cand[:, :, k]), (m, n_cand))
        X_next[k, :] = X_cand[i, j, :]
        idx_next[k, 0] = i
        y_cand[i, j, :] = np.inf  # never select this candidate again (B5)
    return X_next, idx_next
