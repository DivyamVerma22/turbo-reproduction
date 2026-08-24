"""One trust region's surrogate step: fit the GP, place the box, draw Thompson samples.

This is the block that TuRBO-1 and TuRBO-m share. Sect. 2 describes it once and then says
of the multi-region case only that "Each trust region TR_l ... utilizes an independent local
GP model" -- so the per-region machinery is identical and the two algorithms differ solely
in how candidates are *selected* across regions (thompson.select_candidates vs
select_candidates_across) and in how restarts are managed.

Ordering here is load-bearing (PAPER_SPEC.md §5):

    standardize -> fit GP -> pick center -> derive bounds from the CURRENT fitted
    lengthscales -> candidates -> Thompson draws

Deriving the bounds before the refit would silently reuse the previous batch's geometry.

The random-number consumption order (candidate set first, then the draw seed) is part of
the reproducibility contract: changing it changes realized trajectories for a given seed
even though the algorithm is unchanged.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import gpytorch
import numpy as np
import torch

from . import config
from .candidates import create_candidates
from .gp import GP, train_gp
from .thompson import thompson_draws
from .trust_region import select_center, trust_region_bounds
from .utils import standardize

__all__ = ["SurrogateSettings", "LocalProposal", "propose_from_trust_region"]


@dataclass(frozen=True)
class SurrogateSettings:
    """Knobs shared by every local model in a run.

    Defaults come from `config` and are unchanged from the pre-refactor signatures. The
    three behavioral ones (`center_stat`, `use_predictive`, `noisy`) each correspond to a
    documented paper/official-code conflict; see REPRODUCTION_NOTES.md §1.

    Attributes:
        use_ard: App. C specifies ARD "for all experiments"; the isotropic path exists only
            for the Sect. 3.6 ablation.
        n_training_steps: Adam steps on the marginal likelihood (FROM_OFFICIAL_CODE).
        max_cholesky_size: point count above which GPyTorch switches to CG + Lanczos (App. C).
        center_stat: "median" (official code) or "mean" (the ordinary reading of App. C).
        use_predictive: sample the predictive distribution (official code) rather than the
            latent posterior that Sect. 2's equation specifies.
        noisy: use the paper's noisy centering rule (Sect. 2) instead of the best observation.
    """

    use_ard: bool = True
    n_training_steps: int = config.N_TRAINING_STEPS
    max_cholesky_size: int = config.MAX_CHOLESKY_SIZE
    center_stat: str = config.CENTER_STAT
    use_predictive: bool = config.USE_PREDICTIVE_DRAWS
    noisy: bool = False


@dataclass(frozen=True)
class LocalProposal:
    """Everything one trust region produced this batch.

    Attributes:
        X_cand: (n_cand, d) float64 candidate set in [0,1]^d.
        y_cand: (n_cand, q) float64 Thompson draws, already de-standardized (E7) so they
            are comparable against draws from other trust regions.
        x_center: (1, d) float64 trust-region center x* (E3).
        lengthscales: (d,) float64 fitted ARD lengthscales lambda_i, used for the box (E2).
        hypers: fitted GP `state_dict`, for TuRBO-m's warm-start cache (PAPER_SPEC.md §10 B8).
    """

    X_cand: np.ndarray
    y_cand: np.ndarray
    x_center: np.ndarray
    lengthscales: np.ndarray
    hypers: dict[str, Any]


def _fitted_lengthscales(gp: GP, dim: int) -> np.ndarray:
    """Extract lambda_i as (d,), broadcasting the isotropic case to every dimension.

    An isotropic kernel (Sect. 3.6 ablation) exposes a single lengthscale; the trust-region
    rescaling in E2 needs one per dimension, and repeating it yields the cube that isotropy
    implies.
    """
    lengthscales = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
    if lengthscales.size == 1:
        lengthscales = np.repeat(lengthscales, dim)
    return lengthscales


def propose_from_trust_region(
    X_unit: np.ndarray,
    fX: np.ndarray,
    *,
    length: float,
    batch_size: int,
    n_cand: int,
    rng: np.random.Generator,
    settings: SurrogateSettings,
    hypers: dict[str, Any] | None = None,
) -> LocalProposal:
    """Fit one local GP and draw a batch of Thompson samples inside its trust region.

    Args:
        X_unit: (n, d) float64, this trust region's points **in [0,1]^d** (App. C).
        fX: (n,) or (n, 1) float64 observed objective values, unstandardized.
        length: base side length L of this trust region.
        batch_size: q, the number of independent posterior realizations (E5).
        n_cand: candidate-set size, App. D's min{100d, 5000}.
        rng: the run's generator. Consumed by `create_candidates`, then once more for the
            Thompson draw seed.
        settings: shared surrogate configuration.
        hypers: cached `state_dict` to warm-start from. When supplied, the GP is NOT refit
            (`num_steps=0`) -- TuRBO-m does this for regions that received no points, whose
            training data is therefore unchanged (PAPER_SPEC.md §10 B8, turbo_m.py L165).

    Returns:
        A `LocalProposal`. Candidate selection is left to the caller, because that is
        exactly where TuRBO-1 and TuRBO-m differ (Sect. 2).
    """
    dim = X_unit.shape[1]

    # App. C: "the function values are standardized before fitting the GP" (E7).
    fX_std, mu, sigma = standardize(fX, center=settings.center_stat)

    # App. C: "The GP hyperparameters are fitted before proposing a new batch by optimizing
    # the log-marginal likelihood." Must precede the bound computation below.
    num_steps = 0 if hypers else settings.n_training_steps
    gp = train_gp(
        train_x=torch.as_tensor(X_unit, dtype=torch.float64),
        train_y=torch.as_tensor(fX_std, dtype=torch.float64),
        use_ard=settings.use_ard,
        num_steps=num_steps,
        hypers=hypers or None,
        max_cholesky_size=settings.max_cholesky_size,
    )
    fitted_hypers = {k: v.detach().clone() for k, v in gp.state_dict().items()}

    # Sect. 2 (E3): best observation, or smallest posterior mean under noise.
    posterior_mean = None
    if settings.noisy:
        # Evaluating the posterior AT the training inputs is exactly what Sect. 2 asks for
        # ("the observation with the smallest posterior mean"), but GPyTorch warns that the
        # input matches its stored training data and suspects a forgotten `.train()`. The
        # eval-mode posterior is the intended quantity, so silence just this warning rather
        # than training reviewers to ignore warnings in general.
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.GPInputWarning)
            posterior_mean = gp(torch.as_tensor(X_unit, dtype=torch.float64)).mean.cpu().numpy()
    x_center = select_center(X_unit, fX_std, posterior_mean, noisy=settings.noisy)

    # Sect. 2 (E2): the box is derived from the lengthscales just fitted.
    lengthscales = _fitted_lengthscales(gp, dim)
    lb_tr, ub_tr = trust_region_bounds(x_center, lengthscales, length)

    # App. D (E6), then Sect. 2 (E5). RNG order: candidates, then the draw seed.
    X_cand = create_candidates(x_center, lb_tr, ub_tr, rng, n_cand=n_cand)
    y_cand = thompson_draws(
        gp,
        X_cand,
        batch_size,
        mu=mu,
        sigma=sigma,
        use_predictive=settings.use_predictive,
        max_cholesky_size=settings.max_cholesky_size,
        seed=int(rng.integers(0, config.TORCH_SEED_MAX)),
    )

    return LocalProposal(
        X_cand=X_cand,
        y_cand=y_cand,
        x_center=x_center,
        lengthscales=lengthscales,
        hypers=fitted_hypers,
    )
