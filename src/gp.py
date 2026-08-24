"""Local GP surrogate: Matern-5/2 with ARD, constant mean, Gaussian likelihood.

Paper anchor (App. C, verbatim):
  "the GP is parameterized using a Matern-5/2 kernel with ARD and a constant mean
   function for all experiments. The GP hyperparameters are fitted before proposing a
   new batch by optimizing the log-marginal likelihood."
  "We use a Matern-5/2 kernel with ARD for TuRBO and use the following bounds for the
   hyperparameters: (lengthscale) lambda_i in [0.005, 2.0], (signal variance)
   s^2 in [0.05, 20.0], (noise variance) sigma^2 in [0.0005, 0.1]."

App. C also specifies GPyTorch with CG solves and Lanczos log-determinants (following
Dong et al. 2017a) for scalability.

See PAPER_SPEC.md E8 for the marginal-likelihood objective and §3 (`gp.train_gp`) for shapes.
"""

from __future__ import annotations

import math

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from . import config

__all__ = ["GP", "train_gp", "HYPER_BOUNDS"]


# --- Hyperparameter box constraints (App. C) --------------------------------------
# Defined in config.py with their provenance (including the noise-bound conflict with the
# official code, PAPER_SPEC.md §10 A5); re-exported so existing imports keep working.
HYPER_BOUNDS = config.HYPER_BOUNDS


class GP(ExactGP):
    """Exact GP: ConstantMean + ScaleKernel(MaternKernel(nu=2.5, ARD)).  (App. C)

    The Matern-5/2 covariance (standard form, Rasmussen & Williams; the paper names the
    kernel but does not write it out) is

        r        = sqrt( sum_i (x_i - x'_i)^2 / lambda_i^2 )
        k(x, x') = s^2 * (1 + sqrt(5) r + (5/3) r^2) * exp(-sqrt(5) r)
    """

    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint,
                 outputscale_constraint, ard_dims):
        super().__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(
            nu=2.5,  # App. C: Matern-5/2
            ard_num_dims=ard_dims,
            lengthscale_constraint=lengthscale_constraint,
        )
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def train_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    use_ard: bool = True,
    num_steps: int = config.N_TRAINING_STEPS,
    lr: float = config.ADAM_LR,
    hypers: dict | None = None,
    max_cholesky_size: int = config.MAX_CHOLESKY_SIZE,
) -> GP:
    """Fit GP hyperparameters by maximizing the log-marginal likelihood.  (App. C)

    Args:
        train_x: (n, d) float64, **already rescaled to [0,1]^d** (App. C).
        train_y: (n,)  float64, **already standardized** (App. C; see utils.standardize).
        use_ard: App. C specifies ARD "for all experiments". The isotropic path exists
            only for the Sect. 3.6 regression ablation ("For the sake of illustration, we
            used an isotropic kernel").
        num_steps: gradient steps on the marginal likelihood.
        lr: Adam learning rate.
        hypers: optional state_dict to warm-start from (used by TuRBO-m, see turbo_m.py).
        max_cholesky_size: above this many points GPyTorch switches from Cholesky to
            CG + Lanczos (App. C).

    Returns the fitted model in eval mode.

    # [FROM_OFFICIAL_CODE] (PAPER_SPEC.md §10 B1, B2) The paper says only "optimizing the
    # log-marginal likelihood" -- it names no optimizer, learning rate, step count, or
    # initialization. All four come from uber-research/TuRBO gp.py L79-L89:
    #   Adam(lr=0.1), 50 steps, init outputscale=1.0, lengthscale=0.5, noise=0.005.
    # Alternatives: L-BFGS with random restarts (the more common GP-fitting choice, and one
    # that will generally find different optima than 50 Adam steps).
    """
    assert train_x.ndim == 2, "train_x must be (n, d)"
    assert train_y.ndim == 1, "train_y must be (n,)"
    assert train_x.shape[0] == train_y.shape[0]
    # Mixed dtypes are silently coerced by GPyTorch (the model follows train_x, the
    # likelihood follows train_y), which can downcast the whole fit to float32 without
    # warning. REPRODUCTION_NOTES C14 commits to float64, so require them to agree.
    assert train_x.dtype == train_y.dtype, (
        f"train_x is {train_x.dtype} but train_y is {train_y.dtype}; "
        "mixed dtypes silently downcast the GP fit"
    )
    assert train_x.device == train_y.device, (
        f"train_x is on {train_x.device} but train_y is on {train_y.device}"
    )

    dim = train_x.shape[1]

    lo, hi = HYPER_BOUNDS["lengthscale"]
    if use_ard:
        lengthscale_constraint = Interval(lo, hi)
        ard_dims = dim
    else:
        # [FROM_OFFICIAL_CODE] (PAPER_SPEC.md §10 A6) The paper gives no bound for the
        # isotropic case; the official code uses [0.005, sqrt(d)] (gp.py L52).
        lengthscale_constraint = Interval(lo, math.sqrt(dim))
        ard_dims = None

    outputscale_constraint = Interval(*HYPER_BOUNDS["outputscale"])
    noise_constraint = Interval(*HYPER_BOUNDS["noise"])

    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(
        device=train_x.device, dtype=train_y.dtype
    )
    model = GP(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
    ).to(device=train_x.device, dtype=train_x.dtype)

    model.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)

    if hypers:
        model.load_state_dict(hypers)
    else:
        model.initialize(**config.GP_INIT)

    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=lr)
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        for _ in range(num_steps):
            optimizer.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            optimizer.step()

    model.eval()
    likelihood.eval()
    return model
