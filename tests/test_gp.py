"""GP surrogate: constraints, shapes, and fitting.  (PAPER_SPEC.md E8; App. C)"""

import numpy as np
import pytest
import torch

from src.gp import HYPER_BOUNDS, train_gp
from src.utils import standardize


@pytest.fixture(scope="module")
def fitted():
    """A small GP fit on a smooth 4D function, standardized as App. C requires."""
    rng = np.random.default_rng(0)
    X = rng.random((40, 4))
    y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2]
    y_std, _, _ = standardize(y)
    gp = train_gp(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y_std, dtype=torch.float64),
        num_steps=50,
    )
    return gp, X, y_std


def test_hyper_bounds_match_appendix_c():
    """App. C: lambda in [0.005, 2.0], s^2 in [0.05, 20.0], sigma^2 in [0.0005, 0.1]."""
    assert HYPER_BOUNDS["lengthscale"] == (0.005, 2.0)
    assert HYPER_BOUNDS["outputscale"] == (0.05, 20.0)
    assert HYPER_BOUNDS["noise"] == (0.0005, 0.1)  # paper value, not the code's 0.2


def test_ard_gives_one_lengthscale_per_dimension(fitted):
    """App. C: "Matern-5/2 kernel with ARD"."""
    gp, X, _ = fitted
    ls = gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    assert ls.shape == (X.shape[1],)


def test_fitted_hyperparameters_respect_the_box_constraints(fitted):
    gp, _, _ = fitted
    ls = gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    os_ = float(gp.covar_module.outputscale.detach())
    noise = float(gp.likelihood.noise.detach())
    lo, hi = HYPER_BOUNDS["lengthscale"]
    assert np.all(ls >= lo) and np.all(ls <= hi)
    assert HYPER_BOUNDS["outputscale"][0] <= os_ <= HYPER_BOUNDS["outputscale"][1]
    assert HYPER_BOUNDS["noise"][0] <= noise <= HYPER_BOUNDS["noise"][1]


def test_posterior_shapes(fitted):
    gp, X, _ = fitted
    with torch.no_grad():
        post = gp(torch.as_tensor(X, dtype=torch.float64))
    assert post.mean.shape == (X.shape[0],)
    assert post.variance.shape == (X.shape[0],)


def test_posterior_interpolates_training_data(fitted):
    """A correctly fitted GP should track its own training targets closely."""
    gp, X, y_std = fitted
    with torch.no_grad():
        mean = gp(torch.as_tensor(X, dtype=torch.float64)).mean.numpy()
    assert np.corrcoef(mean, y_std)[0, 1] > 0.9


def test_fitting_improves_the_marginal_likelihood():
    """E8: hyperparameters are chosen by maximizing the log-marginal likelihood."""
    from gpytorch.mlls import ExactMarginalLogLikelihood

    rng = np.random.default_rng(1)
    X = rng.random((30, 3))
    y = np.cos(4 * X[:, 0]) + X[:, 1]
    y_std, _, _ = standardize(y)
    Xt = torch.as_tensor(X, dtype=torch.float64)
    yt = torch.as_tensor(y_std, dtype=torch.float64)

    def mll_of(steps):
        gp = train_gp(Xt, yt, num_steps=steps)
        gp.train()
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with torch.no_grad():
            return float(mll(gp(Xt), yt))

    assert mll_of(50) > mll_of(0), "50 Adam steps must beat the initialization"


def test_isotropic_kernel_has_a_single_lengthscale():
    """Sect. 3.6 ablation only: "we used an isotropic kernel"."""
    rng = np.random.default_rng(2)
    X = rng.random((20, 5))
    y_std, _, _ = standardize(rng.random(20))
    gp = train_gp(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y_std, dtype=torch.float64),
        use_ard=False,
        num_steps=10,
    )
    assert gp.covar_module.base_kernel.lengthscale.numel() == 1


def test_warm_start_from_cached_hypers_reproduces_the_model():
    """TuRBO-m caches hypers and refits with num_steps=0 (PAPER_SPEC.md §10 B8)."""
    rng = np.random.default_rng(3)
    X = torch.as_tensor(rng.random((25, 4)), dtype=torch.float64)
    y = torch.as_tensor(standardize(rng.random(25))[0], dtype=torch.float64)
    gp1 = train_gp(X, y, num_steps=20)
    hypers = {k: v.detach().clone() for k, v in gp1.state_dict().items()}
    gp2 = train_gp(X, y, num_steps=0, hypers=hypers)
    np.testing.assert_allclose(
        gp1.covar_module.base_kernel.lengthscale.detach().numpy(),
        gp2.covar_module.base_kernel.lengthscale.detach().numpy(),
        rtol=1e-10,
    )


def test_train_gp_rejects_mismatched_shapes():
    with pytest.raises(AssertionError):
        train_gp(torch.zeros((10, 3), dtype=torch.float64),
                 torch.zeros(9, dtype=torch.float64), num_steps=1)


def test_train_gp_rejects_2d_targets():
    """train_y must be (n,), not (n, 1) -- a classic silent-broadcast bug."""
    with pytest.raises(AssertionError):
        train_gp(torch.zeros((10, 3), dtype=torch.float64),
                 torch.zeros((10, 1), dtype=torch.float64), num_steps=1)
