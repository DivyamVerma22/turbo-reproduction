"""Thompson draws and batch selection.  (PAPER_SPEC.md E5, E7)"""

import numpy as np
import pytest
import torch

from src.gp import train_gp
from src.thompson import select_candidates, select_candidates_across, thompson_draws
from src.utils import standardize


@pytest.fixture(scope="module")
def gp_and_cands():
    rng = np.random.default_rng(0)
    X = rng.random((30, 3))
    y_std, mu, sigma = standardize(np.sin(3 * X[:, 0]) - X[:, 1])
    gp = train_gp(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y_std, dtype=torch.float64),
        num_steps=30,
    )
    X_cand = rng.random((50, 3))
    return gp, X_cand, mu, sigma


# --- draws -------------------------------------------------------------------------
def test_draw_shape_is_n_cand_by_q(gp_and_cands):
    """E5: q independent realizations over the candidate set."""
    gp, X_cand, mu, sigma = gp_and_cands
    y = thompson_draws(gp, X_cand, batch_size=7, mu=mu, sigma=sigma)
    assert y.shape == (50, 7) and y.dtype == np.float64
    assert np.all(np.isfinite(y))


def test_draws_are_independent_across_the_batch(gp_and_cands):
    """App. E: "This process is repeated independently for multiple suggestions (q > 1)"."""
    gp, X_cand, mu, sigma = gp_and_cands
    y = thompson_draws(gp, X_cand, batch_size=5, mu=mu, sigma=sigma)
    assert not np.allclose(y[:, 0], y[:, 1])


def test_destandardization_is_applied(gp_and_cands):
    """E7: draws must be mapped back through (mu, sigma) before they are compared."""
    gp, X_cand, _, _ = gp_and_cands
    torch.manual_seed(0)
    raw = thompson_draws(gp, X_cand, batch_size=3, mu=0.0, sigma=1.0)
    torch.manual_seed(0)
    shifted = thompson_draws(gp, X_cand, batch_size=3, mu=10.0, sigma=2.0)
    np.testing.assert_allclose(shifted, 10.0 + 2.0 * raw, rtol=1e-10)


def test_predictive_draws_are_noisier_than_latent_draws(gp_and_cands):
    """PAPER_SPEC.md §10 B7: the predictive distribution adds sigma^2 observation noise."""
    gp, X_cand, mu, sigma = gp_and_cands
    lat = np.std([thompson_draws(gp, X_cand, 1, mu, sigma, use_predictive=False)
                  for _ in range(15)], axis=0).mean()
    pred = np.std([thompson_draws(gp, X_cand, 1, mu, sigma, use_predictive=True)
                   for _ in range(15)], axis=0).mean()
    assert pred > lat


# --- single-TR selection -----------------------------------------------------------
def test_select_returns_q_points_of_the_right_shape():
    X_cand = np.arange(20, dtype=np.float64).reshape(10, 2)
    y_cand = np.random.default_rng(0).random((10, 4))
    X_next = select_candidates(X_cand, y_cand.copy())
    assert X_next.shape == (4, 2)


def test_select_picks_the_argmin_of_each_realization():
    X_cand = np.arange(10, dtype=np.float64).reshape(5, 2)
    y_cand = np.array([
        [5.0, 9.0],
        [1.0, 8.0],   # best for column 0
        [7.0, 0.5],   # best for column 1
        [6.0, 4.0],
        [8.0, 3.0],
    ])
    X_next = select_candidates(X_cand, y_cand.copy())
    np.testing.assert_allclose(X_next[0], X_cand[1])
    np.testing.assert_allclose(X_next[1], X_cand[2])


def test_select_never_returns_a_duplicate():
    """PAPER_SPEC.md §10 B5: a chosen candidate is masked out of every column."""
    X_cand = np.arange(12, dtype=np.float64).reshape(6, 2)
    # Column 0 is the best everywhere -> without de-duplication all q picks collide.
    y_cand = np.tile(np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]), (1, 4))
    X_next = select_candidates(X_cand, y_cand.copy())
    assert len(np.unique(X_next, axis=0)) == 4


# --- cross-TR selection (the bandit) -----------------------------------------------
def test_select_across_shapes():
    rng = np.random.default_rng(0)
    X_cand = rng.random((3, 10, 4))
    y_cand = rng.random((3, 10, 5))
    X_next, idx_next = select_candidates_across(X_cand, y_cand.copy())
    assert X_next.shape == (5, 4)
    assert idx_next.shape == (5, 1) and idx_next.dtype == int


def test_bandit_allocates_to_the_trust_region_with_better_draws():
    """Sect. 2: x_i in argmin_l argmin_x f_l^(i) -- allocation follows the draws."""
    X_cand = np.random.default_rng(0).random((2, 10, 3))
    y_cand = np.ones((2, 10, 4))
    y_cand[1] = -5.0  # TR 1 dominates every realization
    _, idx_next = select_candidates_across(X_cand, y_cand.copy())
    assert np.all(idx_next == 1)


def test_bandit_can_split_a_batch_across_trust_regions():
    X_cand = np.random.default_rng(0).random((2, 4, 2))
    y_cand = np.ones((2, 4, 2))
    y_cand[0, :, 0] = -1.0   # TR 0 wins realization 0
    y_cand[1, :, 1] = -1.0   # TR 1 wins realization 1
    _, idx_next = select_candidates_across(X_cand, y_cand.copy())
    assert set(idx_next.ravel()) == {0, 1}


def test_select_across_rejects_non_finite_draws():
    X_cand = np.zeros((2, 3, 2))
    y_cand = np.zeros((2, 3, 2))
    y_cand[0, 0, 0] = np.nan
    with pytest.raises(AssertionError):
        select_candidates_across(X_cand, y_cand)


def test_cross_tr_comparison_needs_destandardized_draws():
    """E7: two TRs with different (mu, sigma) are only comparable after de-standardizing.

    TR 0's draws are genuinely better in objective units; TR 1 only looks better in
    standardized units. The bandit must follow the de-standardized values.
    """
    X_cand = np.zeros((2, 3, 1))
    y_std = np.array([[[-1.0]] * 3, [[-2.0]] * 3])  # TR 1 looks better standardized

    # TR 0: mu=0, sigma=1  -> -1.0 ;  TR 1: mu=10, sigma=1 -> 8.0
    y_destd = np.stack([0.0 + 1.0 * y_std[0], 10.0 + 1.0 * y_std[1]])
    _, idx = select_candidates_across(X_cand, y_destd.copy())
    assert np.all(idx == 0), "de-standardized comparison must prefer TR 0"

    _, idx_wrong = select_candidates_across(X_cand, y_std.copy())
    assert np.all(idx_wrong == 1), "raw standardized draws pick the wrong TR"
