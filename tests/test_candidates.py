"""Candidate-set construction inside a trust region.  (PAPER_SPEC.md E6; App. D)"""

import numpy as np
import pytest

from src.candidates import create_candidates, n_candidates, perturbation_probability


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_candidate_set_size_matches_appendix_d():
    """App. D: "a candidate set of size min{100d, 5000}"."""
    assert n_candidates(6) == 600
    assert n_candidates(14) == 1400
    assert n_candidates(50) == 5000     # 100*50 == 5000, at the cap
    assert n_candidates(200) == 5000    # capped


def test_perturbation_probability_matches_appendix_d():
    """App. D: "with probability min{1, 20/d}"."""
    assert perturbation_probability(10) == 1.0
    assert perturbation_probability(20) == 1.0
    assert perturbation_probability(200) == pytest.approx(0.1)


def test_candidates_shape_and_dtype(rng):
    x_center = np.full((1, 6), 0.5)
    lb, ub = np.full((1, 6), 0.3), np.full((1, 6), 0.7)
    X = create_candidates(x_center, lb, ub, rng, n_cand=64)
    assert X.shape == (64, 6) and X.dtype == np.float64


def test_candidates_lie_inside_the_trust_region(rng):
    x_center = np.full((1, 8), 0.5)
    lb, ub = np.full((1, 8), 0.3), np.full((1, 8), 0.7)
    X = create_candidates(x_center, lb, ub, rng, n_cand=256)
    assert X.min() >= 0.3 - 1e-12 and X.max() <= 0.7 + 1e-12


def test_candidates_stay_in_the_unit_cube_at_the_boundary(rng):
    """The TR is clipped, so candidates must never leave [0,1]^d."""
    x_center = np.full((1, 5), 0.02)
    lb, ub = np.zeros((1, 5)), np.full((1, 5), 0.42)
    X = create_candidates(x_center, lb, ub, rng, n_cand=256)
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_low_dimension_perturbs_every_coordinate(rng):
    """For d <= 20, min{1, 20/d} == 1, so no coordinate keeps the center value."""
    d = 10
    x_center = np.full((1, d), 0.5)
    lb, ub = np.full((1, d), 0.2), np.full((1, d), 0.8)
    X = create_candidates(x_center, lb, ub, rng, n_cand=200)
    assert not np.any(np.all(X == x_center, axis=1)), "no candidate should equal the center"


def test_high_dimension_perturbs_only_a_subset(rng):
    """App. D: "In order to not perturb all coordinates at once" -- ~20/d per coordinate."""
    d = 200
    x_center = np.full((1, d), 0.5)
    lb, ub = np.full((1, d), 0.2), np.full((1, d), 0.8)
    X = create_candidates(x_center, lb, ub, rng, n_cand=500)
    frac = (X != x_center).mean()
    assert 0.06 < frac < 0.15, f"expected ~0.10 perturbed, got {frac:.3f}"


def test_every_candidate_perturbs_at_least_one_coordinate(rng):
    """Edge case (PAPER_SPEC.md §10 B6): an empty mask would duplicate the center."""
    d = 200
    x_center = np.full((1, d), 0.5)
    lb, ub = np.full((1, d), 0.2), np.full((1, d), 0.8)
    X = create_candidates(x_center, lb, ub, rng, n_cand=2000)
    n_changed = (X != x_center).sum(axis=1)
    assert n_changed.min() >= 1


def test_last_dimension_is_reachable_by_the_empty_mask_fix():
    """Regression: the official code's randint(0, d-1) can never pick the last dimension.

    With a degenerate TR (lb == ub == center) every perturbed value equals the center, so
    instead we check the fix directly: forcing many empty masks must touch dimension d-1.
    """
    d = 40  # d > 20 so that empty masks are possible
    x_center = np.full((1, d), 0.5)
    lb, ub = np.full((1, d), 0.0), np.full((1, d), 1.0)
    touched_last = False
    for seed in range(30):
        X = create_candidates(x_center, lb, ub, np.random.default_rng(seed), n_cand=500)
        if np.any(X[:, -1] != 0.5):
            touched_last = True
            break
    assert touched_last, "the last dimension must be reachable (upstream off-by-one not reproduced)"


def test_candidates_are_reproducible_from_the_generator():
    x_center = np.full((1, 6), 0.5)
    lb, ub = np.full((1, 6), 0.3), np.full((1, 6), 0.7)
    a = create_candidates(x_center, lb, ub, np.random.default_rng(7), n_cand=32)
    b = create_candidates(x_center, lb, ub, np.random.default_rng(7), n_cand=32)
    np.testing.assert_array_equal(a, b)


def test_a_new_candidate_set_is_drawn_each_call(rng):
    """App. D: "A new candidate set is generated for each batch"."""
    x_center = np.full((1, 6), 0.5)
    lb, ub = np.full((1, 6), 0.3), np.full((1, 6), 0.7)
    a = create_candidates(x_center, lb, ub, rng, n_cand=32)
    b = create_candidates(x_center, lb, ub, rng, n_cand=32)
    assert not np.array_equal(a, b)
