"""Domain rescaling, LHD, standardization.  (PAPER_SPEC.md §3, E7; App. A, App. C)"""

import numpy as np
import pytest

from src.utils import as_generator, from_unit_cube, latin_hypercube, standardize, to_unit_cube


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_unit_cube_roundtrip_is_identity(rng):
    lb = np.array([-5.0, -3.0, 0.0])
    ub = np.array([10.0, 4.0, 1.0])
    X = from_unit_cube(rng.random((20, 3)), lb, ub)
    np.testing.assert_allclose(from_unit_cube(to_unit_cube(X, lb, ub), lb, ub), X, atol=1e-12)


def test_to_unit_cube_maps_bounds_to_corners():
    lb, ub = np.array([-5.0, -3.0]), np.array([10.0, 4.0])
    corners = np.vstack([lb, ub])
    np.testing.assert_allclose(to_unit_cube(corners, lb, ub), [[0, 0], [1, 1]], atol=1e-12)


def test_to_unit_cube_rejects_degenerate_bounds():
    with pytest.raises(AssertionError):
        to_unit_cube(np.zeros((2, 2)), np.array([1.0, 0.0]), np.array([1.0, 1.0]))


def test_latin_hypercube_shape_and_range(rng):
    X = latin_hypercube(20, 6, rng)
    assert X.shape == (20, 6) and X.dtype == np.float64
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_latin_hypercube_is_stratified(rng):
    """The defining LHD property: one point per stratum in every dimension."""
    n, d = 25, 4
    X = latin_hypercube(n, d, rng)
    for j in range(d):
        strata = np.floor(X[:, j] * n).astype(int).clip(0, n - 1)
        assert len(np.unique(strata)) == n, f"dimension {j} is not stratified"


def test_latin_hypercube_is_reproducible_from_seed():
    a = latin_hypercube(10, 3, as_generator(42))
    b = latin_hypercube(10, 3, as_generator(42))
    np.testing.assert_array_equal(a, b)


def test_standardize_median_matches_paper_spec_e7():
    fX = np.array([1.0, 2.0, 3.0, 100.0])
    out, mu, sigma = standardize(fX, center="median")
    assert mu == pytest.approx(2.5)  # median, not mean -- PAPER_SPEC.md §10 A4
    assert sigma == pytest.approx(fX.std())
    np.testing.assert_allclose(out, (fX - mu) / sigma)


def test_standardize_mean_option():
    fX = np.array([1.0, 2.0, 3.0, 100.0])
    _, mu, _ = standardize(fX, center="mean")
    assert mu == pytest.approx(26.5)


def test_standardize_constant_values_do_not_produce_nan():
    """Edge case: a batch where every value is identical (sigma == 0)."""
    out, _, sigma = standardize(np.full(8, 3.0))
    assert sigma == 1.0
    assert np.all(np.isfinite(out))


def test_standardize_rejects_unknown_center():
    with pytest.raises(ValueError):
        standardize(np.arange(5.0), center="mode")
