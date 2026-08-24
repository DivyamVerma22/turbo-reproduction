"""Trust-region geometry, centering, and the resize rule.  (PAPER_SPEC.md E2, E3, E4)"""

import numpy as np
import pytest

from src.trust_region import (
    TR_DEFAULTS,
    TrustRegionState,
    default_failtol,
    is_success,
    select_center,
    trust_region_bounds,
    trust_region_weights,
)


# --- E2: geometry ------------------------------------------------------------------
def test_weights_have_unit_product():
    """prod(w) == 1 so that prod(L_i) == L^d -- the paper's stated volume invariant."""
    rng = np.random.default_rng(0)
    for d in (1, 2, 10, 200):
        w = trust_region_weights(rng.uniform(0.005, 2.0, d))
        assert np.prod(w) == pytest.approx(1.0, rel=1e-9)


def test_volume_invariant():
    """Sect. 2: side lengths are rescaled "while maintaining a total volume of L^d"."""
    lengthscales = np.array([0.1, 0.5, 2.0, 0.02])
    L = 0.4
    w = trust_region_weights(lengthscales)
    np.testing.assert_allclose(np.prod(w * L), L ** len(lengthscales), rtol=1e-9)


def test_weights_preserve_lengthscale_ratios():
    """L_i must stay proportional to lambda_i (Sect. 2)."""
    lengthscales = np.array([0.1, 0.4, 1.6])
    w = trust_region_weights(lengthscales)
    np.testing.assert_allclose(w / w[0], lengthscales / lengthscales[0], rtol=1e-9)


def test_weights_are_scale_invariant():
    """Only the ratios matter: scaling every lengthscale leaves w unchanged."""
    ls = np.array([0.1, 0.4, 1.6])
    np.testing.assert_allclose(trust_region_weights(ls), trust_region_weights(10 * ls), rtol=1e-9)


def test_isotropic_lengthscales_give_a_cube():
    w = trust_region_weights(np.full(5, 0.7))
    np.testing.assert_allclose(w, np.ones(5), rtol=1e-9)


def test_bounds_are_centered_and_clipped():
    x_center = np.array([[0.5, 0.5]])
    lb, ub = trust_region_bounds(x_center, np.array([1.0, 1.0]), 0.4)
    assert lb.shape == ub.shape == (1, 2)
    np.testing.assert_allclose(lb, [[0.3, 0.3]])
    np.testing.assert_allclose(ub, [[0.7, 0.7]])
    np.testing.assert_allclose((lb + ub) / 2, x_center)


def test_bounds_clip_at_the_domain_boundary():
    """App. D: candidates live in "the intersection of the TR and the domain [0,1]^d"."""
    lb, ub = trust_region_bounds(np.array([[0.02, 0.99]]), np.array([1.0, 1.0]), 0.8)
    assert lb.min() >= 0.0 and ub.max() <= 1.0
    assert lb[0, 0] == 0.0 and ub[0, 1] == 1.0


def test_bounds_reject_positive_lengthscale_violation():
    with pytest.raises(AssertionError):
        trust_region_weights(np.array([0.1, -1.0]))


# --- E3: centering -----------------------------------------------------------------
def test_center_is_best_observation_when_noise_free():
    X = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]])
    fX = np.array([3.0, 1.0, 2.0])
    np.testing.assert_allclose(select_center(X, fX), [[0.9, 0.9]])


def test_center_uses_posterior_mean_when_noisy():
    """Sect. 2: under noise, "the observation with the smallest posterior mean"."""
    X = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]])
    fX = np.array([3.0, 1.0, 2.0])          # best observed is row 1
    posterior = np.array([0.5, 2.0, 1.0])   # best posterior mean is row 0
    np.testing.assert_allclose(select_center(X, fX, posterior, noisy=True), [[0.1, 0.1]])


def test_noisy_center_requires_posterior_mean():
    with pytest.raises(ValueError, match="posterior_mean"):
        select_center(np.zeros((2, 2)), np.zeros(2), noisy=True)


# --- E4: success test and resize rule ----------------------------------------------
def test_success_requires_the_official_relative_margin():
    """PAPER_SPEC.md §10 A1: code demands improvement by 1e-3 * abs(f_best)."""
    assert not is_success(np.array([[99.99]]), 100.0, success_tol=1e-3)
    assert is_success(np.array([[99.0]]), 100.0, success_tol=1e-3)


def test_success_tol_zero_recovers_the_papers_literal_rule():
    """Sect. 2 as written: any strict improvement is a success."""
    assert is_success(np.array([[99.99]]), 100.0, success_tol=0.0)
    assert not is_success(np.array([[100.0]]), 100.0, success_tol=0.0)


def test_success_uses_the_batch_minimum():
    """App. D: "an improvement from at least one evaluation in the batch a success"."""
    assert is_success(np.array([[200.0], [50.0], [300.0]]), 100.0, success_tol=0.0)


def test_success_handles_negative_incumbent():
    """Edge case: the margin is relative to abs(f_best), so sign must not flip the test."""
    assert is_success(np.array([[-110.0]]), -100.0, success_tol=1e-3)
    assert not is_success(np.array([[-100.05]]), -100.0, success_tol=1e-3)


def test_expand_after_tau_succ_consecutive_successes():
    """Sect. 2: "After tau_succ consecutive successes, we double the size of the TR"."""
    s = TrustRegionState(length=0.4, succtol=3, failtol=4)
    s.update(success=True)
    s.update(success=True)
    assert s.length == 0.4, "must not expand before tau_succ"
    s.update(success=True)
    assert s.length == 0.8
    assert s.succcount == 0, "counters reset after a resize"


def test_shrink_after_tau_fail_consecutive_failures():
    """Sect. 2: "After tau_fail consecutive failures, we halve the size of the TR"."""
    s = TrustRegionState(length=0.4, succtol=3, failtol=3)
    for _ in range(2):
        s.update(success=False)
    assert s.length == 0.4
    s.update(success=False)
    assert s.length == 0.2
    assert s.failcount == 0


def test_success_resets_the_failure_counter():
    """Sect. 2 counts CONSECUTIVE failures."""
    s = TrustRegionState(length=0.4, succtol=3, failtol=3)
    s.update(success=False)
    s.update(success=False)
    s.update(success=True)
    assert s.failcount == 0
    s.update(success=False)
    assert s.length == 0.4, "the streak was broken, so no halving"


def test_expansion_is_capped_at_length_max():
    """Sect. 2: L <- min{L_max, 2L}."""
    s = TrustRegionState(length=1.0, succtol=1, failtol=4, length_max=1.6)
    s.update(success=True)
    assert s.length == 1.6


def test_failure_counter_accumulates_batch_size_for_turbo_m():
    """App. D: "add q_l to the failure counter"; halve when incrementing PAST the tolerance."""
    s = TrustRegionState(length=0.4, succtol=3, failtol=5)
    s.update(success=False, n_failures=3)
    assert s.length == 0.4
    s.update(success=False, n_failures=3)  # 6 >= 5 -> must trigger
    assert s.length == 0.2
    assert s.failcount == 0


def test_convergence_threshold():
    """App. D: "terminate the TR when L < L_min"."""
    s = TrustRegionState(length=TR_DEFAULTS["length_min"])
    assert not s.is_converged
    s.length /= 2.0
    assert s.is_converged


def test_reset_restores_initial_length_and_counters():
    s = TrustRegionState(length=0.01, succcount=2, failcount=1)
    s.hypers = {"a": 1}
    s.reset()
    assert s.length == TR_DEFAULTS["length_init"]
    assert s.succcount == 0 and s.failcount == 0 and s.hypers == {}


# --- tau_fail ----------------------------------------------------------------------
def test_failtol_follows_the_paper_for_turbo_1():
    """App. D: tau_fail = ceil(d/q)."""
    assert default_failtol(dim=14, batch_size=50) == 1
    assert default_failtol(dim=200, batch_size=100) == 2
    assert default_failtol(dim=10, batch_size=3) == 4


def test_failtol_for_turbo_m_uses_the_sequential_case():
    """App. D: TuRBO-m uses "the same tolerances as in the sequential case (q = 1)"."""
    assert default_failtol(dim=12, batch_size=50, n_trust_regions=5) == 12
