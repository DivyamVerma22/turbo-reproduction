"""Mutation-resistant invariant tests.

Every test names the paper requirement or implementation invariant it protects. These are
written against plausible WRONG implementations: each one should fail if the corresponding
detail is changed, not merely if the code crashes.

Grouped as:
  1. tensor shapes                 5. finite outputs
  2. mathematical invariants       6. gradient flow
  3. edge cases and masks          7. serialization round trips
  4. determinism under a seed      8. minimal end-to-end evaluation path
"""

import numpy as np
import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from src.benchmarks import Benchmark, get_benchmark
from src.candidates import create_candidates, n_candidates, perturbation_probability
from src.evaluate import best_so_far, mean_standard_error, run_replications, summarize
from src.gp import HYPER_BOUNDS, train_gp
from src.thompson import select_candidates, select_candidates_across, thompson_draws
from src.trust_region import (
    TrustRegionState,
    default_failtol,
    is_success,
    select_center,
    trust_region_bounds,
    trust_region_weights,
)
from src.turbo_1 import Turbo1
from src.turbo_m import TurboM
from src.utils import as_generator, from_unit_cube, latin_hypercube, standardize, to_unit_cube


def quadratic(dim=3) -> Benchmark:
    return Benchmark("quadratic", lambda x: float(np.sum(x**2)),
                     np.full(dim, -2.0), np.full(dim, 2.0), dim, 0.0)


@pytest.fixture(scope="module")
def fitted_gp():
    """A GP fit on 24 points of a smooth 3D function, standardized per App. C."""
    rng = as_generator(0)
    X = rng.random((24, 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 - 0.5 * X[:, 2]
    y_std, mu, sigma = standardize(y)
    gp = train_gp(torch.as_tensor(X, dtype=torch.float64),
                  torch.as_tensor(y_std, dtype=torch.float64), num_steps=40)
    return gp, X, y_std, mu, sigma


# =====================================================================================
# 1. Tensor shapes
# =====================================================================================
def test_trust_region_bounds_shape_is_1_by_d_not_d():
    """Protects: PAPER_SPEC.md §3 -- bounds are (1, d) so they broadcast against
    (n_cand, d) candidates. A (d,) return would broadcast silently but break `np.tile`
    of the center and the `x_center.shape == lb.shape` assertion in create_candidates."""
    lb, ub = trust_region_bounds(np.full((1, 5), 0.5), np.ones(5), 0.4)
    assert lb.shape == (1, 5) and ub.shape == (1, 5)
    assert lb.ndim == 2 and ub.ndim == 2


def test_thompson_draws_are_n_cand_by_q_not_q_by_n_cand():
    """Protects: PAPER_SPEC.md §3 / E5 -- y_cand is (n_cand, q). GPyTorch's `.sample`
    returns (q, n_cand), so a missing transpose gives a transposed array that still
    'works' for square shapes. This test uses n_cand != q to catch that."""
    rng = as_generator(1)
    X = torch.as_tensor(rng.random((12, 2)), dtype=torch.float64)
    y = torch.as_tensor(standardize(rng.random(12))[0], dtype=torch.float64)
    gp = train_gp(X, y, num_steps=5)
    y_cand = thompson_draws(gp, rng.random((17, 2)), batch_size=3, seed=0)
    assert y_cand.shape == (17, 3), "n_cand must be axis 0, q axis 1"


def test_optimizer_history_row_count_equals_n_evals():
    """Protects: evaluation-budget accounting -- X, fX and n_evals must agree, or the
    best-so-far trace in evaluate.py is misaligned with the evaluation axis of Fig. 2-4."""
    b = quadratic()
    opt = Turbo1(f=b, lb=b.lb, ub=b.ub, n_init=6, max_evals=18, batch_size=3,
                 n_training_steps=3, seed=0).optimize()
    assert opt.X.shape[0] == opt.fX.shape[0] == opt.n_evals
    assert opt.fX.shape[1] == 1, "fX must stay a column vector (n, 1)"


def test_turbo_m_ownership_vector_partitions_the_history():
    """Protects: App. D -- every evaluated point belongs to exactly one TR, or is orphaned
    by a restart (-1). Counts must sum to n_evals; a leak means some TR's GP silently
    trains on the wrong data."""
    b = quadratic()
    opt = TurboM(f=b, lb=b.lb, ub=b.ub, n_init=4, max_evals=30, n_trust_regions=3,
                 batch_size=3, n_training_steps=3, seed=0).optimize()
    counts = [int((opt._idx.ravel() == i).sum()) for i in range(opt.n_trust_regions)]
    orphaned = int((opt._idx.ravel() == -1).sum())
    assert sum(counts) + orphaned == opt.n_evals


# =====================================================================================
# 2. Mathematical invariants
# =====================================================================================
def test_trust_region_side_widths_are_proportional_to_lengthscales():
    """Protects: §2 "Trust regions" (E2) -- L_i = lambda_i L / (prod lambda_j)^(1/d).
    Existing tests use isotropic lengthscales, which cannot distinguish L_i proportional
    to lambda_i from L_i proportional to 1/lambda_i. This uses anisotropic lengthscales
    with an interior center so no clipping occurs."""
    lengthscales = np.array([0.1, 0.4, 1.6])
    L = 0.2
    lb, ub = trust_region_bounds(np.full((1, 3), 0.5), lengthscales, L)
    widths = (ub - lb).ravel()
    np.testing.assert_allclose(widths / widths[0], lengthscales / lengthscales[0], rtol=1e-9)
    np.testing.assert_allclose(np.prod(widths), L**3, rtol=1e-9)


def test_trust_region_width_equals_base_length_when_isotropic():
    """Protects: §2 -- the base side length L is the WIDTH, not the half-width. An
    implementation using +/- L instead of +/- L/2 doubles every trust region."""
    lb, ub = trust_region_bounds(np.full((1, 4), 0.5), np.ones(4), 0.3)
    np.testing.assert_allclose((ub - lb).ravel(), np.full(4, 0.3), rtol=1e-12)


def test_standardize_uses_population_std_ddof_zero():
    """Protects: App. C / E7 -- standardization divides by numpy's default std (ddof=0),
    matching the official implementation. Switching to ddof=1 rescales every GP target and
    silently shifts the fitted signal variance."""
    fX = np.array([1.0, 2.0, 4.0, 8.0])
    _, _, sigma = standardize(fX)
    assert sigma == pytest.approx(fX.std(ddof=0))
    assert sigma != pytest.approx(fX.std(ddof=1))


def test_standard_error_uses_sample_std_ddof_one():
    """Protects: §3 -- "mean performances with one standard error". The SEM of a sample
    uses ddof=1; ddof=0 understates the error bars in every reported figure."""
    traces = np.array([[1.0], [2.0], [6.0]])
    _, sem = mean_standard_error(traces)
    expected = traces.ravel().std(ddof=1) / np.sqrt(3)
    assert sem[0] == pytest.approx(expected)
    assert sem[0] != pytest.approx(traces.ravel().std(ddof=0) / np.sqrt(3))


def test_center_is_invariant_to_standardization():
    """Protects: §2 (E3) -- the center is the argmin, so it must not change when values
    are affinely rescaled. turbo_1 passes STANDARDIZED values to select_center; if the
    implementation ever compared against a threshold instead of taking an argmin, this
    would break."""
    X = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]])
    fX = np.array([3.0, 1.0, 2.0])
    fX_std, _, _ = standardize(fX)
    np.testing.assert_allclose(select_center(X, fX), select_center(X, fX_std))


def test_thompson_draw_mean_converges_to_the_posterior_mean(fitted_gp):
    """Protects: §2 (E5) -- draws come from the GP POSTERIOR at the candidates. An
    implementation sampling the prior, or returning the mean plus unrelated noise, would
    pass the shape and independence tests but fail this one."""
    gp, _, _, _, _ = fitted_gp
    X_cand = as_generator(3).random((6, 3))
    draws = thompson_draws(gp, X_cand, batch_size=600, seed=0)
    with torch.no_grad():
        post_mean = gp(torch.as_tensor(X_cand, dtype=torch.float64)).mean.numpy()
    np.testing.assert_allclose(draws.mean(axis=1), post_mean, atol=0.25)


def test_destandardization_is_an_exact_affine_map(fitted_gp):
    """Protects: E7 -- y = mu + sigma * y_std exactly. In TuRBO-m this map is what makes
    draws from different TRs comparable; an approximate or omitted map breaks the bandit."""
    gp, _, _, _, _ = fitted_gp
    X_cand = as_generator(4).random((8, 3))
    raw = thompson_draws(gp, X_cand, 4, mu=0.0, sigma=1.0, seed=7)
    mapped = thompson_draws(gp, X_cand, 4, mu=-3.5, sigma=2.5, seed=7)
    np.testing.assert_allclose(mapped, -3.5 + 2.5 * raw, rtol=1e-12)


def test_expansion_and_shrink_are_exact_factors_of_two():
    """Protects: §2 -- "we double the size" / "we halve the size". A factor other than 2
    (e.g. 1.5) changes the trust-region schedule while leaving every counter test green."""
    s = TrustRegionState(length=0.4, succtol=1, failtol=1, length_max=100.0)
    s.update(success=True)
    assert s.length == pytest.approx(0.8)
    s.update(success=False)
    assert s.length == pytest.approx(0.4)


def test_trust_region_defaults_match_appendix_d_literally():
    """Protects: App. D -- "tau_succ = 3, tau_fail = ceil(d/q), L_min = 2^-7, L_max = 1.6,
    and L_init = 0.8".

    Found by mutation testing: every other trust-region test either passes tolerances in
    explicitly or reads TR_DEFAULTS to assert TR_DEFAULTS, so changing these constants was
    caught by NOTHING. Asserted here against literals from the paper."""
    from src.trust_region import TR_DEFAULTS

    assert TR_DEFAULTS["succtol"] == 3
    assert TR_DEFAULTS["length_init"] == 0.8
    assert TR_DEFAULTS["length_max"] == 1.6
    assert TR_DEFAULTS["length_min"] == 0.0078125  # 2^-7, written out so the test is
    assert TR_DEFAULTS["length_min"] == 2.0**-7    # not self-referential


def test_a_default_trust_region_state_starts_at_appendix_d_values():
    """Companion: the defaults must actually reach a constructed state, not just live in
    the constants dict."""
    s = TrustRegionState()
    assert s.length == 0.8 and s.succtol == 3
    assert s.length_max == 1.6 and s.length_min == 2.0**-7
    assert s.succcount == 0 and s.failcount == 0


def test_optimizer_adopts_appendix_d_defaults_without_overrides():
    """Protects: App. D -- "In all experiments, we use the following hyperparameters".
    A constructed optimizer (before any update) must carry the paper's schedule."""
    b = quadratic()
    opt = Turbo1(f=b, lb=b.lb, ub=b.ub, n_init=4, max_evals=12, batch_size=2)
    assert opt.state.length == 0.8
    assert opt.state.succtol == 3
    assert opt.state.length_max == 1.6
    assert opt.state.length_min == 2.0**-7
    assert opt.failtol == default_failtol(b.dim, 2)


def test_best_so_far_never_increases_and_starts_at_the_first_value():
    """Protects: §3 -- convergence curves are running minima of the evaluation sequence."""
    fX = np.array([5.0, 7.0, 3.0, 9.0, 1.0])
    trace = best_so_far(fX)
    assert trace[0] == fX[0]
    assert np.all(np.diff(trace) <= 0)
    assert trace[-1] == fX.min()


# =====================================================================================
# 3. Edge cases and masks
# =====================================================================================
def test_candidates_are_a_scrambled_sobol_sequence_not_uniform_random():
    """Protects: App. D -- "a scrambled Sobol sequence within the intersection of the TR
    and the domain". Substituting uniform random sampling passes EVERY other candidate
    test (shape, bounds, determinism, mask fraction). This pins the actual generator by
    reconstructing the draw from the same seed stream."""
    dim, n_cand = 4, 32          # dim <= 20 so the mask is all-True and X_cand == pert
    x_center = np.full((1, dim), 0.5)
    lb, ub = np.full((1, dim), 0.25), np.full((1, dim), 0.75)

    X_cand = create_candidates(x_center, lb, ub, as_generator(11), n_cand=n_cand)

    replay = as_generator(11)
    sobol_seed = int(replay.integers(0, 1_000_000))
    expected = SobolEngine(dim, scramble=True, seed=sobol_seed).draw(n_cand).to(
        dtype=torch.float64).numpy()
    expected = lb + (ub - lb) * expected
    np.testing.assert_allclose(X_cand, expected, rtol=1e-12)


def test_unperturbed_coordinates_keep_the_center_value_exactly():
    """Protects: App. D -- "the value of the center otherwise". A wrong implementation
    that jitters the unperturbed coordinates would still produce the right mask fraction."""
    dim = 200                     # d > 20 so p = 20/d < 1 and some coords stay at center
    x_center = np.full((1, dim), 0.5)
    X = create_candidates(x_center, np.zeros((1, dim)), np.ones((1, dim)),
                          as_generator(5), n_cand=200)
    untouched = X[X == 0.5]
    assert untouched.size > 0, "expected some coordinates to keep the center value"
    assert np.all(untouched == 0.5), "unperturbed coordinates must be exactly the center"


def test_perturbation_probability_is_exactly_one_at_the_d_equals_20_boundary():
    """Protects: App. D -- min{1, 20/d}. Off-by-one at the boundary (e.g. 20/(d+1)) would
    silently stop perturbing every coordinate for the 10D synthetic suite."""
    assert perturbation_probability(20) == 1.0
    assert perturbation_probability(19) == 1.0
    assert perturbation_probability(21) == pytest.approx(20.0 / 21.0)


def test_candidate_count_is_exact_at_the_5000_cap_boundary():
    """Protects: App. D -- min{100d, 5000}. d = 50 is exactly at the cap."""
    assert n_candidates(49) == 4900
    assert n_candidates(50) == 5000
    assert n_candidates(51) == 5000


def test_degenerate_trust_region_produces_candidates_at_the_center():
    """Edge case: once L collapses toward L_min the TR can be numerically empty in some
    coordinate. Candidates must stay inside [lb, ub] rather than producing NaNs."""
    x_center = np.full((1, 3), 0.5)
    X = create_candidates(x_center, x_center.copy(), x_center.copy(),
                          as_generator(0), n_cand=16)
    assert np.all(np.isfinite(X))
    np.testing.assert_allclose(X, np.tile(x_center, (16, 1)))


def test_trust_region_clipped_at_a_corner_stays_in_the_unit_cube():
    """Protects: App. D -- "the intersection of the TR and the domain [0,1]^d" when the
    center sits in a corner and L_max would otherwise overshoot both faces."""
    lb, ub = trust_region_bounds(np.zeros((1, 3)), np.ones(3), 1.6)
    assert np.all(lb == 0.0) and np.all(ub <= 1.0)
    X = create_candidates(np.zeros((1, 3)), lb, ub, as_generator(0), n_cand=64)
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_success_with_a_zero_incumbent_uses_a_strict_improvement():
    """Edge case for the relative margin in §10 A1: 1e-3 * abs(0) == 0, so at f_best = 0
    the rule degenerates to strict improvement. Guards against a NaN or sign error."""
    assert is_success(np.array([[-0.001]]), 0.0)
    assert not is_success(np.array([[0.0]]), 0.0)


def test_latin_hypercube_with_a_single_point_stays_in_range():
    """Edge case: n_init = 1. centers = [0.5] and the jitter is +/- 0.5, so the point must
    still land inside [0, 1]."""
    X = latin_hypercube(1, 4, as_generator(0))
    assert X.shape == (1, 4)
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_failtol_is_at_least_one_for_huge_batches():
    """Protects: App. D -- tau_fail = ceil(d/q). With q >> d this must not round to 0, or
    the trust region would halve on every batch regardless of progress."""
    assert default_failtol(dim=6, batch_size=1000) == 1


def test_single_dimension_problem_is_well_formed():
    """Edge case d = 1: the geometric-mean normalization degenerates to w = [1]."""
    np.testing.assert_allclose(trust_region_weights(np.array([0.37])), np.array([1.0]))
    lb, ub = trust_region_bounds(np.array([[0.5]]), np.array([0.37]), 0.4)
    np.testing.assert_allclose((ub - lb).ravel(), [0.4])


# =====================================================================================
# 4. Determinism under a fixed seed
# =====================================================================================
def test_turbo_m_is_reproducible_from_a_seed():
    """Protects: PAPER_SPEC.md §10 C5. TuRBO-1 determinism is covered elsewhere; the
    multi-TR path draws from the generator m times per batch, so its consumption order is
    a separate risk."""
    b = quadratic()
    kw = dict(f=b, lb=b.lb, ub=b.ub, n_init=4, max_evals=24, n_trust_regions=2,
              batch_size=4, n_training_steps=3, seed=99)
    a, c = TurboM(**kw).optimize(), TurboM(**kw).optimize()
    np.testing.assert_allclose(a.X, c.X)
    np.testing.assert_array_equal(a._idx, c._idx)


def test_turbo_m_differs_across_seeds():
    """Complement to the above: identical trajectories across seeds would mean the
    generator is not actually threaded into the multi-TR path."""
    b = quadratic()
    kw = dict(f=b, lb=b.lb, ub=b.ub, n_init=4, max_evals=24, n_trust_regions=2,
              batch_size=4, n_training_steps=3)
    assert not np.allclose(TurboM(**kw, seed=1).optimize().X,
                           TurboM(**kw, seed=2).optimize().X)


def test_thompson_draws_are_reproducible_from_an_explicit_seed(fitted_gp):
    """Protects: C5 -- GPyTorch samples from torch's GLOBAL RNG, so the draw must be
    seeded explicitly. Without the fork+seed, two calls diverge."""
    gp, _, _, _, _ = fitted_gp
    X_cand = as_generator(6).random((10, 3))
    np.testing.assert_allclose(thompson_draws(gp, X_cand, 3, seed=42),
                               thompson_draws(gp, X_cand, 3, seed=42))


def test_seeded_draw_does_not_disturb_the_global_torch_rng(fitted_gp):
    """Protects: the fork in thompson_draws. If it used a bare torch.manual_seed, a caller's
    own RNG stream would be silently reset mid-run."""
    gp, _, _, _, _ = fitted_gp
    torch.manual_seed(1234)
    before = torch.rand(3)
    torch.manual_seed(1234)
    thompson_draws(gp, as_generator(0).random((5, 3)), 2, seed=999)
    after = torch.rand(3)
    torch.testing.assert_close(before, after)


def test_benchmarks_are_deterministic():
    """Protects: the synthetic objectives are pure functions -- required for the
    reproducibility guarantee and for App. A's noise-free comparison."""
    b = get_benchmark("ackley10")
    x = as_generator(0).random(10)
    assert b(x) == b(x)


# =====================================================================================
# 5. Finite outputs
# =====================================================================================
def test_extreme_objective_scale_still_yields_a_finite_fit_and_draws():
    """Protects: App. C -- standardization exists so the GP sees O(1) targets. Objectives
    at 1e8 must not overflow the fit or produce non-finite Thompson draws."""
    rng = as_generator(0)
    X = rng.random((20, 3))
    y_std, mu, sigma = standardize(rng.random(20) * 1e8 + 1e8)
    assert np.all(np.isfinite(y_std))
    gp = train_gp(torch.as_tensor(X, dtype=torch.float64),
                  torch.as_tensor(y_std, dtype=torch.float64), num_steps=10)
    draws = thompson_draws(gp, rng.random((10, 3)), 3, mu=mu, sigma=sigma, seed=0)
    assert np.all(np.isfinite(draws))


def test_tiny_objective_scale_still_yields_a_finite_fit():
    """Complement: values at 1e-8 must not collapse sigma to zero and divide by ~0."""
    rng = as_generator(1)
    y_std, _, sigma = standardize(rng.random(16) * 1e-8)
    assert sigma > 0 and np.all(np.isfinite(y_std))


def test_optimizer_never_records_a_non_finite_objective_value():
    """Protects: the evaluation loop -- a NaN entering fX would poison the GP fit and the
    best-so-far trace without raising."""
    b = quadratic()
    opt = Turbo1(f=b, lb=b.lb, ub=b.ub, n_init=6, max_evals=18, batch_size=3,
                 n_training_steps=3, seed=0).optimize()
    assert np.all(np.isfinite(opt.fX)) and np.all(np.isfinite(opt.X))


def test_marginal_likelihood_is_finite_for_near_duplicate_points():
    """Edge case: as a trust region collapses, training points become nearly identical and
    the kernel matrix approaches singular. The noise floor (App. C: sigma^2 >= 5e-4) should
    keep the marginal likelihood finite."""
    X = np.full((8, 3), 0.5) + as_generator(0).random((8, 3)) * 1e-9
    y_std, _, _ = standardize(as_generator(1).random(8))
    Xt = torch.as_tensor(X, dtype=torch.float64)
    yt = torch.as_tensor(y_std, dtype=torch.float64)
    gp = train_gp(Xt, yt, num_steps=5)
    gp.train()
    loss = -ExactMarginalLogLikelihood(gp.likelihood, gp)(gp(Xt), yt)
    assert torch.isfinite(loss)


# =====================================================================================
# 6. Gradient flow
# =====================================================================================
def _fresh_gp_for_grads():
    rng = as_generator(0)
    X = torch.as_tensor(rng.random((16, 3)), dtype=torch.float64)
    y = torch.as_tensor(standardize(rng.random(16))[0], dtype=torch.float64)
    gp = train_gp(X, y, num_steps=0)
    gp.train()
    return gp, X, y


def test_every_gp_hyperparameter_receives_a_finite_gradient():
    """Protects: App. C -- ALL of {constant mean, ARD lengthscales, signal variance, noise}
    are fitted by maximizing the marginal likelihood. A parameter detached from the graph
    would silently stay at its initial value forever."""
    gp, X, y = _fresh_gp_for_grads()
    loss = -ExactMarginalLogLikelihood(gp.likelihood, gp)(gp(X), y)
    loss.backward()
    named = dict(gp.named_parameters())
    assert named, "GP exposes no trainable parameters"
    for name, p in named.items():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.all(torch.isfinite(p.grad)), f"{name} has a non-finite gradient"


def test_ard_lengthscale_gradient_is_per_dimension_and_nonzero():
    """Protects: App. C -- ARD means one lengthscale PER DIMENSION, each independently
    fitted. A shared/isotropic lengthscale would give a single gradient entry, and the
    trust-region rescaling in E2 would degenerate to a cube."""
    gp, X, y = _fresh_gp_for_grads()
    loss = -ExactMarginalLogLikelihood(gp.likelihood, gp)(gp(X), y)
    loss.backward()
    g = gp.covar_module.base_kernel.raw_lengthscale.grad
    assert g.numel() == X.shape[1], "expected one lengthscale gradient per dimension"
    assert torch.any(g != 0), "lengthscales are not being trained"


def test_objective_evaluation_carries_no_gradient():
    """Protects: §1 / PAPER_SPEC.md §6 -- the black-box objective is gradient-free
    ("closed form expressions and derivatives are unavailable"). Evaluations must be plain
    floats, never autograd-tracked tensors."""
    b = quadratic()
    opt = Turbo1(f=b, lb=b.lb, ub=b.ub, n_init=4, max_evals=10, batch_size=2,
                 n_training_steps=2, seed=0).optimize()
    assert isinstance(opt.fX, np.ndarray) and opt.fX.dtype == np.float64


# =====================================================================================
# 7. Serialization / checkpoint round trips
# =====================================================================================
def test_gp_state_dict_round_trip_reproduces_the_posterior(fitted_gp, tmp_path):
    """Protects: PAPER_SPEC.md §10 B8 -- TuRBO-m CACHES fitted hypers and reuses them for
    trust regions that received no points. If a state_dict round trip did not restore the
    exact posterior, those trust regions would silently drift."""
    gp, X, y_std, _, _ = fitted_gp
    path = tmp_path / "hypers.pt"
    torch.save(gp.state_dict(), path)

    Xt = torch.as_tensor(X, dtype=torch.float64)
    yt = torch.as_tensor(y_std, dtype=torch.float64)
    restored = train_gp(Xt, yt, num_steps=0, hypers=torch.load(path, weights_only=True))

    with torch.no_grad():
        a = gp(Xt).mean.numpy()
        b = restored(Xt).mean.numpy()
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_cached_hypers_survive_a_round_trip_of_all_four_hyperparameters(fitted_gp, tmp_path):
    """Companion to the above: check the VALUES, not just the resulting posterior, so a
    partial state_dict (e.g. dropping the noise term) is caught."""
    gp, X, y_std, _, _ = fitted_gp
    path = tmp_path / "h.pt"
    torch.save(gp.state_dict(), path)
    restored = train_gp(torch.as_tensor(X, dtype=torch.float64),
                        torch.as_tensor(y_std, dtype=torch.float64),
                        num_steps=0, hypers=torch.load(path, weights_only=True))
    for name in ("covar_module.base_kernel.lengthscale", "covar_module.outputscale",
                 "likelihood.noise"):
        orig = gp.get_submodule(name.rsplit(".", 1)[0])
        rest = restored.get_submodule(name.rsplit(".", 1)[0])
        attr = name.rsplit(".", 1)[1]
        torch.testing.assert_close(getattr(orig, attr), getattr(rest, attr))


def test_gp_hyperparameter_initialization_values(tmp_path):
    """Protects: PAPER_SPEC.md §10 B2 (audit item U10) -- the initialization values
    (outputscale 1.0, lengthscale 0.5, noise 0.005) come from the official code and are
    otherwise unasserted, so a silent edit would go unnoticed. Also confirms each sits
    inside its App. C box constraint."""
    rng = as_generator(0)
    gp = train_gp(torch.as_tensor(rng.random((10, 3)), dtype=torch.float64),
                  torch.as_tensor(standardize(rng.random(10))[0], dtype=torch.float64),
                  num_steps=0)
    assert float(gp.covar_module.outputscale) == pytest.approx(1.0, abs=1e-6)
    assert float(gp.covar_module.base_kernel.lengthscale.mean()) == pytest.approx(0.5, abs=1e-6)
    assert float(gp.likelihood.noise) == pytest.approx(0.005, abs=1e-6)
    assert HYPER_BOUNDS["outputscale"][0] <= 1.0 <= HYPER_BOUNDS["outputscale"][1]
    assert HYPER_BOUNDS["lengthscale"][0] <= 0.5 <= HYPER_BOUNDS["lengthscale"][1]
    assert HYPER_BOUNDS["noise"][0] <= 0.005 <= HYPER_BOUNDS["noise"][1]


def test_gp_fitting_constants_match_the_official_code():
    """Protects: PAPER_SPEC.md §10 B1 -- Adam(lr=0.1), 50 steps, Cholesky switch at 2000.

    Found by mutation testing after configuration was centralized: nothing pinned
    `train_gp`'s `lr` default, because the stage-3 smoke test constructs its own optimizer
    with an explicit learning rate. Changing ADAM_LR was caught by no test."""
    import inspect

    from src import config
    from src.gp import train_gp

    defaults = inspect.signature(train_gp).parameters
    assert defaults["lr"].default == 0.1
    assert defaults["num_steps"].default == 50
    assert defaults["max_cholesky_size"].default == 2000
    assert config.ADAM_LR == 0.1
    assert config.N_TRAINING_STEPS == 50
    assert config.MAX_CHOLESKY_SIZE == 2000


# =====================================================================================
# 8. Minimal end-to-end evaluation path
# =====================================================================================
def test_minimal_evaluation_pipeline_is_internally_consistent():
    """Protects: §3 reporting -- benchmark -> optimize -> best_so_far -> mean_standard_error
    -> summarize must agree with each other. This is the path that would produce Fig. 8, so
    a mismatch between the trace endpoint and the reported best value would corrupt every
    published number while every unit test stayed green."""
    results = run_replications(
        "hartmann6", "turbo-1", n_replications=3, base_seed=0,
        n_init=8, max_evals=24, batch_size=4, n_training_steps=3,
    )
    assert len(results) == 3
    assert [r.seed for r in results] == [0, 1, 2]

    for r in results:
        assert r.trace.shape == (r.n_evals,)
        assert np.all(np.diff(r.trace) <= 0), "trace must be a running minimum"
        assert r.trace[-1] == pytest.approx(r.best_value), (
            "the trace endpoint must equal the reported best value"
        )
        assert r.n_evals <= r.settings["max_evals"]

    traces = np.vstack([r.trace for r in results])
    mean, sem = mean_standard_error(traces)
    assert mean.shape == sem.shape == (traces.shape[1],)
    assert np.all(np.diff(mean) <= 1e-12), "the mean of running minima is non-increasing"

    s = summarize(results)
    finals = np.array([r.best_value for r in results])
    assert s["mean_best"] == pytest.approx(finals.mean())
    assert s["mean_best"] == pytest.approx(mean[-1])
    assert s["standard_error"] == pytest.approx(sem[-1])
    assert s["min_best"] <= s["median_best"] <= s["max_best"]


def test_replications_differ_but_are_individually_reproducible():
    """Protects: §3 -- error bars are only meaningful if replications are independent, and
    rule 11 checkability requires each to be re-runnable from its recorded seed."""
    kw = dict(benchmark="hartmann6", algorithm="turbo-1", n_init=8, max_evals=20,
              batch_size=4, n_training_steps=3)
    a = run_replications(n_replications=2, base_seed=0, **kw)
    b = run_replications(n_replications=2, base_seed=0, **kw)
    assert a[0].best_value != a[1].best_value, "replications must differ"
    assert a[0].best_value == pytest.approx(b[0].best_value), "same seed must reproduce"


def test_end_to_end_beats_random_search_on_a_smooth_problem():
    """Protects: the whole loop actually optimizes. Every structural test can pass while
    the algorithm proposes points that ignore the surrogate; this catches that."""
    b = quadratic(dim=4)
    opt = Turbo1(f=b, lb=b.lb, ub=b.ub, n_init=15, max_evals=90, batch_size=5,
                 n_training_steps=20, seed=0).optimize()
    rs = min(b(x) for x in from_unit_cube(as_generator(0).random((90, 4)), b.lb, b.ub))
    assert opt.best_value < rs


def test_unit_cube_round_trip_holds_across_the_optimizer_boundary():
    """Protects: App. C -- the optimizer works in [0,1]^d internally but reports points in
    the original box. A missing from_unit_cube would put every reported point in [0,1]."""
    b = quadratic()
    opt = Turbo1(f=b, lb=b.lb, ub=b.ub, n_init=6, max_evals=14, batch_size=4,
                 n_training_steps=3, seed=0).optimize()
    unit = to_unit_cube(opt.X, opt.lb, opt.ub)
    assert unit.min() >= -1e-9 and unit.max() <= 1 + 1e-9
    np.testing.assert_allclose(from_unit_cube(unit, opt.lb, opt.ub), opt.X, atol=1e-9)
    assert opt.X.min() < 0.0, "reported points must span the original box, not [0,1]"
