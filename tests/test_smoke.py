"""Smallest end-to-end execution path, staged.

Each stage is the TuRBO analogue of a standard training-pipeline stage:

    fixture         -> a tiny 2D synthetic dataset (8 points)
    forward pass    -> GP posterior over a candidate set
    loss            -> exact marginal log likelihood (the ONLY differentiated quantity;
                       PAPER_SPEC.md §6 -- the black-box objective f is never differentiated)
    optimizer step  -> one Adam step on the GP hyperparameters
    one iteration   -> one full TuRBO batch: refit -> center -> bounds -> candidates ->
                       Thompson draws -> select -> evaluate -> trust-region update
    end-to-end      -> a complete Turbo1.optimize() and TurboM.optimize() on the fixture

The stages are deliberately ordered so that a break shows up at the earliest failing stage
rather than as an opaque failure of the whole run.

Everything here is float64 on CPU (REPRODUCTION_NOTES C14) and uses a fixed seed.
"""

import numpy as np
import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.benchmarks import Benchmark
from src.candidates import create_candidates
from src.gp import train_gp
from src.thompson import select_candidates, select_candidates_across, thompson_draws
from src.trust_region import TR_DEFAULTS, TrustRegionState, is_success, select_center, trust_region_bounds
from src.turbo_1 import Turbo1
from src.turbo_m import TurboM
from src.utils import as_generator, from_unit_cube, latin_hypercube, standardize, to_unit_cube

DIM = 2
N_TINY = 8
SEED = 0


# --- stage 0: tiny fixture ----------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_problem():
    """A 2D quadratic on [-2, 2]^2 -- cheap, smooth, minimum 0 at the origin."""
    return Benchmark(
        name="tiny_quadratic",
        f=lambda x: float(np.sum(x**2)),
        lb=np.full(DIM, -2.0),
        ub=np.full(DIM, 2.0),
        dim=DIM,
        global_min=0.0,
    )


@pytest.fixture(scope="module")
def tiny_dataset(tiny_problem):
    """8 LHD points, evaluated, rescaled to the unit cube and standardized (App. C)."""
    rng = as_generator(SEED)
    X = from_unit_cube(latin_hypercube(N_TINY, DIM, rng), tiny_problem.lb, tiny_problem.ub)
    fX = np.array([[tiny_problem(x)] for x in X])
    X_unit = to_unit_cube(X, tiny_problem.lb, tiny_problem.ub)
    fX_std, mu, sigma = standardize(fX)
    return {
        "X": X, "fX": fX, "X_unit": X_unit,
        "fX_std": fX_std, "mu": mu, "sigma": sigma,
        "X_torch": torch.as_tensor(X_unit, dtype=torch.float64),
        "y_torch": torch.as_tensor(fX_std, dtype=torch.float64),
    }


def test_stage0_fixture_shapes_dtypes_and_ranges(tiny_dataset):
    d = tiny_dataset
    assert d["X"].shape == (N_TINY, DIM) and d["X"].dtype == np.float64
    assert d["fX"].shape == (N_TINY, 1)
    assert d["X_unit"].min() >= 0.0 and d["X_unit"].max() <= 1.0
    assert d["fX_std"].shape == (N_TINY,), "GP targets must be (n,), not (n, 1)"
    assert d["X_torch"].dtype == torch.float64 and d["y_torch"].dtype == torch.float64
    assert d["X_torch"].device.type == "cpu"
    assert np.all(np.isfinite(d["fX_std"]))


# --- stage 1: forward pass ----------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_gp(tiny_dataset):
    return train_gp(tiny_dataset["X_torch"], tiny_dataset["y_torch"], num_steps=5)


def test_stage1_forward_pass_on_training_inputs(tiny_gp, tiny_dataset):
    """The GP posterior evaluates and returns finite, correctly shaped moments."""
    with torch.no_grad():
        post = tiny_gp(tiny_dataset["X_torch"])
    assert post.mean.shape == (N_TINY,)
    assert post.variance.shape == (N_TINY,)
    assert post.mean.dtype == torch.float64
    assert torch.all(torch.isfinite(post.mean)) and torch.all(post.variance > 0)


def test_stage1_forward_pass_on_unseen_candidates(tiny_gp):
    """The forward pass that actually matters: the posterior over a candidate set."""
    X_cand = torch.as_tensor(
        np.random.default_rng(SEED).random((16, DIM)), dtype=torch.float64
    )
    with torch.no_grad():
        post = tiny_gp(X_cand)
    assert post.mean.shape == (16,)
    assert torch.all(torch.isfinite(post.mean))


# --- stage 2: the loss --------------------------------------------------------------
def test_stage2_marginal_log_likelihood_computes(tiny_dataset):
    """PAPER_SPEC.md E8 / §6: the negative log marginal likelihood is the only loss."""
    gp = train_gp(tiny_dataset["X_torch"], tiny_dataset["y_torch"], num_steps=0)
    gp.train()
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    loss = -mll(gp(tiny_dataset["X_torch"]), tiny_dataset["y_torch"])
    assert loss.shape == torch.Size([]), "loss must be a scalar"
    assert torch.isfinite(loss)
    assert loss.requires_grad, "loss must be differentiable w.r.t. the hyperparameters"


def test_stage2_loss_backward_populates_gradients(tiny_dataset):
    gp = train_gp(tiny_dataset["X_torch"], tiny_dataset["y_torch"], num_steps=0)
    gp.train()
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    loss = -mll(gp(tiny_dataset["X_torch"]), tiny_dataset["y_torch"])
    loss.backward()
    grads = [p.grad for p in gp.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.all(torch.isfinite(g)) for g in grads)


# --- stage 3: one optimizer step ----------------------------------------------------
def test_stage3_single_adam_step_changes_hyperparameters(tiny_dataset):
    """One Adam step on the marginal likelihood (PAPER_SPEC.md §7; App. C)."""
    gp = train_gp(tiny_dataset["X_torch"], tiny_dataset["y_torch"], num_steps=0)
    gp.train()
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    before = gp.covar_module.base_kernel.lengthscale.detach().clone()

    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)  # App. C rate, FROM_OFFICIAL_CODE
    optimizer.zero_grad()
    loss0 = -mll(gp(tiny_dataset["X_torch"]), tiny_dataset["y_torch"])
    loss0.backward()
    optimizer.step()

    after = gp.covar_module.base_kernel.lengthscale.detach().clone()
    assert not torch.allclose(before, after), "one Adam step must move the hyperparameters"

    with torch.no_grad():
        loss1 = -mll(gp(tiny_dataset["X_torch"]), tiny_dataset["y_torch"])
    assert torch.isfinite(loss1)


def test_stage3_fifty_steps_reduce_the_loss(tiny_dataset):
    """The configured schedule: 50 Adam steps at lr 0.1 must improve the objective."""
    X, y = tiny_dataset["X_torch"], tiny_dataset["y_torch"]

    def nll(steps):
        gp = train_gp(X, y, num_steps=steps)
        gp.train()
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with torch.no_grad():
            return float(-mll(gp(X), y))

    assert nll(50) < nll(0)


# --- stage 4: one full TuRBO iteration ----------------------------------------------
def test_stage4_one_turbo_iteration_executes(tiny_problem, tiny_dataset, tiny_gp):
    """Refit -> center -> bounds -> candidates -> draws -> select -> evaluate -> update.

    This is the smallest unit that exercises every module together, in the order
    PAPER_SPEC.md §5 requires.
    """
    rng = as_generator(SEED)
    d = tiny_dataset
    state = TrustRegionState(length=TR_DEFAULTS["length_init"], succtol=3, failtol=2)
    q = 3

    # center (E3) -- best observation, noise-free case
    x_center = select_center(d["X_unit"], d["fX_std"])
    assert x_center.shape == (1, DIM)

    # bounds from the CURRENT fitted lengthscales (E2) -- must follow the refit
    lengthscales = tiny_gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    lb_tr, ub_tr = trust_region_bounds(x_center, lengthscales, state.length)
    assert lb_tr.shape == ub_tr.shape == (1, DIM)
    assert np.all(lb_tr >= 0.0) and np.all(ub_tr <= 1.0)

    # candidates (E6)
    X_cand = create_candidates(x_center, lb_tr, ub_tr, rng, n_cand=32)
    assert X_cand.shape == (32, DIM)
    assert np.all(X_cand >= lb_tr - 1e-12) and np.all(X_cand <= ub_tr + 1e-12)

    # Thompson draws (E5), de-standardized (E7)
    y_cand = thompson_draws(tiny_gp, X_cand, q, mu=d["mu"], sigma=d["sigma"], seed=SEED)
    assert y_cand.shape == (32, q) and np.all(np.isfinite(y_cand))

    # selection (E5)
    X_next_unit = select_candidates(X_cand, y_cand.copy())
    assert X_next_unit.shape == (q, DIM)
    assert len(np.unique(X_next_unit, axis=0)) == q, "selected points must be distinct"

    # evaluate the true objective
    X_next = from_unit_cube(X_next_unit, tiny_problem.lb, tiny_problem.ub)
    fX_next = np.array([[tiny_problem(x)] for x in X_next])
    assert fX_next.shape == (q, 1) and np.all(np.isfinite(fX_next))

    # trust-region update (E4), against the pre-batch incumbent
    incumbent = float(d["fX"].min())
    length_before = state.length
    state.update(success=is_success(fX_next, incumbent), n_failures=1)
    assert state.length in (length_before, length_before / 2.0, min(2 * length_before, state.length_max))
    assert state.succcount >= 0 and state.failcount >= 0


def test_stage4_cross_tr_selection_executes(tiny_gp, tiny_dataset):
    """The TuRBO-m bandit step over two (here identical) local models."""
    rng = as_generator(SEED)
    d, q, m, n_cand = tiny_dataset, 2, 2, 16
    x_center = select_center(d["X_unit"], d["fX_std"])
    ls = tiny_gp.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    lb_tr, ub_tr = trust_region_bounds(x_center, ls, 0.8)

    X_cand = np.stack([create_candidates(x_center, lb_tr, ub_tr, rng, n_cand=n_cand)
                       for _ in range(m)])
    y_cand = np.stack([thompson_draws(tiny_gp, X_cand[i], q, mu=d["mu"], sigma=d["sigma"],
                                      seed=SEED + i) for i in range(m)])
    assert X_cand.shape == (m, n_cand, DIM)
    assert y_cand.shape == (m, n_cand, q)

    X_next, idx_next = select_candidates_across(X_cand, y_cand.copy())
    assert X_next.shape == (q, DIM) and idx_next.shape == (q, 1)
    assert set(idx_next.ravel()) <= set(range(m))


# --- stage 5: end-to-end ------------------------------------------------------------
def test_stage5_turbo1_end_to_end(tiny_problem):
    """The smallest complete TuRBO-1 run."""
    opt = Turbo1(
        f=tiny_problem, lb=tiny_problem.lb, ub=tiny_problem.ub,
        n_init=N_TINY, max_evals=N_TINY + 6, batch_size=3,
        n_training_steps=3, seed=SEED,
    ).optimize()

    assert opt.n_evals <= opt.max_evals
    assert opt.X.shape == (opt.n_evals, DIM) and opt.fX.shape == (opt.n_evals, 1)
    assert np.all(np.isfinite(opt.fX))
    assert np.all(opt.X >= opt.lb - 1e-9) and np.all(opt.X <= opt.ub + 1e-9)
    assert opt.best_value <= opt.fX[:N_TINY].min(), "must not do worse than its initial design"
    assert tiny_problem(opt.best_point) == pytest.approx(opt.best_value)


def test_stage5_turbom_end_to_end(tiny_problem):
    """The smallest complete TuRBO-m run (m = 2)."""
    opt = TurboM(
        f=tiny_problem, lb=tiny_problem.lb, ub=tiny_problem.ub,
        n_init=4, max_evals=20, n_trust_regions=2, batch_size=2,
        n_training_steps=3, seed=SEED,
    ).optimize()

    assert opt.n_evals <= opt.max_evals
    assert opt.X.shape[0] == opt.fX.shape[0] == opt._idx.shape[0] == opt.n_evals
    assert set(np.unique(opt._idx)) <= set(range(-1, opt.n_trust_regions))
    assert np.all(np.isfinite(opt.fX))


def test_stage5_run_is_deterministic_under_a_fixed_seed(tiny_problem):
    """Guards the whole path: NumPy generator AND the forked torch RNG."""
    def run():
        return Turbo1(
            f=tiny_problem, lb=tiny_problem.lb, ub=tiny_problem.ub,
            n_init=N_TINY, max_evals=N_TINY + 6, batch_size=3,
            n_training_steps=3, seed=SEED,
        ).optimize()

    a, b = run(), run()
    np.testing.assert_allclose(a.X, b.X)
    np.testing.assert_allclose(a.fX, b.fX)


# --- regression guards for the defects this pass found ------------------------------
def test_batch_larger_than_candidate_set_is_rejected():
    """Exhausting the candidate set used to return the same point repeatedly."""
    X_cand = np.arange(10, dtype=np.float64).reshape(5, 2)
    y_cand = np.random.default_rng(SEED).random((5, 8))
    with pytest.raises(ValueError, match="exceeds the candidate set"):
        select_candidates(X_cand, y_cand)


def test_pooled_batch_larger_than_all_candidates_is_rejected():
    X_cand = np.random.default_rng(SEED).random((2, 3, 2))
    y_cand = np.random.default_rng(SEED).random((2, 3, 9))
    with pytest.raises(ValueError, match="exceeds the pooled candidate set"):
        select_candidates_across(X_cand, y_cand)


def test_mixed_dtype_training_inputs_are_rejected():
    """Mixed dtypes silently downcast the GP fit, contradicting the float64 policy."""
    X = torch.rand(8, DIM, dtype=torch.float32)
    y = torch.rand(8, dtype=torch.float64)
    with pytest.raises(AssertionError, match="silently downcast"):
        train_gp(X, y, num_steps=1)
