"""End-to-end TuRBO-1 / TuRBO-m behavior.  (PAPER_SPEC.md §5)

These run the real algorithm on small budgets. They assert structural invariants and
optimization progress, NOT paper numbers -- reproducing a paper figure requires the full
protocol in evaluate.SYNTHETIC_PROTOCOL (rule 11).
"""

import numpy as np
import pytest

from src.benchmarks import Benchmark, get_benchmark
from src.evaluate import best_so_far, mean_standard_error, run_replications, summarize
from src.turbo_1 import Turbo1
from src.turbo_m import TurboM


def sphere(dim=4) -> Benchmark:
    """A cheap smooth objective so end-to-end tests stay fast."""
    return Benchmark(
        name="sphere",
        f=lambda x: float(np.sum(x**2)),
        lb=np.full(dim, -5.0),
        ub=np.full(dim, 5.0),
        dim=dim,
        global_min=0.0,
    )


# --- TuRBO-1 ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def turbo1_run():
    b = sphere()
    opt = Turbo1(f=b, lb=b.lb, ub=b.ub, n_init=10, max_evals=40, batch_size=5,
                 n_training_steps=10, seed=0)
    return opt.optimize()


def test_history_shapes(turbo1_run):
    opt = turbo1_run
    assert opt.X.shape == (opt.n_evals, opt.dim)
    assert opt.fX.shape == (opt.n_evals, 1)
    assert opt.X.dtype == np.float64


def test_budget_is_never_exceeded(turbo1_run):
    """PAPER_SPEC.md §10 C3: max_evals is inclusive of the initial design."""
    assert turbo1_run.n_evals <= turbo1_run.max_evals


def test_all_points_lie_inside_the_domain(turbo1_run):
    opt = turbo1_run
    assert np.all(opt.X >= opt.lb - 1e-9) and np.all(opt.X <= opt.ub + 1e-9)


def test_best_point_matches_best_value(turbo1_run):
    opt = turbo1_run
    assert opt.f(opt.best_point) == pytest.approx(opt.best_value)


def test_optimization_improves_on_the_initial_design(turbo1_run):
    opt = turbo1_run
    initial_best = opt.fX[: opt.n_init].min()
    assert opt.best_value <= initial_best


def test_candidate_set_size_follows_appendix_d(turbo1_run):
    assert turbo1_run.n_cand == min(100 * turbo1_run.dim, 5000)


def test_runs_are_reproducible_from_a_seed():
    b = sphere()
    kw = dict(f=b, lb=b.lb, ub=b.ub, n_init=8, max_evals=24, batch_size=4,
              n_training_steps=5, seed=123)
    a = Turbo1(**kw).optimize()
    c = Turbo1(**kw).optimize()
    np.testing.assert_allclose(a.X, c.X)
    assert a.best_value == pytest.approx(c.best_value)


def test_different_seeds_give_different_runs():
    b = sphere()
    kw = dict(f=b, lb=b.lb, ub=b.ub, n_init=8, max_evals=24, batch_size=4, n_training_steps=5)
    a = Turbo1(**kw, seed=1).optimize()
    c = Turbo1(**kw, seed=2).optimize()
    assert not np.allclose(a.X, c.X)


def test_batch_size_one_is_supported():
    """Sect. 3.8 sweeps q in {1, 2, 4, ..., 64}; q=1 is the sequential case."""
    b = sphere(3)
    opt = Turbo1(f=b, lb=b.lb, ub=b.ub, n_init=6, max_evals=12, batch_size=1,
                 n_training_steps=5, seed=0).optimize()
    assert opt.n_evals <= 12


def test_noisy_centering_path_runs():
    """Sect. 2's noisy rule -- not implemented in the official code (PAPER_SPEC.md §10 A3)."""
    b = sphere(3)
    opt = Turbo1(f=b, lb=b.lb, ub=b.ub, n_init=8, max_evals=16, batch_size=4,
                 n_training_steps=5, noisy=True, seed=0).optimize()
    assert opt.n_evals <= 16 and np.isfinite(opt.best_value)


def test_rejects_budget_smaller_than_the_initial_design():
    b = sphere()
    with pytest.raises(AssertionError):
        Turbo1(f=b, lb=b.lb, ub=b.ub, n_init=50, max_evals=20, batch_size=5)


def test_trust_region_shrinks_on_a_flat_objective():
    """A constant objective can never produce a success, so L must halve repeatedly.

    With d=3, q=5 -> failtol = ceil(3/5) = 1, so every batch halves L. Collapsing
    L_init=0.8 below L_min=2^-7 takes 7 halvings, i.e. 5 init + 7*5 = 40 evaluations for
    the first cycle; the budget below leaves room for a second start.
    """
    flat = Benchmark("flat", lambda x: 1.0, np.zeros(3), np.ones(3), 3, 1.0)
    opt = Turbo1(f=flat, lb=flat.lb, ub=flat.ub, n_init=5, max_evals=70, batch_size=5,
                 n_training_steps=5, seed=0).optimize()
    assert opt.n_restarts >= 2, "a flat objective must collapse the TR and restart"


# --- TuRBO-m ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def turbom_run():
    b = sphere()
    opt = TurboM(f=b, lb=b.lb, ub=b.ub, n_init=6, max_evals=48, n_trust_regions=3,
                 batch_size=6, n_training_steps=10, seed=0)
    return opt.optimize()


def test_turbo_m_initializes_every_trust_region(turbom_run):
    opt = turbom_run
    owners = opt._idx.ravel()[: opt.n_trust_regions * opt.n_init]
    assert set(owners) == set(range(opt.n_trust_regions))


def test_turbo_m_history_and_ownership_stay_aligned(turbom_run):
    opt = turbom_run
    assert opt.X.shape[0] == opt.fX.shape[0] == opt._idx.shape[0] == opt.n_evals


def test_turbo_m_ownership_indices_are_valid(turbom_run):
    """-1 marks points orphaned by a restart (PAPER_SPEC.md §10 B9)."""
    opt = turbom_run
    assert set(np.unique(opt._idx)) <= set(range(-1, opt.n_trust_regions))


def test_turbo_m_respects_the_budget(turbom_run):
    assert turbom_run.n_evals <= turbom_run.max_evals


def test_turbo_m_failtol_uses_the_sequential_case(turbom_run):
    """App. D: "the same tolerances as in the sequential case (q = 1)"."""
    assert turbom_run.failtol == turbom_run.dim


def test_turbo_m_every_trust_region_state_carries_the_sequential_failtol(turbom_run):
    """All per-TR states must agree with the TuRBO-m tolerance (App. D)."""
    assert all(s.failtol == turbom_run.dim for s in turbom_run.states)


def test_turbo_m_inherited_state_is_not_stale(turbom_run):
    """The single-TR state inherited from Turbo1 must not keep the TuRBO-1 tolerance.

    Turbo1.__init__ builds `self.state` before TurboM overwrites `self.failtol`, so without
    a refresh the inherited object silently carries ceil(d/q) instead of the sequential-case
    value d. It is unread by the TuRBO-m path today, but evaluate.py reads `state.succtol`,
    so a stale object here is a trap.
    """
    assert turbom_run.state.failtol == turbom_run.failtol == turbom_run.dim


def test_turbo_m_improves_on_its_initial_designs(turbom_run):
    opt = turbom_run
    initial = opt.fX[: opt.n_trust_regions * opt.n_init].min()
    assert opt.best_value <= initial


def test_turbo_m_requires_more_than_one_region():
    b = sphere()
    with pytest.raises(AssertionError):
        TurboM(f=b, lb=b.lb, ub=b.ub, n_init=5, max_evals=40, n_trust_regions=1, batch_size=5)


def test_turbo_m_requires_budget_for_all_initial_designs():
    b = sphere()
    with pytest.raises(AssertionError):
        TurboM(f=b, lb=b.lb, ub=b.ub, n_init=20, max_evals=50, n_trust_regions=5, batch_size=5)


# --- evaluate -----------------------------------------------------------------------
def test_best_so_far_is_monotone_non_increasing():
    trace = best_so_far(np.array([5.0, 7.0, 3.0, 4.0, 1.0]))
    np.testing.assert_allclose(trace, [5.0, 5.0, 3.0, 3.0, 1.0])
    assert np.all(np.diff(trace) <= 0)


def test_mean_standard_error_matches_the_definition():
    traces = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean, sem = mean_standard_error(traces)
    np.testing.assert_allclose(mean, [3.0, 4.0])
    np.testing.assert_allclose(sem, traces.std(axis=0, ddof=1) / np.sqrt(3))


def test_mean_standard_error_single_replication_has_zero_sem():
    mean, sem = mean_standard_error(np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(mean, [1.0, 2.0])
    np.testing.assert_allclose(sem, [0.0, 0.0])


def test_run_replications_records_the_settings_used():
    """Rule 11: a claim is only checkable if the settings are recorded with the result."""
    results = run_replications(
        "hartmann6", "turbo-1", n_replications=2, base_seed=0,
        n_init=6, max_evals=14, batch_size=4, n_training_steps=5,
    )
    assert len(results) == 2
    assert [r.seed for r in results] == [0, 1]
    for r in results:
        assert r.trace.shape == (r.n_evals,)
        assert r.settings["n_init"] == 6 and r.settings["batch_size"] == 4
    s = summarize(results)
    assert s["n_replications"] == 2 and np.isfinite(s["mean_best"])


def test_run_replications_rejects_an_unknown_algorithm():
    with pytest.raises(ValueError, match="unknown algorithm"):
        run_replications("hartmann6", "cma-es", n_replications=1)


def test_turbo_makes_real_progress_on_hartmann6():
    """Sanity check on a real paper benchmark (App. A domain), not a paper-number claim."""
    b = get_benchmark("hartmann6")
    opt = Turbo1(f=b, lb=b.lb, ub=b.ub, n_init=20, max_evals=120, batch_size=10,
                 n_training_steps=30, seed=0).optimize()
    random_baseline = np.random.default_rng(0).random((120, 6))
    random_best = min(b(x) for x in random_baseline)
    assert opt.best_value < random_best, "TuRBO should beat random search on Hartmann6"
