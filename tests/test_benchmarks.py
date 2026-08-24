"""Benchmark objectives and their domains.  (App. A; Sect. 3.5; App. F)"""

import numpy as np
import pytest

from src.benchmarks import (
    SYNTHETIC_SUITE,
    Benchmark,
    ackley,
    cosmological_constants,
    get_benchmark,
    hartmann6,
    levy,
    lunar_lander,
    negate,
    rastrigin,
    robot_pushing,
    rover,
)


# --- known global minima ------------------------------------------------------------
def test_ackley_minimum_at_origin():
    assert ackley(np.zeros(10)) == pytest.approx(0.0, abs=1e-12)
    assert ackley(np.ones(10)) > 0.0


def test_levy_minimum_at_ones():
    assert levy(np.ones(10)) == pytest.approx(0.0, abs=1e-12)
    assert levy(np.zeros(10)) > 0.0


def test_rastrigin_minimum_at_origin():
    assert rastrigin(np.zeros(10)) == pytest.approx(0.0, abs=1e-12)
    assert rastrigin(0.5 * np.ones(10)) > 0.0


def test_hartmann6_known_optimum():
    """Standard Hartmann-6 optimum, approx -3.32237."""
    x_opt = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    assert hartmann6(x_opt) == pytest.approx(-3.32237, abs=1e-4)


def test_hartmann6_rejects_wrong_dimension():
    with pytest.raises(AssertionError):
        hartmann6(np.zeros(5))


# --- domains from the paper ---------------------------------------------------------
@pytest.mark.parametrize(
    "name,dim,lo,hi",
    [
        ("ackley10", 10, -5.0, 10.0),      # App. A
        ("levy10", 10, -5.0, 10.0),        # App. A
        ("rastrigin10", 10, -3.0, 4.0),    # App. A
        ("hartmann6", 6, 0.0, 1.0),        # App. A
        ("ackley200", 200, -5.0, 10.0),    # Sect. 3.5
    ],
)
def test_domains_match_the_paper(name, dim, lo, hi):
    b = get_benchmark(name)
    assert b.dim == dim
    assert np.all(b.lb == lo) and np.all(b.ub == hi)
    assert b.lb.shape == (dim,) and b.ub.shape == (dim,)


def test_benchmark_call_returns_a_scalar_float():
    b = get_benchmark("ackley10")
    v = b(np.zeros(10))
    assert isinstance(v, float) and np.isfinite(v)


def test_all_suite_entries_are_finite_on_a_random_point():
    rng = np.random.default_rng(0)
    for name, b in SYNTHETIC_SUITE.items():
        x = b.lb + (b.ub - b.lb) * rng.random(b.dim)
        assert np.isfinite(b(x)), f"{name} produced a non-finite value"


def test_global_min_is_attainable_within_the_domain():
    """A sanity check that our stated optimum is actually inside the stated box."""
    for name in ("ackley10", "levy10", "rastrigin10"):
        b = get_benchmark(name)
        target = {"ackley10": 0.0, "levy10": 1.0, "rastrigin10": 0.0}[name]
        x = np.full(b.dim, target)
        assert np.all(x >= b.lb) and np.all(x <= b.ub)
        assert b(x) == pytest.approx(b.global_min, abs=1e-9)


def test_unknown_benchmark_raises():
    with pytest.raises(KeyError):
        get_benchmark("branin")


# --- sign convention (§2 minimizes; §3.1-3.4 plot reward) ----------------------------
def test_negate_converts_a_reward_into_a_minimization_objective():
    """§2 minimizes, but the Sect. 3 problems report reward (higher better)."""
    reward = lambda x: float(np.sum(x))  # noqa: E731
    cost = negate(reward)
    x = np.array([1.0, 2.0])
    assert cost(x) == pytest.approx(-3.0)
    assert cost(x) == pytest.approx(-reward(x))


def test_negate_round_trips():
    reward = lambda x: float(np.sum(x**2))  # noqa: E731
    x = np.array([2.0, -1.0])
    assert negate(negate(reward))(x) == pytest.approx(reward(x))


def test_negated_benchmark_minimum_is_the_reward_maximum():
    """A reward maximized at the origin must become a cost minimized at the origin."""
    reward = lambda x: -float(np.sum(x**2))  # noqa: E731  maximized (=0) at the origin
    b = Benchmark("neg", negate(reward), np.full(2, -1.0), np.full(2, 1.0), 2, 0.0)
    assert b(np.zeros(2)) == pytest.approx(0.0)
    assert b(np.array([0.5, 0.5])) > b(np.zeros(2)), "cost must increase away from the optimum"


# --- benchmarks the paper does not fully specify -------------------------------------
@pytest.mark.parametrize(
    "fn,needle",
    [
        (robot_pushing, "Wang et al. 2018"),
        (rover, "Wang et al. 2018"),
        (lunar_lander, "50 seeds are not published"),
        (cosmological_constants, "NOT reproducible"),
    ],
)
def test_unreproducible_benchmarks_fail_loudly(fn, needle):
    """Rule 11: these must never silently substitute a stand-in objective."""
    with pytest.raises(NotImplementedError) as exc:
        fn(np.zeros(12))
    assert needle in str(exc.value)
