"""TuRBO-1: a single trust region with restarts.

Paper anchor (Sect. 2). The outer restart loop comes from "Trust regions":
  "Whenever L falls below a given minimum threshold L_min, we discard the respective TR
   and initialize a new one with side length L_init."

Execution order matters and is load-bearing (PAPER_SPEC.md §5):
  1. refit the GP  -> 2. pick the center -> 3. derive TR bounds from the *current*
  fitted ARD lengthscales -> 4. candidates -> 5. Thompson draws -> 6. select -> 7.
  evaluate -> 8. update counters (against the pre-batch incumbent) -> 9. append.
Reordering 1/3 silently reuses stale geometry; reordering 8/9 makes every batch read as a
non-improvement so the TR only ever shrinks.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import config
from .candidates import n_candidates
from .local_model import SurrogateSettings, propose_from_trust_region
from .thompson import select_candidates
from .trust_region import TR_DEFAULTS, TrustRegionState, default_failtol, is_success
from .utils import as_generator, from_unit_cube, latin_hypercube, to_unit_cube

__all__ = ["Turbo1"]


class Turbo1:
    """The TuRBO-1 algorithm (Sect. 2).

    Minimizes `f` over the box [lb, ub]. Sect. 2: "Find x* in Omega such that
    f(x*) <= f(x), for all x in Omega". Reward-style benchmarks must be negated by the
    caller; see benchmarks.py.

    Args:
        f: objective to MINIMIZE, mapping a (d,) float64 point to a float.
        lb, ub: (d,) float64 box bounds.
        n_init: initial design size. Per-experiment values are in PAPER_SPEC.md §9.
        max_evals: total evaluation budget, inclusive of initial points.
        batch_size: q.
        noisy: use the paper's noisy centering rule (Sect. 2). See trust_region.select_center.
        success_tol: see trust_region.is_success. 0.0 = the paper's literal rule.
        center_stat: "median" (official code) or "mean". See utils.standardize.
        use_predictive: see thompson.thompson_draws.
        seed: int or np.random.Generator.

    # [FROM_OFFICIAL_CODE] (PAPER_SPEC.md §10 C3) The paper never says whether `max_evals`
    # includes the initial design. The official code counts it (turbo_1.py L253).
    # Using: inclusive. Alternatives: exclusive, which would hand TuRBO extra evaluations
    # relative to the baselines it is compared against.
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        n_init: int,
        max_evals: int,
        batch_size: int = 1,
        noisy: bool = False,
        success_tol: float = config.SUCCESS_TOL,
        center_stat: str = config.CENTER_STAT,
        use_predictive: bool = config.USE_PREDICTIVE_DRAWS,
        use_ard: bool = True,
        n_training_steps: int = config.N_TRAINING_STEPS,
        max_cholesky_size: int = config.MAX_CHOLESKY_SIZE,
        seed: int | np.random.Generator | None = None,
        verbose: bool = False,
    ):
        lb = np.asarray(lb, dtype=np.float64).ravel()
        ub = np.asarray(ub, dtype=np.float64).ravel()
        assert lb.shape == ub.shape and np.all(ub > lb)
        assert max_evals > n_init and max_evals > batch_size
        assert n_init > 0 and batch_size > 0

        self.f = f
        self.lb, self.ub = lb, ub
        self.dim = len(lb)
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.success_tol = success_tol
        self.verbose = verbose
        # One object rather than six loose attributes; the individual names remain
        # readable as properties below, since evaluate.py records them per run.
        self.surrogate = SurrogateSettings(
            use_ard=use_ard,
            n_training_steps=n_training_steps,
            max_cholesky_size=max_cholesky_size,
            center_stat=center_stat,
            use_predictive=use_predictive,
            noisy=noisy,
        )
        self.rng = as_generator(seed)

        self.n_cand = n_candidates(self.dim)  # App. D
        self.failtol = default_failtol(self.dim, batch_size, n_trust_regions=1)

        # Global history across restarts.
        self.X = np.zeros((0, self.dim), dtype=np.float64)
        self.fX = np.zeros((0, 1), dtype=np.float64)
        self.n_evals = 0
        self.n_restarts = 0

        self.state = self._new_state()
        self._X: np.ndarray = np.zeros((0, self.dim), dtype=np.float64)
        self._fX: np.ndarray = np.zeros((0, 1), dtype=np.float64)

    def _new_state(self) -> TrustRegionState:
        return TrustRegionState(
            length=TR_DEFAULTS["length_init"],
            succtol=TR_DEFAULTS["succtol"],
            failtol=self.failtol,
            length_min=TR_DEFAULTS["length_min"],
            length_max=TR_DEFAULTS["length_max"],
            length_init=TR_DEFAULTS["length_init"],
        )

    # --- objective ---------------------------------------------------------------
    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate f on (n, d) points in the ORIGINAL box. Returns (n, 1)."""
        fX = np.array([[float(self.f(x))] for x in X], dtype=np.float64)
        self.n_evals += len(X)
        return fX

    def _append(self, X: np.ndarray, fX: np.ndarray) -> None:
        self._X = np.vstack((self._X, X))
        self._fX = np.vstack((self._fX, fX))
        self.X = np.vstack((self.X, X))
        self.fX = np.vstack((self.fX, fX))

    # --- one batch ---------------------------------------------------------------
    def _propose(self) -> np.ndarray:
        """One Thompson-sampling batch from the current trust region.

        Returns (q, d) float64 in the unit cube. The surrogate step is shared with
        TuRBO-m (`local_model.propose_from_trust_region`); what is specific to TuRBO-1 is
        that selection ranges over a single region.
        """
        proposal = propose_from_trust_region(
            to_unit_cube(self._X, self.lb, self.ub),
            self._fX,
            length=self.state.length,
            batch_size=self.batch_size,
            n_cand=self.n_cand,
            rng=self.rng,
            settings=self.surrogate,
        )
        return select_candidates(proposal.X_cand, proposal.y_cand)

    # --- main loop ---------------------------------------------------------------
    def optimize(self) -> "Turbo1":
        """Run until the evaluation budget is exhausted, restarting on TR collapse."""
        while self.n_evals < self.max_evals:
            # Restart: fresh TR, fresh initial design, local history discarded.
            # Sect. 2: "we discard the respective TR and initialize a new one".
            # [PARTIALLY_SPECIFIED] (PAPER_SPEC.md §10 B9) "Discard" is taken to include
            # the TR's data, matching turbo_1.py. The points remain in self.X/self.fX for
            # reporting. Alternatives: carry the data into the new TR's GP -- which would
            # contradict "discard".
            self.state.reset()
            self._X = np.zeros((0, self.dim), dtype=np.float64)
            self._fX = np.zeros((0, 1), dtype=np.float64)

            n_init = min(self.n_init, self.max_evals - self.n_evals)
            X_init = from_unit_cube(
                latin_hypercube(n_init, self.dim, self.rng), self.lb, self.ub
            )
            self._append(X_init, self._evaluate(X_init))
            if self.verbose:
                print(f"[restart {self.n_restarts}] fbest = {self._fX.min():.4g}")
            self.n_restarts += 1

            while self.n_evals < self.max_evals and not self.state.is_converged:
                f_incumbent = float(self._fX.min())  # BEFORE the batch is appended

                X_next = from_unit_cube(self._propose(), self.lb, self.ub)
                n_take = min(self.batch_size, self.max_evals - self.n_evals)
                X_next = X_next[:n_take]
                fX_next = self._evaluate(X_next)

                # App. D: "we consider an improvement from at least one evaluation in the
                # batch a success". TuRBO-1 increments the failure counter by 1.
                self.state.update(
                    success=is_success(fX_next, f_incumbent, self.success_tol),
                    n_failures=1,
                )
                self._append(X_next, fX_next)

                if self.verbose and fX_next.min() < f_incumbent:
                    print(f"  {self.n_evals}) new best {fX_next.min():.4g} (L={self.state.length:.4g})")
        return self

    # --- surrogate settings (read-only views; evaluate.py records these) ----------
    @property
    def use_ard(self) -> bool:
        return self.surrogate.use_ard

    @property
    def n_training_steps(self) -> int:
        return self.surrogate.n_training_steps

    @property
    def max_cholesky_size(self) -> int:
        return self.surrogate.max_cholesky_size

    @property
    def center_stat(self) -> str:
        return self.surrogate.center_stat

    @property
    def use_predictive(self) -> bool:
        return self.surrogate.use_predictive

    @property
    def noisy(self) -> bool:
        return self.surrogate.noisy

    # --- results -----------------------------------------------------------------
    @property
    def best_value(self) -> float:
        return float(self.fX.min())

    @property
    def best_point(self) -> np.ndarray:
        return self.X[int(np.argmin(self.fX.ravel())), :]
