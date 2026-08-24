"""TuRBO-m: m trust regions with an implicit multi-armed bandit across them.

Paper anchor (Sect. 2, "Trust region Bayesian optimization", verbatim):
  "TuRBO maintains m trust regions simultaneously. Each trust region TR_l with
   l in {1, ..., m} is a hyperrectangle of base side length L_l <= L_max, and utilizes an
   independent local GP model. This gives rise to a classical exploitation-exploration
   trade-off that we model by a multi-armed bandit that treats each TR as a lever."

App. D (verbatim) for the per-TR counters:
  "We use separate success and failure counters for each TR. We consider a batch a success
   for TR_l if q_l > 0 points are selected from this TR and at least one is better than the
   best solution in this TR. ... If all q_l > 0 evaluations are worse than the current best
   solution we consider this a failure and set the success counter to zero and add q_l to
   the failure counter."

Structurally this is TuRBO-1 plus three overrides: per-TR counters, joint candidate
selection across TRs, and a per-TR restart path. Everything else -- GP fitting, TR
geometry, candidate generation -- is the shared code in the sibling modules.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .local_model import propose_from_trust_region
from .thompson import select_candidates_across
from .trust_region import TrustRegionState, default_failtol, is_success
from .turbo_1 import Turbo1
from .utils import from_unit_cube, latin_hypercube, to_unit_cube

__all__ = ["TurboM"]


class TurboM(Turbo1):
    """The TuRBO-m algorithm (Sect. 2).

    Args:
        n_init: initial design size **per trust region**. Sect. 3.1: "TuRBO-20 where we use
            50 initial points for each trust region".
        n_trust_regions: m.
        (all other arguments as Turbo1)
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        lb: np.ndarray,
        ub: np.ndarray,
        n_init: int,
        max_evals: int,
        n_trust_regions: int,
        **kwargs,
    ):
        super().__init__(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals, **kwargs)
        assert n_trust_regions > 1, "use Turbo1 for a single trust region"
        assert max_evals > n_trust_regions * n_init, (
            "not enough budget for the initial designs of all trust regions"
        )
        self.n_trust_regions = n_trust_regions
        # App. D: TuRBO-m uses "the same tolerances as in the sequential case (q = 1)".
        self.failtol = default_failtol(self.dim, self.batch_size, n_trust_regions)
        # Turbo1.__init__ already built `self.state` using the TuRBO-1 tolerance, so it is
        # stale by the time we get here. Rebuild it alongside the per-TR states; the
        # TuRBO-m path does not read it, but evaluate.py does read `state.succtol`.
        self.state = self._new_state()
        self.states = [self._new_state() for _ in range(n_trust_regions)]
        # TR ownership per history row; -1 marks points orphaned by a restart.
        self._idx = np.zeros((0, 1), dtype=int)

    def _new_state(self) -> TrustRegionState:
        """A fresh trust-region state carrying the TuRBO-m tolerance.

        `getattr` is needed because `Turbo1.__init__` calls this before `self.failtol` has
        been overwritten with the sequential-case value (App. D); the caller there rebuilds
        the state once the tolerance is set.
        """
        state = super()._new_state()
        state.failtol = getattr(self, "failtol", state.failtol)
        return state

    # --- data access -------------------------------------------------------------
    def _tr_data(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Active (X, fX) belonging to trust region i, in the unit cube."""
        rows = np.where(self._idx.ravel() == i)[0]
        return to_unit_cube(self.X[rows, :], self.lb, self.ub), self.fX[rows, 0].ravel()

    def _append_tr(self, X: np.ndarray, fX: np.ndarray, idx: np.ndarray) -> None:
        self.X = np.vstack((self.X, X))
        self.fX = np.vstack((self.fX, fX))
        self._idx = np.vstack((self._idx, idx))

    def _init_tr(self, i: int) -> None:
        """Draw and evaluate a fresh initial design for trust region i."""
        n_init = max(0, min(self.n_init, self.max_evals - self.n_evals))
        if n_init == 0:
            # [UNSPECIFIED] (PAPER_SPEC.md §10 C4) Neither source says what happens when a
            # restart is triggered with no budget left; the official code adds n_init
            # evaluations after the budget check (turbo_m.py L221) and overshoots.
            # Using: truncate to the remaining budget, and report actual n_evals.
            return
        X_init = from_unit_cube(latin_hypercube(n_init, self.dim, self.rng), self.lb, self.ub)
        fX_init = self._evaluate(X_init)
        self._append_tr(X_init, fX_init, i * np.ones((n_init, 1), dtype=int))

    # --- one batch ---------------------------------------------------------------
    def _propose_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Build candidates and draws for every trust region, then select q points jointly.

        Sect. 2: "we need to select a batch of q candidates drawn from the union of all
        trust regions". The per-region surrogate step is identical to TuRBO-1's and is
        shared via `local_model.propose_from_trust_region`; the joint argmin below is the
        implicit bandit that makes this TuRBO-m rather than m independent runs.

        Returns:
            X_next: (q, d) float64 in the unit cube.
            idx_next: (q, 1) int, the owning trust region of each selected point.
        """
        X_cand = np.zeros((self.n_trust_regions, self.n_cand, self.dim), dtype=np.float64)
        y_cand = np.full(
            (self.n_trust_regions, self.n_cand, self.batch_size), np.inf, dtype=np.float64
        )

        for i, state in enumerate(self.states):
            X_unit, fX = self._tr_data(i)
            proposal = propose_from_trust_region(
                X_unit,
                fX,
                length=state.length,
                batch_size=self.batch_size,
                n_cand=self.n_cand,
                rng=self.rng,
                settings=self.surrogate,
                # A region that received no points last batch has unchanged training data,
                # so its cached hypers are reused and the GP is not refit
                # (PAPER_SPEC.md §10 B8; turbo_m.py L165).
                hypers=state.hypers or None,
            )
            state.hypers = proposal.hypers
            X_cand[i], y_cand[i] = proposal.X_cand, proposal.y_cand

        # Sect. 2: x_i in argmin_l argmin_{x in TR_l} f_l^(i) -- the bandit allocation.
        return select_candidates_across(X_cand, y_cand)

    # --- main loop ---------------------------------------------------------------
    def optimize(self) -> "TurboM":
        """Run until the budget is exhausted, restarting individual regions as they collapse.

        Returns self, so `TurboM(...).optimize()` reads as one expression.
        """
        for i in range(self.n_trust_regions):
            self._init_tr(i)

        while self.n_evals < self.max_evals:
            # Incumbents BEFORE this batch is appended (App. D: "the best solution in this TR").
            incumbents = {}
            for i in range(self.n_trust_regions):
                _, fX_i = self._tr_data(i)
                incumbents[i] = float(fX_i.min()) if fX_i.size else np.inf

            X_next, idx_next = self._propose_all()
            n_take = min(self.batch_size, self.max_evals - self.n_evals)
            X_next, idx_next = X_next[:n_take], idx_next[:n_take]
            X_next = from_unit_cube(X_next, self.lb, self.ub)
            fX_next = self._evaluate(X_next)

            # Update only the TRs that actually received points this batch (App. D: "if
            # q_l > 0 points are selected from this TR").
            for i in range(self.n_trust_regions):
                rows = np.where(idx_next.ravel() == i)[0]
                if rows.size == 0:
                    continue
                fX_i = fX_next[rows]
                self.states[i].update(
                    success=is_success(fX_i, incumbents[i], self.success_tol),
                    n_failures=len(rows),  # App. D: "add q_l to the failure counter"
                )
                # The TR's data changed, so its cached hypers are stale.
                self.states[i].hypers = {}

            self._append_tr(X_next, fX_next, idx_next)

            # Restart any collapsed TR.
            for i, state in enumerate(self.states):
                if state.is_converged:
                    if self.verbose:
                        _, fX_i = self._tr_data(i)
                        print(f"  {self.n_evals}) TR-{i} converged to {fX_i.min():.4g}")
                    state.reset()
                    # [PARTIALLY_SPECIFIED] (PAPER_SPEC.md §10 B9) Sect. 2 says "discard the
                    # respective TR". The official code orphans the points via _idx = -1
                    # (turbo_m.py L192): they stay in the global history for reporting but
                    # no TR's GP ever trains on them again.
                    self._idx[self._idx.ravel() == i, 0] = -1
                    self.n_restarts += 1
                    self._init_tr(i)
        return self
