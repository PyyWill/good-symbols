"""Solver-free feasibility test for a candidate mode plan (feasibility-based bound tightening)."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .milp import MILP


class BoundPropagator:
    """Interval propagation over the native rows with the first ``pinned`` (mode) variables fixed.

    ``infeasible`` is a sound necessary test: it rejects a plan only when some row cannot meet its bounds under
    interval arithmetic or a continuous variable's bounds cross.  Pinned variables are never tightened and
    every comparison uses a relative tolerance.
    """

    def __init__(self, problem: MILP, pinned: int, tolerance: float = 1e-6):
        A = sparse.csr_matrix(problem.constraints.A)
        self.tolerance = tolerance
        self.positive, self.negative = A.maximum(0).tocsr(), A.minimum(0).tocsr()
        entries = A.tocoo()
        order = np.argsort(entries.col, kind="stable")                    # entries grouped by column
        self.rows, self.cols, self.coefficients = entries.row[order], entries.col[order], entries.data[order]
        self.segments = np.flatnonzero(np.r_[True, self.cols[1:] != self.cols[:-1]])
        self.segment_cols = self.cols[self.segments]
        self.free = self.segment_cols >= pinned
        self.update(problem)

    def update(self, problem: MILP) -> BoundPropagator:
        """Adopt this step's right-hand sides and variable bounds (same matrix)."""
        self.lower = np.asarray(problem.constraints.lb, dtype=float)
        self.upper = np.asarray(problem.constraints.ub, dtype=float)
        self.var_lower = np.asarray(problem.bounds.lb, dtype=float)
        self.var_upper = np.asarray(problem.bounds.ub, dtype=float)
        finite = lambda v: np.abs(np.where(np.isfinite(v), v, 0.0))
        self.row_tolerance = self.tolerance * (1 + finite(self.lower) + finite(self.upper))
        return self

    def infeasible(self, plan, sweeps: int = 20) -> bool:
        lo, hi = self.var_lower.copy(), self.var_upper.copy()
        z = np.asarray(plan, dtype=float).ravel()
        lo[:z.size] = hi[:z.size] = z
        rows, cols, coefficients = self.rows, self.cols, self.coefficients
        positive = coefficients > 0
        columns = self.segment_cols[self.free]
        with np.errstate(invalid="ignore", divide="ignore"):
            for _ in range(sweeps):
                row_min = self.positive @ lo + self.negative @ hi
                row_max = self.positive @ hi + self.negative @ lo
                if (np.any(row_max < self.lower - self.row_tolerance)
                        or np.any(row_min > self.upper + self.row_tolerance)):
                    return True
                slack_up, slack_down = self.upper - row_min, row_max - self.lower
                candidate_hi = np.where(positive, lo[cols] + slack_up[rows] / coefficients,
                                        lo[cols] - slack_down[rows] / coefficients)
                candidate_lo = np.where(positive, hi[cols] - slack_down[rows] / coefficients,
                                        hi[cols] + slack_up[rows] / coefficients)
                candidate_hi = np.where(np.isfinite(candidate_hi), candidate_hi, np.inf)
                candidate_lo = np.where(np.isfinite(candidate_lo), candidate_lo, -np.inf)
                new_hi, new_lo = hi.copy(), lo.copy()
                new_hi[columns] = np.minimum(hi[columns], np.minimum.reduceat(candidate_hi, self.segments)[self.free])
                new_lo[columns] = np.maximum(lo[columns], np.maximum.reduceat(candidate_lo, self.segments)[self.free])
                if np.any(new_lo > new_hi + self.tolerance * (1 + np.abs(new_hi))):
                    return True
                change = max(np.max(np.abs(np.where(np.isfinite(new_hi) & np.isfinite(hi), new_hi - hi, 0))),
                             np.max(np.abs(np.where(np.isfinite(new_lo) & np.isfinite(lo), new_lo - lo, 0))))
                lo, hi = new_lo, new_hi
                if change < 1e-9:
                    break
        return False
