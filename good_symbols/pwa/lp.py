"""The conditioned problem P^psi: the native MILP's LP relaxation with a symbol plan injected.

Fixing ``psi_t = W z_t`` for every stage leaves an LP whose optimal ``z`` is integral when ``W`` is a relative
affine-TU certificate, so one LP solve replaces the mixed-integer search (Section 2.2).  Consecutive solves
within an episode share one persistent HiGHS model and warm-start from the previous basis.
"""
from __future__ import annotations

import hashlib
import time

import highspy
import numpy as np
from scipy import linalg, sparse
from scipy.optimize import OptimizeResult

from .milp import MILP

TOLERANCE = 1e-6


class ConditionedLP:
    """``min c^T x`` over the native constraints with integrality dropped and ``(I_H kron W) z = psi``."""

    def __init__(self, problem: MILP, W: np.ndarray, horizon: int, cache: LPCache | None = None):
        self.cost, self.integrality = problem.cost, problem.integrality
        self.bounds, self.constraints = problem.bounds, problem.constraints
        n, nz = len(self.cost), W.shape[1]
        self.projection = sparse.hstack([sparse.kron(sparse.eye(horizon), W),
                                         sparse.csr_matrix((horizon * len(W), n - horizon * nz))]).tocsr()
        self._model = (cache if cache is not None else LPCache()).bind(self)

    def solve(self, psi, deadline: float) -> OptimizeResult | None:
        """Optimal solution for the stage-wise symbol plan ``psi`` (H x k), or None if infeasible."""
        return self._model.solve(self, np.asarray(psi, dtype=float).ravel(), deadline)

    def residuals(self, result: OptimizeResult, psi) -> dict[str, float]:
        """Integrality, symbol and primal residuals of a solution; raises if any exceeds the tolerance."""
        x = result.x
        z = x[self.integrality != 0]
        activity = self.constraints.A @ x
        errors = {
            "integer": float(np.max(np.abs(z - np.rint(z)), initial=0)),
            "psi": float(np.max(np.abs(self.projection @ x - np.asarray(psi).ravel()), initial=0)),
            "primal": float(max(np.max(self.constraints.lb - activity, initial=0),
                                np.max(activity - self.constraints.ub, initial=0),
                                np.max(self.bounds.lb - x, initial=0), np.max(x - self.bounds.ub, initial=0))),
        }
        if max(errors.values()) > TOLERANCE:
            raise RuntimeError(f"conditioned LP solution violates its constraints: {errors}")
        return errors


class LPCache:
    """One persistent LP per episode, rebuilt only when the matrix, cost or variable bounds change."""

    def __init__(self, eliminate_equalities: bool = True):
        self.eliminate_equalities = eliminate_equalities
        self.model: _PersistentLP | None = None
        self._key: bytes | None = None

    def bind(self, lp: ConditionedLP) -> _PersistentLP:
        key = _structure_key(lp)
        if key != self._key:
            self.model, self._key = _PersistentLP(lp, self.eliminate_equalities), key
        return self.model


def _structure_key(lp: ConditionedLP) -> bytes:
    digest = hashlib.sha256()
    for matrix in (sparse.csr_matrix(lp.constraints.A), lp.projection):
        for array in (matrix.shape, matrix.indptr, matrix.indices, matrix.data):
            digest.update(np.ascontiguousarray(array).tobytes())
    for array in (lp.cost, lp.bounds.lb, lp.bounds.ub, np.isfinite(lp.constraints.lb),
                  np.isfinite(lp.constraints.ub), lp.constraints.lb == lp.constraints.ub):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.digest()


class _PersistentLP:
    """A HiGHS LP over the null space of the equality rows; per solve only row bounds change.

    Equalities (native ones and the injected symbol rows) are eliminated once: ``x = offset(rhs) + T y``.
    The remaining rows, including the native variable bounds, are deduplicated and kept as row bounds.
    """

    def __init__(self, lp: ConditionedLP, eliminate: bool):
        self.eliminate = eliminate
        self.A = sparse.vstack([lp.constraints.A, lp.projection]).tocsr()
        n = len(lp.cost)
        if eliminate:
            equalities = np.flatnonzero(np.isfinite(lp.constraints.lb) & (lp.constraints.lb == lp.constraints.ub))
            equalities = np.r_[equalities, np.arange(len(lp.constraints.lb), self.A.shape[0])]
            E = self.A[equalities].toarray()
            _, R, row_order = linalg.qr(E.T, mode="economic", pivoting=True)
            rank = int(np.sum(np.abs(np.diag(R)) > 1e-10))
            independent = E[row_order[:rank]]
            _, _, column_order = linalg.qr(independent, mode="economic", pivoting=True)
            self.pivots, free = column_order[:rank], column_order[rank:]
            self.equality_rows = equalities[row_order[:rank]]
            self.inverse = linalg.solve(independent[:, self.pivots], np.eye(rank))
            self.transform = np.zeros((n, len(free)))
            self.transform[free] = np.eye(len(free))
            self.transform[self.pivots] = -self.inverse @ independent[:, free]
            if np.max(np.abs(E @ self.transform), initial=0) > 1e-9:
                raise RuntimeError("equality elimination failed")
            self.A = sparse.vstack([self.A, sparse.eye(n)]).tocsr()
            reduced = self.A @ self.transform
            reduced[np.abs(reduced) < 1e-12] = 0.0
            self.nonzero = np.any(reduced != 0.0, axis=1)
            matrix, self.group = np.unique(reduced[self.nonzero], axis=0, return_inverse=True)
            self.group = np.ravel(self.group)
            cost, lower, upper = self.transform.T @ lp.cost, lp.bounds.lb[free], lp.bounds.ub[free]
        else:
            matrix, cost, lower, upper = self.A, lp.cost, lp.bounds.lb, lp.bounds.ub
        matrix = sparse.csr_matrix(matrix)
        self.shape = matrix.shape
        model = highspy.HighsLp()
        model.num_row_, model.num_col_ = matrix.shape
        model.col_cost_, model.col_lower_, model.col_upper_ = cost, lower, upper
        self.row_lower, self.row_upper = np.full(matrix.shape[0], -np.inf), np.full(matrix.shape[0], np.inf)
        model.row_lower_, model.row_upper_ = self.row_lower, self.row_upper
        model.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        model.a_matrix_.start_, model.a_matrix_.index_ = matrix.indptr, matrix.indices
        model.a_matrix_.value_ = matrix.data
        self.highs = highspy.Highs()
        for option, value in (("output_flag", False), ("threads", 1), ("solver", "simplex"),
                              ("primal_feasibility_tolerance", 1e-8), ("dual_feasibility_tolerance", 1e-8)):
            if self.highs.setOptionValue(option, value) != highspy.HighsStatus.kOk:
                raise RuntimeError(f"cannot set HiGHS option {option}")
        if self.highs.passModel(model) == highspy.HighsStatus.kError:
            raise RuntimeError("cannot construct the conditioned LP")

    def solve(self, lp: ConditionedLP, psi: np.ndarray, deadline: float) -> OptimizeResult | None:
        if psi.size != lp.projection.shape[0]:
            raise ValueError("the conditioned LP needs a symbol for every stage of the horizon")
        lower, upper = np.r_[lp.constraints.lb, psi], np.r_[lp.constraints.ub, psi]
        if self.eliminate:
            offset = np.zeros(len(lp.cost))
            offset[self.pivots] = self.inverse @ lower[self.equality_rows]
            activity = self.A @ offset
            lower = np.r_[lower, lp.bounds.lb] - activity
            upper = np.r_[upper, lp.bounds.ub] - activity
            if np.max(lower[~self.nonzero], initial=0) > 1e-7 or np.min(upper[~self.nonzero], initial=0) < -1e-7:
                return None
            row_lower, row_upper = np.full(self.shape[0], -np.inf), np.full(self.shape[0], np.inf)
            np.maximum.at(row_lower, self.group, lower[self.nonzero])
            np.minimum.at(row_upper, self.group, upper[self.nonzero])
            lower, upper = row_lower, row_upper
        if np.any(lower > upper + 1e-7):
            return None
        if self.eliminate and self.shape[1] == 0:                # the symbols fix every variable
            result = OptimizeResult(x=offset, fun=float(lp.cost @ offset), success=True, status=0)
            lp.residuals(result, psi)
            return result
        changed = np.flatnonzero((lower != self.row_lower) | (upper != self.row_upper)).astype(np.int32)
        if hasattr(self.highs, "changeRowsBounds"):
            status = self.highs.changeRowsBounds(len(changed), changed, lower[changed], upper[changed])
        else:                                                    # older highspy: single-row binding only
            failed = [r for r in changed if self.highs.changeRowBounds(int(r), float(lower[r]), float(upper[r]))
                      == highspy.HighsStatus.kError]
            status = highspy.HighsStatus.kError if failed else highspy.HighsStatus.kOk
        if status == highspy.HighsStatus.kError:
            raise RuntimeError("cannot update the conditioned LP")
        self.row_lower, self.row_upper = lower, upper
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("decision time budget exhausted")
        self.highs.setOptionValue("time_limit", self.highs.getRunTime() + remaining)  # HiGHS's clock accumulates
        self.highs.run()
        status = self.highs.getModelStatus()
        if status == highspy.HighsModelStatus.kInfeasible:
            return None
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"conditioned LP: {self.highs.modelStatusToString(status)}")
        x = np.asarray(self.highs.getSolution().col_value)
        if self.eliminate:
            x = offset + self.transform @ x
        result = OptimizeResult(x=x, fun=float(lp.cost @ x), success=True, status=0)
        lp.residuals(result, psi)
        return result
