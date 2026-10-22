"""Mixed-integer linear programs as data.

Every task writes its finite-horizon planning problem P_PWA (Eq. 3) as a :class:`MILP` whose first ``H * nz``
variables are the binary modes ``z_0, ..., z_{H-1}`` (stage-major).  Symbols ``psi_t = W z_t`` act on exactly
these columns, which is all the symbol-guided controller needs to know about a task's formulation.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp
from scipy.sparse import csr_matrix

Terms = Mapping[int, float] | Iterable[tuple[int, float]]


@dataclass(frozen=True)
class MILP:
    """``min c^T x``  s.t.  ``constraints.lb <= A x <= constraints.ub``, ``bounds``, integrality."""

    cost: np.ndarray
    integrality: np.ndarray
    bounds: Bounds
    constraints: LinearConstraint

    def solve(self, *, time_limit: float, mip_rel_gap: float) -> OptimizeResult:
        """Solve with HiGHS (through SciPy)."""
        return milp(self.cost, integrality=self.integrality, bounds=self.bounds, constraints=self.constraints,
                    options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap})


class Variables:
    """Allocates consecutive variable indices in named blocks, e.g. ``z = variables.add(H, nz)``."""

    def __init__(self):
        self.n = 0

    def add(self, *shape: int) -> np.ndarray:
        size = int(np.prod(shape))
        block = np.arange(self.n, self.n + size).reshape(shape)
        self.n += size
        return block


class Rows:
    """Sparse constraint rows ``lower <= a^T x <= upper`` over ``n`` variables, assembled once.

    ``terms`` is a mapping or an iterable of ``(column, coefficient)`` pairs; repeated columns accumulate.
    An optional ``tag`` labels a row (e.g. the coupling rows that form ``B``).
    """

    def __init__(self, n: int):
        self.n = n
        self.columns: list[int] = []
        self.values: list[float] = []
        self.starts = [0]
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.tags: list = []

    def __len__(self) -> int:
        return len(self.lower)

    def add(self, terms: Terms, lower: float, upper: float, tag=None) -> int:
        row: dict[int, float] = {}
        for column, value in (terms.items() if isinstance(terms, Mapping) else terms):
            row[int(column)] = row.get(int(column), 0.0) + float(value)
        for column in sorted(row):
            if abs(row[column]) > 1e-14:
                self.columns.append(column)
                self.values.append(row[column])
        self.starts.append(len(self.columns))
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        self.tags.append(tag)
        return len(self.lower) - 1

    def eq(self, terms: Terms, rhs: float, tag=None) -> int:
        return self.add(terms, rhs, rhs, tag)

    def le(self, terms: Terms, rhs: float, tag=None) -> int:
        return self.add(terms, -np.inf, rhs, tag)

    def ge(self, terms: Terms, rhs: float, tag=None) -> int:
        return self.add(terms, rhs, np.inf, tag)

    def matrix(self) -> csr_matrix:
        return csr_matrix((self.values, self.columns, self.starts), shape=(len(self), self.n))

    def constraint(self) -> LinearConstraint:
        return LinearConstraint(self.matrix(), np.asarray(self.lower), np.asarray(self.upper))
