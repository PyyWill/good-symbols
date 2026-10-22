"""Counterexample-guided Partition Refinement (CPR) for PWA systems (Algorithm 1 in the paper).

The discrete interface of a PWA planning problem (Eq. 3) consists of the mode equalities ``E z = e``, the
discrete-only inequalities ``P z <= p`` and the coupling rows ``B z + B^c xi <= b``.  CPR invents a symbol
basis ``W`` whose assignments ``psi = W z`` leave a conditioned problem with an exact LP relaxation, i.e. a
relative affine-TU certificate (Definition 2.3):

    (1) P = P~ + U_P W + V_P E,    (2) B = U_B W + V_B E,    (3) [P~; E; W] is totally unimodular.

Each round verifies the current ``W``, first re-bases it within its row space (Sparsify) and otherwise
promotes one variable of a minimal non-TU obstruction to a new symbol (Split).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product

import numpy as np

from .tu import is_totally_unimodular

SPARSIFY_RADIUS = 3       # re-basing adds up to RADIUS - 1 signed rows to a row
FLOOR_MAX_COLUMNS = 18    # rank-floor search enumerates {0, +-1} rows of the row space up to this width
FLOOR_MAX_POOL = 5_000
FLOOR_DFS_POOL = 60
FLOOR_DFS_BUDGET = 5_000


@dataclass(frozen=True)
class Certificate:
    """A relative affine-TU certificate; ``W`` has one row per invented symbol."""

    W: np.ndarray
    P_tilde: np.ndarray
    U_P: np.ndarray
    V_P: np.ndarray
    U_B: np.ndarray
    V_B: np.ndarray
    rank_bound: int                 # rank[B; E] - rank E, the lower bound of Lemma 2.1
    splits: tuple[int, ...]         # columns promoted to symbols by Split, in order

    @property
    def k(self) -> int:
        return int(self.W.shape[0])

    def check(self, E, P, B) -> dict[str, bool]:
        """Re-verify the three certificate conditions against the interface."""
        K = self.W.shape[1]
        E, P, B = _rows(E, K), _rows(P, K), _rows(B, K)
        return {
            "W in {0, +-1}": bool(np.isin(self.W, (-1, 0, 1)).all()),
            "(1) P = P~ + U_P W + V_P E": bool(np.array_equal(P, self.P_tilde + self.U_P @ self.W + self.V_P @ E)),
            "(2) B = U_B W + V_B E": bool(np.array_equal(B, self.U_B @ self.W + self.V_B @ E)),
            "(3) [P~; E; W] is TU": is_totally_unimodular(np.vstack([self.P_tilde, E, self.W])),
        }


def cpr(E, P, B, *, radius: int = SPARSIFY_RADIUS) -> Certificate:
    """Invent a symbol basis ``W`` for the discrete interface ``(E, P, B)``."""
    B = np.atleast_2d(np.asarray(B, dtype=int))
    K = B.shape[1]
    E, P = _rows(E, K), _rows(P, K)
    if not is_totally_unimodular(E):
        raise ValueError("CPR assumes a totally unimodular equality block E")
    rank_bound = _rank(np.vstack([B, E])) - _rank(E)

    # Columns with identical [B; E; P] entries are interchangeable for verification: keep one of each.
    keys = [tuple(column) for column in np.vstack([B, E, P]).T]
    first = {key: c for c, key in reversed(list(enumerate(keys)))}
    reps = sorted(first.values())
    expand = np.array([reps.index(first[key]) for key in keys])
    Br, Er, Pr = B[:, reps], E[:, reps], P[:, reps]

    W = _sparsify(_initial_basis(Br, Er), radius)            # W_0: rows of B not recoverable from E
    splits: list[int] = []
    floor_searched = False
    while True:
        M = _verification_matrix(Pr, Er, W)                   # Verify
        certified = is_totally_unimodular(M)
        if not floor_searched and (len(W) > rank_bound or (len(W) == rank_bound and not certified)):
            floor_searched = True                             # Sparsify: one budgeted search at the rank floor
            floor = _rank_floor_basis(Br, Er, Pr, rank_bound)
            if floor is not None:
                W = floor
                continue
        if certified:
            break
        if len(splits) >= 4 * K + 10:
            raise RuntimeError("CPR did not converge")
        core = _strip_unit_rows(M)                            # Split: promote one obstructing column
        c = _unbalanced_cycle_column(core)
        if c is None or not _splittable(W, c):
            candidates = [c for c in _minimal_non_tu_columns(core) if core[:, c].any() and _splittable(W, c)]
            if not candidates:
                raise RuntimeError("non-TU obstruction without a splittable column")
            c = min(candidates, key=lambda c: (not is_totally_unimodular(_verification_matrix(Pr, Er, _split(W, c))),
                                               len(_split(W, c))))
        splits.append(reps[c])
        W = _split(W, c)

    W = W[:, expand]
    P_tilde, U_P, V_P = _residualize(P, W, E)
    coefficients = np.array([_integer_combination(row, np.vstack([W, E])) for row in B], dtype=int)
    return Certificate(W=W, P_tilde=P_tilde, U_P=U_P, V_P=V_P, U_B=coefficients[:, :len(W)],
                       V_B=coefficients[:, len(W):], rank_bound=rank_bound, splits=tuple(splits))


# ----------------------------------------------------------------------------- linear algebra over Z
def _rows(M, K: int) -> np.ndarray:
    return np.zeros((0, K), dtype=int) if M is None or np.size(M) == 0 else np.asarray(M, dtype=int).reshape(-1, K)


def _rank(M: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(M.astype(float))) if M.size else 0


def _unique_rows(rows, K: int) -> np.ndarray:
    out: list[np.ndarray] = []
    for row in rows:
        if not any(np.array_equal(row, kept) for kept in out):
            out.append(np.asarray(row, dtype=int))
    return np.reshape(out, (len(out), K)).astype(int)


def _integer_combination(row, basis) -> np.ndarray | None:
    """Integer ``x`` with ``x @ basis == row``, or None; free coefficients are searched in {-2, ..., 2}."""
    basis, row = np.atleast_2d(np.asarray(basis, dtype=int)), np.asarray(row, dtype=int)
    m = basis.shape[0]
    if m == 0:
        return np.zeros(0, dtype=int) if not row.any() else None
    reduced, pivots = _rref([[Fraction(int(v)) for v in column] + [Fraction(int(rhs))]
                             for column, rhs in zip(basis.T, row)], m)
    if any(r[m] != 0 and not any(r[:m]) for r in reduced):
        return None
    free = [c for c in range(m) if c not in pivots]
    for values in product(range(-2, 3), repeat=len(free)):
        x = dict(zip(free, map(Fraction, values)))
        for i, c in enumerate(pivots):
            x[c] = reduced[i][m] - sum(reduced[i][j] * x[j] for j in free)
        if all(value.denominator == 1 for value in x.values()):
            return np.array([int(x[c]) for c in range(m)], dtype=int)
    return None


def _rref(rows: list[list[Fraction]], ncols: int) -> tuple[list[list[Fraction]], list[int]]:
    """Reduced row echelon form over the rationals of the first ``ncols`` columns."""
    pivots, r = [], 0
    for c in range(ncols):
        if r == len(rows):
            break
        pivot = next((i for i in range(r, len(rows)) if rows[i][c] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        rows[r] = [v / rows[r][c] for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                rows[i] = [a - rows[i][c] * b for a, b in zip(rows[i], rows[r])]
        pivots.append(c)
        r += 1
    return rows, pivots


def _residualize(P: np.ndarray, W: np.ndarray, E: np.ndarray):
    """``P~``: the rows of P that are not integer combinations of [W; E]; their coefficients go to U_P, V_P."""
    basis = np.vstack([W, E])
    P_tilde = np.zeros_like(P)
    U, V = np.zeros((len(P), len(W)), dtype=int), np.zeros((len(P), len(E)), dtype=int)
    for i, row in enumerate(P):
        x = _integer_combination(row, basis)
        if x is None:
            P_tilde[i] = row
        else:
            U[i], V[i] = x[:len(W)], x[len(W):]
    return P_tilde, U, V


def _verification_matrix(P: np.ndarray, E: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.vstack([_residualize(P, W, E)[0], E, W])


# ------------------------------------------------------------------------------------ CPR steps
def _initial_basis(B: np.ndarray, E: np.ndarray) -> np.ndarray:
    selected: list[np.ndarray] = []
    for row in B:
        if _integer_combination(row, np.vstack([np.reshape(selected, (-1, B.shape[1])), E])) is None:
            selected.append(row.copy())
    return _unique_rows(selected, B.shape[1])


def _sparsify(W: np.ndarray, radius: int) -> np.ndarray:
    """Replace rows by sparser {0, +-1} rows of the same row space (signed sums of up to radius-1 others)."""
    rows = [row.copy() for row in W]
    changed = True
    while changed:
        changed = False
        for i in range(len(rows)):
            others = [j for j in range(len(rows)) if j != i]
            found = next((candidate for size in range(1, radius) for subset in combinations(others, size)
                          for signs in product((1, -1), repeat=size)
                          if _sparser(candidate := rows[i] + sum(s * rows[j] for s, j in zip(signs, subset)), rows[i])),
                         None)
            if found is not None:
                rows[i], changed = found, True
                break
    return _unique_rows([row for row in rows if row.any()], W.shape[1])


def _sparser(candidate: np.ndarray, row: np.ndarray) -> bool:
    return bool(np.isin(candidate, (-1, 0, 1)).all() and 0 < np.count_nonzero(candidate) < np.count_nonzero(row))


def _split(W: np.ndarray, c: int) -> np.ndarray:
    """Make ``z_c`` an explicit symbol: remove it from the other rows and append the unit row ``e_c``."""
    rows = W.copy()
    rows[:, c] = 0
    return _unique_rows([*(row for row in rows if row.any()), np.eye(W.shape[1], dtype=int)[c]], W.shape[1])


def _splittable(W: np.ndarray, c: int) -> bool:
    """Isolating column ``c`` strictly refines the basis."""
    return bool(any(row[c] != 0 and np.count_nonzero(row) > 1 for row in W))


def _strip_unit_rows(M: np.ndarray) -> np.ndarray:
    return M[np.count_nonzero(M, axis=1) >= 2]


def _minimal_non_tu_columns(M: np.ndarray) -> list[int]:
    """Columns of an inclusion-minimal non-TU submatrix, by greedy row and column deletion."""
    rows, cols = list(range(M.shape[0])), list(range(M.shape[1]))
    changed = True
    while changed:
        changed = False
        for r in list(rows):
            if len(rows) > 1 and not is_totally_unimodular(M[np.ix_([x for x in rows if x != r], cols)]):
                rows.remove(r)
                changed = True
        for c in list(cols):
            if len(cols) > 1 and not is_totally_unimodular(M[np.ix_(rows, [x for x in cols if x != c])]):
                cols.remove(c)
                changed = True
    return cols


def _unbalanced_cycle_column(M: np.ndarray) -> int | None:
    """Fast obstruction locator on the two-nonzero columns of ``M`` (a signed graph on its rows).

    An unbalanced cycle certifies non-TU; return the column whose removal leaves the fewest of them.
    """
    edges = {}
    for c, column in enumerate(M.T):
        nonzero = np.flatnonzero(column)
        if len(nonzero) == 2:
            i, j = int(nonzero[0]), int(nonzero[1])
            edges[c] = (i, j, int(column[i] * column[j] == 1))
    cycles = _unbalanced_cycles(edges, M.shape[0])
    if not cycles:
        return None
    return min(sorted(set().union(*cycles)),
               key=lambda c: len(_unbalanced_cycles({k: v for k, v in edges.items() if k != c}, M.shape[0])))


def _unbalanced_cycles(edges: dict[int, tuple[int, int, int]], n: int) -> list[set[int]]:
    """Fundamental cycles (w.r.t. a spanning forest) with odd parity, as sets of columns."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree: dict[int, list[tuple[int, int, int]]] = {v: [] for v in range(n)}
    chords = []
    for c, (i, j, parity) in edges.items():
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree[i].append((j, c, parity))
            tree[j].append((i, c, parity))
        else:
            chords.append((c, i, j, parity))

    def path(i: int, j: int) -> list[tuple[int, int]]:
        previous, via, queue = {i: None}, {}, [i]
        for u in queue:
            if u == j:
                break
            for v, c, parity in tree[u]:
                if v not in previous:
                    previous[v], via[v] = u, (c, parity)
                    queue.append(v)
        steps, u = [], j
        while previous[u] is not None:
            steps.append(via[u])
            u = previous[u]
        return steps

    cycles = []
    for c, i, j, parity in chords:
        steps = path(i, j)
        if (parity + sum(p for _, p in steps)) % 2 == 1:
            cycles.append({c, *(column for column, _ in steps)})
    return cycles


# -------------------------------------------------------------------- Sparsify at the rank floor
def _rank_floor_basis(B: np.ndarray, E: np.ndarray, P: np.ndarray, k: int) -> np.ndarray | None:
    """Search (within budget) for a certifying {0, +-1} basis with exactly ``k`` rows in rowspan[B; E]."""
    if k <= 0:
        return None
    pool = _ternary_rowspace(np.vstack([B, E]))
    if not pool:
        return None
    W = _greedy_floor_basis(pool, B, E, P, k)
    if W is None and len(pool) <= FLOOR_DFS_POOL:
        W = _dfs_floor_basis(pool, B, E, P, k)
    return W


def _ternary_rowspace(M: np.ndarray) -> list[np.ndarray]:
    """Sign-normalized {0, +-1} vectors in the row space of ``M`` (meet in the middle), sparsest first."""
    K = M.shape[1]
    if K > FLOOR_MAX_COLUMNS:
        return []
    _, s, vh = np.linalg.svd(M.astype(float), full_matrices=True)
    N = vh[int(np.sum(s > 1e-9)):].T                          # null space: rowspace vectors v satisfy v N = 0
    if N.shape[1] == 0:
        return list(np.eye(K, dtype=int))[:FLOOR_MAX_POOL]
    half = K // 2
    key = lambda values, block: tuple(np.rint(np.asarray(values) @ block * 1e8).astype(np.int64).tolist())
    left: dict[tuple, list[tuple]] = {}
    for values in product((-1, 0, 1), repeat=half):
        left.setdefault(key(values, N[:half]), []).append(values)
    found = set()
    for values in product((-1, 0, 1), repeat=K - half):
        for prefix in left.get(key(values, -N[half:]), []):
            row = np.array(prefix + values, dtype=int)
            if row.any():
                found.add(tuple((row if row[np.flatnonzero(row)[0]] > 0 else -row).tolist()))
    ordered = sorted(found, key=lambda row: (sum(1 for v in row if v), row))
    return [np.array(row, dtype=int) for row in ordered[:FLOOR_MAX_POOL]]


def _certifies_floor(W: np.ndarray, B: np.ndarray, E: np.ndarray, P: np.ndarray, k: int) -> bool:
    basis = np.vstack([W, E])
    return bool(W.shape == (k, B.shape[1]) and np.isin(W, (-1, 0, 1)).all()
                and _rank(basis) == _rank(np.vstack([B, E]))
                and all(_integer_combination(row, basis) is not None for row in B)
                and is_totally_unimodular(_verification_matrix(P, E, W)))


def _greedy_floor_basis(pool, B, E, P, k) -> np.ndarray | None:
    rows, rank = [], _rank(E)
    for row in pool:
        if len(rows) >= k:
            break
        new_rank = _rank(np.vstack([*rows, row, E]))
        if new_rank > rank:
            rows.append(row)
            rank = new_rank
    W = np.reshape(rows, (len(rows), B.shape[1])).astype(int)
    return W if len(rows) == k and _certifies_floor(W, B, E, P, k) else None


def _dfs_floor_basis(pool, B, E, P, k, budget: int = FLOOR_DFS_BUDGET) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    calls = 0

    def extend(start: int, rank: int) -> np.ndarray | None:
        nonlocal calls
        if calls >= budget:
            return None
        if len(rows) == k:
            calls += 1
            W = np.array(rows, dtype=int)
            return W if _certifies_floor(W, B, E, P, k) else None
        if len(pool) - start < k - len(rows):
            return None
        for i in range(start, len(pool)):
            new_rank = _rank(np.vstack([*rows, pool[i], E]))
            if new_rank <= rank:
                continue
            rows.append(pool[i])
            if (W := extend(i + 1, new_rank)) is not None:
                return W
            rows.pop()
            if calls >= budget:
                return None
        return None

    return extend(0, _rank(E))
