"""The total-unimodularity oracle IsTU used by CPR.

A {0, +-1} matrix whose columns have at most two nonzeros is decided exactly by a signed-graph balance
test (Heller and Tompkins).  Any other matrix is passed to ``cmr-tu`` from the Combinatorial Matrix
Recognition library (https://github.com/discopt/cmr), located through the ``CMR_TU`` environment variable
or on ``PATH``.  Without it, only matrices with few square submatrices are checked by enumerating minors.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from itertools import combinations
from math import comb
import os
import shutil
import subprocess

import numpy as np

CMR_TIME_LIMIT = 30.0           # seconds per cmr-tu call
MAX_ENUMERATED_MINORS = 200_000  # largest brute-force check when cmr-tu is unavailable


class TUOracleError(RuntimeError):
    """Total unimodularity could not be decided."""


def is_totally_unimodular(matrix) -> bool:
    """Return whether every square submatrix of ``matrix`` has determinant in {-1, 0, 1}."""
    M = np.atleast_2d(np.asarray(matrix, dtype=int))
    if M.size == 0:
        return True
    if not np.isin(M, (-1, 0, 1)).all():
        return False
    balanced = _signed_graph_balanced(M)
    if balanced is not None:
        return balanced
    binary = os.environ.get("CMR_TU") or shutil.which("cmr-tu")
    if binary:
        return _cmr_tu(binary, M.shape, M.astype(np.int8).tobytes())
    if sum(comb(M.shape[0], k) * comb(M.shape[1], k) for k in range(1, min(M.shape) + 1)) <= MAX_ENUMERATED_MINORS:
        return _all_minors_unimodular(M)
    raise TUOracleError(f"deciding total unimodularity of a {M.shape[0]}x{M.shape[1]} matrix needs cmr-tu: "
                        "build https://github.com/discopt/cmr and put cmr-tu on PATH or set CMR_TU")


def _signed_graph_balanced(M: np.ndarray) -> bool | None:
    """Exact test when no column has more than two nonzeros, else ``None``.

    Such a matrix is TU iff its rows split into two classes where every column's two nonzeros lie in the
    same class when their signs differ and in different classes when they agree.
    """
    neighbours: list[list[tuple[int, int]]] = [[] for _ in range(M.shape[0])]
    for column in M.T:
        rows = np.flatnonzero(column)
        if len(rows) > 2:
            return None
        if len(rows) == 2:
            i, j = int(rows[0]), int(rows[1])
            parity = int(column[i] == column[j])
            neighbours[i].append((j, parity))
            neighbours[j].append((i, parity))
    side: list[int | None] = [None] * M.shape[0]
    for root in range(M.shape[0]):
        if side[root] is not None:
            continue
        side[root], queue = 0, deque([root])
        while queue:
            u = queue.popleft()
            for v, parity in neighbours[u]:
                if side[v] is None:
                    side[v] = side[u] ^ parity
                    queue.append(v)
                elif side[v] != side[u] ^ parity:
                    return False
    return True


@lru_cache(maxsize=20_000)
def _cmr_tu(binary: str, shape: tuple[int, int], payload: bytes) -> bool:
    M = np.frombuffer(payload, dtype=np.int8).reshape(shape)
    text = "\n".join([f"{shape[0]} {shape[1]}", *(" ".join(map(str, row)) for row in M)]) + "\n"
    try:
        result = subprocess.run([binary, "-", "--time-limit", str(CMR_TIME_LIMIT)], input=text, text=True,
                                capture_output=True, timeout=CMR_TIME_LIMIT + 5)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise TUOracleError(f"cmr-tu failed: {error}") from error
    output = result.stdout + result.stderr
    if "IS NOT totally unimodular" in output:
        return False
    if "IS totally unimodular" in output:
        return True
    raise TUOracleError(f"cannot parse cmr-tu output: {output.strip()[:200]}")


def _all_minors_unimodular(M: np.ndarray) -> bool:
    m, n = M.shape
    return all(round(float(np.linalg.det(M[np.ix_(rows, cols)]))) in (-1, 0, 1) for k in range(1, min(m, n) + 1)
               for rows in combinations(range(m), k) for cols in combinations(range(n), k))
