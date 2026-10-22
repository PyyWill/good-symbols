"""Symbols psi = W z and the graph of symbol transitions witnessed in demonstrations."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
import json
from pathlib import Path

import numpy as np

from .demonstrations import Demonstration
from .lp import ConditionedLP
from .milp import MILP


def symbol(z, W, tolerance: float = 1e-6) -> tuple[int, ...]:
    """The symbol assignment ``psi = W z`` of a mode ``z``."""
    psi = np.asarray(W) @ np.asarray(z)
    if not np.isfinite(psi).all() or np.max(np.abs(psi - np.rint(psi)), initial=0) > tolerance:
        raise ValueError("non-integral symbol")
    return tuple(np.rint(psi).astype(int).tolist())


class SymbolGraph:
    """Symbols (nodes) and transitions (edges) known to be legal; anything unseen is treated as illegal."""

    def __init__(self, W, nodes: Sequence[Sequence[int]], edges: Sequence[Sequence[int]], **metadata):
        self.W = np.asarray(W, dtype=int)
        self.nodes = tuple(tuple(int(v) for v in node) for node in nodes)
        self.index = {node: i for i, node in enumerate(self.nodes)}
        self.edges = frozenset((int(a), int(b)) for a, b in edges)
        self.metadata = metadata
        if self.W.ndim != 2 or not self.W.size or len(self.index) != len(self.nodes):
            raise ValueError("a symbol graph needs a 2-D W and unique nodes")
        if any(len(node) != len(self.W) for node in self.nodes) or any(not 0 <= i < len(self.nodes)
                                                                     for edge in self.edges for i in edge):
            raise ValueError("nodes must match W and edges must index nodes")

    @classmethod
    def load(cls, path: Path | str) -> SymbolGraph:
        data = json.loads(Path(path).read_text())
        return cls(data.pop("W"), data.pop("nodes"), data.pop("edges"), **data)

    def save(self, path: Path | str) -> None:
        data = {"W": self.W.tolist(), "nodes": [list(n) for n in self.nodes], "edges": sorted(self.edges),
                **self.metadata}
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    def observed(self, psi, previous: int | None = None) -> int:
        """Node of an observed symbol; raises if the symbol or the transition from ``previous`` is unseen."""
        node = self.index.get(tuple(psi))
        if node is None:
            raise ValueError("unseen symbol")
        if previous is not None and (previous, node) not in self.edges:
            raise ValueError("unseen symbol transition")
        return node

    def extended(self, demonstrations: Sequence[Demonstration]) -> SymbolGraph:
        """Add the symbols and transitions executed by successful demonstrations."""
        nodes, index, edges = list(self.nodes), dict(self.index), set(self.edges)
        for demo in demonstrations:
            if not demo.success:
                continue
            path = []
            for z in demo.modes:
                psi = symbol(z, self.W)
                if psi not in index:
                    index[psi] = len(nodes)
                    nodes.append(psi)
                path.append(index[psi])
            edges.update(zip(path, path[1:]))
        return SymbolGraph(self.W, nodes, sorted(edges), **self.metadata)


def build_symbol_graph(W, demonstrations: Sequence[Demonstration], problem_at: Callable[[Demonstration, int], MILP],
                       horizon: int) -> tuple[SymbolGraph, dict]:
    """Union of the executed transitions and the natively feasible planned transitions of demonstrations.

    ``problem_at(demo, t)`` returns the native MILP at step ``t`` of ``demo``; each saved H-step plan is kept
    only if its conditioned LP is feasible, so no edge is synthesized.
    """
    W = np.asarray(W, dtype=int)
    nodes, edges = set(), set()
    audit = {"episodes": 0, "plans": 0, "rejected_plans": []}
    for i, demo in enumerate(demonstrations):
        if not demo.success:
            continue
        audit["episodes"] += 1
        executed = [symbol(z, W) for z in demo.modes]
        nodes.update(executed)
        edges.update(zip(executed, executed[1:]))
        for t, plan in enumerate(demo.plans):
            if np.max(np.abs(plan - np.rint(plan)), initial=0) > 1e-5:
                raise ValueError(f"non-integral plan in demonstration {i}, step {t}")
            psi = [symbol(z, W) for z in np.rint(plan)]
            lp = ConditionedLP(problem_at(demo, t), W, horizon)
            if lp.solve(psi, deadline=np.inf) is None:
                audit["rejected_plans"].append((i, t))
                continue
            nodes.update(psi)
            edges.update(zip(psi, psi[1:]))
            audit["plans"] += 1
    ordered = sorted(nodes)
    index = {node: i for i, node in enumerate(ordered)}
    graph = SymbolGraph(W, ordered, sorted((index[a], index[b]) for a, b in edges), schema=1,
                        rule="Executed successful-demo edges plus native-LP-validated saved horizon edges; "
                             "no synthetic edges.")
    return graph, audit


def file_digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
