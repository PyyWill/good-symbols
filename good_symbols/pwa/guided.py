"""Symbol-guided control (Fig. 1b): search over symbol plans, then solve the conditioned LP.

At every step the controller lifts the observed mode to its symbol ``psi_t = W z_t``, proposes H-step symbol
plans ``psi_{t:t+H}`` without calling a solver, and injects each into the conditioned LP of the native problem
(:mod:`good_symbols.pwa.lp`).  The lowest-cost feasible plan's first action is executed.

Plans are proposed from a library of the symbol plans that the end-to-end controller chose in demonstrations,
indexed by the symbol observed when they were chosen.  Candidates are ordered by how well their context (the
native problem's state-dependent right-hand side and the time spent in the current symbol) matches the
present one, and filtered by bound propagation.  A single plan is proposed away from a phase boundary; where
demonstrations disagree about leaving the current symbol, up to ``max_candidates`` plans are compared by
their conditioned objective.  Unseen symbols or transitions end the episode.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import time

import numpy as np

from .control import ControlFailure, Decision, decode
from .demonstrations import Demonstration, load_demonstrations
from .lp import ConditionedLP, LPCache
from .milp import MILP
from .propagation import BoundPropagator
from .symbols import SymbolGraph, symbol
from .tasks.base import Task

Plan = tuple[int, ...]   # an H-step symbol plan, as symbol-graph node indices


@dataclass(frozen=True)
class _Contexts:
    """Every demonstrated decision taken while one symbol was observed."""

    embeddings: np.ndarray        # standardized state-dependent right-hand sides of the native problem
    plans: tuple[Plan, ...]
    mode_plans: tuple[np.ndarray, ...]
    dwell: np.ndarray             # steps already spent in the symbol when the plan was chosen


class SymbolPlanLibrary:
    """Demonstrated symbol plans indexed by the observed symbol; built offline, solver-free online."""

    def __init__(self, task: Task, graph: SymbolGraph, demonstrations: Sequence[Demonstration]):
        H, W = task.horizon, graph.W
        self.horizon = H
        self.next_position = 1 if task.pins_initial_mode else 0   # plan position of the next commanded symbol
        contexts: dict[int, list] = {}
        right_hand_sides, template = [], None
        executed, self.dwell_max, final, inner = set(), {}, set(), set()
        for demo in demonstrations:
            if demo.setting != task.setting or not demo.success:
                continue
            path = [graph.index[symbol(z, W)] for z in demo.modes]
            final.add(path[-1])
            inner.update(path[:-1])
            dwell = [1]
            for a, b in zip(path, path[1:]):
                executed.add((a, b))
                dwell.append(dwell[-1] + 1 if b == a else 1)
                self.dwell_max[b] = max(self.dwell_max.get(b, 1), dwell[-1])
            for t, mode_plan in enumerate(demo.plans):
                mode_plan = np.rint(mode_plan[:H])
                plan = tuple(graph.index.get(symbol(z, W)) for z in mode_plan)
                if None in plan:
                    continue
                problem = task.milp(demo.states[t], demo.goal)
                template = template or problem
                rhs = np.asarray(problem.constraints.lb, dtype=float)
                right_hand_sides.append(rhs)
                contexts.setdefault(path[t], []).append((rhs, plan, mode_plan.ravel(), dwell[t]))
        if template is None:
            raise ValueError(f"no demonstrations for {task.name} setting {task.setting}")
        rhs = np.asarray(right_hand_sides)
        finite = np.isfinite(rhs).all(0)
        spread = np.where(finite, np.where(finite, rhs, 0.0).std(0), 0.0)
        self.columns = np.flatnonzero(spread > 1e-9)                    # state-dependent entries only
        self.mean, self.scale = rhs[:, self.columns].mean(0), spread[self.columns]
        self.contexts = {node: _Contexts(np.asarray([self._embed(c[0]) for c in items]), tuple(c[1] for c in items),
                                         tuple(c[2] for c in items), np.asarray([c[3] for c in items], dtype=int))
                         for node, items in contexts.items()}
        self.executed = frozenset(executed)
        self.terminal = frozenset(final - inner)                           # symbols seen only at episode ends
        self.propagator = BoundPropagator(template, H * W.shape[1])

    def _embed(self, rhs: np.ndarray) -> np.ndarray:
        return (np.asarray(rhs, dtype=float)[self.columns] - self.mean) / self.scale

    def propose(self, current: int, dwell: int, problem: MILP, previous: Plan | None,
                max_candidates: int, window: int = 1) -> tuple[list[Plan], str]:
        """Plans to try at this step: one away from a phase boundary, up to ``max_candidates`` at one."""
        entry = self.contexts.get(current)
        if entry is None:
            raise ControlFailure(f"symbol {current} has no demonstrated plans")
        plans, pos = entry.plans, self.next_position
        distance = ((entry.embeddings - self._embed(problem.constraints.lb)) ** 2).sum(1)
        shifted = previous[1:] if previous else None
        propagator = self.propagator.update(problem)

        def continues(plan: Plan) -> int:
            return 0 if shifted is not None and plan[:len(shifted)] == shifted else 1

        def leaves(plan: Plan) -> bool:
            return any(node != current for node in plan[pos:])

        def feasible(order: list[int], limit: int) -> list[Plan]:
            picked, seen = [], set()
            for j in order:
                if plans[j] in seen:
                    continue
                seen.add(plans[j])
                if not propagator.infeasible(entry.mode_plans[j], sweeps=1):
                    picked.append(plans[j])
                    if len(picked) >= limit:
                        return picked
            if not picked:
                raise ControlFailure(f"bound propagation rejects all {len(seen)} plans of symbol {current}")
            return picked

        gap = np.abs(entry.dwell - dwell)
        near = np.flatnonzero(gap <= window) if gap.min() <= window else np.flatnonzero(gap == gap.min())
        if any(leaves(plans[j]) for j in near):
            # Phase boundary: demonstrations disagree about leaving; compare candidates by their objective.
            key = lambda j: (plans[j][pos] in self.terminal, gap[j], continues(plans[j]), distance[j])
            stay = sorted((j for j in range(len(plans)) if not leaves(plans[j])), key=key)[:1]
            picked = feasible(sorted(near.tolist(), key=key) + stay, max_candidates)
            return picked, f"boundary (dwell {dwell}, {len(picked)} candidates)"

        over_budget = dwell > self.dwell_max.get(current, 1)
        has_executed_exit = any((current, b) in self.executed for b in {plan[pos] for plan in plans} if b != current)

        def progress(plan: Plan) -> int:
            if plan[pos] != current:
                return 2 if has_executed_exit and (current, plan[pos]) not in self.executed else 0
            return 0 if leaves(plan) else 1

        key = lambda j: (plans[j][pos] in self.terminal, progress(plans[j]) if over_budget else 0, gap[j],
                         continues(plans[j]), distance[j])
        picked = feasible(sorted(range(len(plans)), key=key), 1)
        return picked, f"dwell {dwell}: {'stay' if picked[0][pos] == current else 'leave'}"


class SymbolGuidedController:
    """Lift, search the symbol space, inject the symbols and solve the conditioned LP (Fig. 1b)."""

    def __init__(self, task: Task, graph: SymbolGraph, library: SymbolPlanLibrary, max_candidates: int = 4):
        self.task, self.graph, self.library, self.max_candidates = task, graph, library, max_candidates
        self.W = graph.W
        self.reset_episode()

    @classmethod
    def from_data(cls, task: Task, **kwargs) -> SymbolGuidedController:
        """Load the released symbol graph and demonstrations of ``task``."""
        demonstrations = load_demonstrations(sorted(task.data_dir.glob("demonstrations*.npz")))
        graph = SymbolGraph.load(task.data_dir / "symbol_graph.json").extended(demonstrations)
        return cls(task, graph, SymbolPlanLibrary(task, graph, demonstrations), **kwargs)

    def reset_episode(self) -> None:
        self.cache = LPCache()
        self.current: int | None = None
        self.dwell = 1
        self.previous: Plan | None = None

    def reset(self, state: np.ndarray) -> None:
        self.reset_episode()
        self.current = self.graph.observed(symbol(self.task.mode(state), self.W))

    def observe(self, state: np.ndarray) -> None:
        node = self.graph.observed(symbol(self.task.mode(state), self.W), self.current)
        self.dwell = self.dwell + 1 if node == self.current else 1
        self.current = node

    def act(self, state: np.ndarray, goal: np.ndarray) -> Decision:
        problem = self.task.milp(state, goal)
        lp = ConditionedLP(problem, self.W, self.task.horizon, self.cache)
        deadline = time.perf_counter() + self.task.config.time_limit        # per-decision budget of the LPs
        plans, note = self.library.propose(self.current, self.dwell, problem, self.previous, self.max_candidates)
        best = None
        for plan in plans:
            psi = np.asarray([self.graph.nodes[i] for i in plan], dtype=float)
            result = lp.solve(psi, deadline)
            if result is not None and (best is None or result.fun < best[1].fun):
                best = (plan, result, psi)
        if best is None:
            raise ControlFailure(f"conditioned LP infeasible for all {len(plans)} symbol plans")
        plan, result, psi = best
        x, mode_plan = decode(self.task, result.x)
        if np.max(np.abs(mode_plan @ self.W.T - psi), initial=0) > 1e-6:
            raise ControlFailure("the conditioned solution does not realize the injected symbols")
        self.previous = plan
        return Decision(self.task.action(x), mode_plan, float(result.fun),
                        {"candidates": len(plans), "plan": list(plan), "note": note})
