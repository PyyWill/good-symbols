"""Receding-horizon control: at every step solve a horizon-H problem, execute its first action.

``E2EController`` solves the native MILP P_PWA end to end (the reference with speed-up 1.00x in Table 1).
The symbol-guided controller in :mod:`good_symbols.pwa.guided` solves conditioned LPs instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .tasks.base import Task

INTEGRALITY_TOLERANCE = 1e-5


class ControlFailure(RuntimeError):
    """The controller cannot produce an action at this state."""


@dataclass
class Decision:
    action: np.ndarray
    plan: np.ndarray            # planned modes z_0, ..., z_{H-1} (H x nz)
    objective: float
    info: dict = field(default_factory=dict)


class Controller(Protocol):
    def reset(self, state: np.ndarray) -> None: ...
    def act(self, state: np.ndarray, goal: np.ndarray) -> Decision: ...
    def observe(self, state: np.ndarray) -> None: ...


def decode(task: Task, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round the mode block of a solution and return ``(x, z_plan)``; raises if it is not integral."""
    x = np.array(x, dtype=float)
    n = task.horizon * task.nz
    violation = float(np.max(np.abs(x[:n] - np.rint(x[:n])), initial=0))
    if violation > INTEGRALITY_TOLERANCE:
        raise ControlFailure(f"non-integral modes ({violation:.1e})")
    x[:n] = np.rint(x[:n])
    return x, x[:n].reshape(task.horizon, task.nz)


class E2EController:
    """End-to-end MILP-MPC: the native mixed-integer problem is solved at every step."""

    def __init__(self, task: Task):
        self.task = task

    def reset(self, state: np.ndarray) -> None:
        pass

    def observe(self, state: np.ndarray) -> None:
        pass

    def act(self, state: np.ndarray, goal: np.ndarray) -> Decision:
        problem = self.task.milp(state, goal)
        result = problem.solve(time_limit=self.task.config.time_limit, mip_rel_gap=self.task.config.mip_rel_gap)
        if result.x is None or not (result.success or result.status == 1):   # status 1: time limit, incumbent kept
            raise ControlFailure(f"native MILP: {result.message}")
        x, plan = decode(self.task, result.x)
        return Decision(self.task.action(x), plan, float(result.fun))


@dataclass
class Episode:
    seed: int
    setting: int
    states: list[np.ndarray]
    modes: list[np.ndarray]
    goal: np.ndarray
    actions: list[np.ndarray] = field(default_factory=list)
    plans: list[np.ndarray] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)
    success: bool = False
    failure: str | None = None

    @property
    def steps(self) -> int:
        return len(self.actions)

    def summary(self) -> dict:
        return {"seed": self.seed, "setting": self.setting, "success": self.success, "steps": self.steps,
                "failure": self.failure}


def rollout(task: Task, controller: Controller, seed: int, *, max_steps: int | None = None) -> Episode:
    """Run one closed-loop episode from ``task.reset(seed)``."""
    state, goal = task.reset(seed)
    episode = Episode(seed, task.setting, [state], [task.mode(state)], goal)
    try:
        controller.reset(state)
        for _ in range(max_steps or task.max_steps):
            if task.success(state, goal):
                break
            decision = controller.act(state, goal)
            state = task.step(state, decision.action)
            episode.states.append(state)
            episode.modes.append(task.mode(state))
            episode.actions.append(decision.action)
            episode.plans.append(decision.plan)
            episode.infos.append({"objective": decision.objective, **decision.info})
            controller.observe(state)
    except (ControlFailure, ValueError, RuntimeError, TimeoutError) as error:
        episode.failure = f"{type(error).__name__}: {error}"
    episode.success = episode.failure is None and bool(task.success(state, goal))
    if not episode.success and episode.failure is None:
        episode.failure = "step limit"
    return episode
