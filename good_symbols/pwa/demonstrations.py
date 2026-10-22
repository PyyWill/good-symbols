"""Demonstrations of the end-to-end MILP controller, stored as ``.npz`` files.

A file holds a JSON ``metadata`` list and, for episode ``i``, the arrays ``{i}_states`` (T+1), ``{i}_z``
(observed modes, T+1), ``{i}_actions`` (T), ``{i}_plans`` (the controller's H-step mode plan at every step,
T x H x nz) and ``{i}_goal``.  Only modes, plans and states are used to build symbol graphs and plan
libraries; no demonstrated action is ever executed.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

FIELDS = ("states", "z", "actions", "plans", "goal")


@dataclass
class Demonstration:
    states: np.ndarray
    modes: np.ndarray
    actions: np.ndarray
    plans: np.ndarray
    goal: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.metadata.get("success", True))

    @property
    def setting(self) -> int:
        return int(self.metadata.get("setting", 1))


def load_demonstrations(paths: Iterable[Path | str]) -> list[Demonstration]:
    """Concatenate demonstration files in the given order."""
    demonstrations = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for i, metadata in enumerate(json.loads(str(data["metadata"]))):
                states, modes, actions, plans, goal = (data[f"{i}_{key}"].copy() for key in FIELDS)
                demonstrations.append(Demonstration(states, modes, actions, plans, goal, metadata))
    return demonstrations


def save_demonstrations(path: Path | str, demonstrations: Sequence[Demonstration]) -> None:
    arrays = {f"{i}_{key}": value for i, demo in enumerate(demonstrations)
              for key, value in zip(FIELDS, (demo.states, demo.modes, demo.actions, demo.plans, demo.goal))}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=np.array(json.dumps([demo.metadata for demo in demonstrations])), **arrays)
