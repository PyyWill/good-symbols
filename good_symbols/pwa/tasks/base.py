"""The interface every PWA task implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..milp import MILP

DATA_ROOT = Path(os.environ.get("GOOD_SYMBOLS_DATA") or Path(__file__).resolve().parents[3] / "data")


@dataclass(frozen=True)
class DiscreteInterface:
    """One stage of the discrete blocks of P_PWA (Eq. 3): ``E z = e``, ``P z <= p`` and the coupling rows ``B``."""

    E: np.ndarray
    P: np.ndarray
    B: np.ndarray
    names: tuple[str, ...] = ()


class Task(ABC):
    """A finite-horizon planning problem written as a MILP, plus the system that executes its actions.

    The state is a flat array that also encodes the current mode ``z`` (see :meth:`mode`).  ``milp`` returns
    the native problem at a state with the H x nz mode variables first; ``action`` decodes the first action
    from a solution of that problem.
    """

    name: ClassVar[str]
    nz: ClassVar[int]                         # binary mode variables per stage
    max_steps: ClassVar[int]                  # episode step limit
    settings: ClassVar[tuple[int, ...]] = (1,)
    pins_initial_mode: ClassVar[bool] = True  # whether the MILP fixes z_0 to the observed mode
    Config: ClassVar[type]

    def __init__(self, setting: int = 1, **overrides: Any):
        if setting not in self.settings:
            raise ValueError(f"{self.name} has settings {self.settings}, not {setting}")
        self.setting = setting
        self.config = replace(self.default_config(), **overrides)

    def default_config(self):
        return self.Config()

    @property
    def horizon(self) -> int:
        return int(self.config.horizon)

    @property
    def data_dir(self) -> Path:
        return DATA_ROOT / self.name

    def launch(self, device: str = "cuda:0") -> None:
        """Start the simulator, if the task has one."""

    def close(self) -> None:
        """Shut the simulator down."""

    @abstractmethod
    def reset(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Initial state and goal of the evaluation episode ``seed``."""

    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Execute one action in the task's physics and return the next state."""

    @abstractmethod
    def success(self, state: np.ndarray, goal: np.ndarray) -> bool: ...

    @abstractmethod
    def mode(self, state: np.ndarray) -> np.ndarray:
        """The binary mode ``z`` observed at ``state``."""

    @abstractmethod
    def milp(self, state: np.ndarray, goal: np.ndarray) -> MILP:
        """The native horizon-H planning problem P_PWA at ``state``."""

    @abstractmethod
    def action(self, x: np.ndarray) -> np.ndarray:
        """The first action of a solution ``x`` of :meth:`milp`."""

    @classmethod
    @abstractmethod
    def discrete_interface(cls) -> DiscreteInterface: ...
