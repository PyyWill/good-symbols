"""MPushT: up to four pushers translate and rotate a planar T to a goal pose."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds

from ...milp import MILP, Rows, Variables
from ..base import DiscreteInterface, Task
from . import physics


@dataclass(frozen=True)
class Config:
    horizon: int = 4
    dt: float = 0.1
    max_faces: int = 4                  # pushers: at most this many faces are pushed at once
    time_limit: float = 20.0
    mip_rel_gap: float = 1e-4
    position_weight: float = 1.0
    angle_weight: float = 1.5
    terminal_weight: float = 6.0
    effort_weight: float = 0.01


class MPushT(Task):
    """The mode ``z`` is the commanded set of contact cells; it is observed as the last command."""

    name = "mpusht"
    nz = physics.NZ
    max_steps = 160
    pins_initial_mode = False
    Config = Config
    position_tolerance, angle_tolerance = 0.15, 0.15

    def __init__(self, setting: int = 1, **overrides):
        super().__init__(setting, **overrides)
        self.last_command: np.ndarray | None = None

    def reset(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """A random start and goal pose at least 0.5 apart in position or angle."""
        rng = np.random.default_rng(seed)
        self.last_command = None
        while True:
            start = np.r_[rng.uniform(-1.0, 1.0, 2), rng.uniform(-np.pi, np.pi)]
            goal = np.r_[rng.uniform(-1.0, 1.0, 2), rng.uniform(-np.pi, np.pi)]
            if np.linalg.norm(start[:2] - goal[:2]) >= 0.5 or abs(_wrap(start[2] - goal[2])) >= 0.5:
                return start, goal

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.last_command = np.asarray(action, dtype=float).copy()
        return physics.step(state, self.last_command, self.config.dt)

    def success(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return bool(np.hypot(*(state[:2] - goal[:2])) < self.position_tolerance
                    and abs(_wrap(state[2] - goal[2])) < self.angle_tolerance)

    def mode(self, state: np.ndarray) -> np.ndarray:
        return np.zeros(self.nz) if self.last_command is None else self.last_command.copy()

    def action(self, x: np.ndarray) -> np.ndarray:
        return x[:self.nz].copy()

    @classmethod
    def discrete_interface(cls) -> DiscreteInterface:
        P = np.zeros((len(physics.CONTIGUITY), cls.nz), dtype=int)
        for row, (i, j) in enumerate(physics.CONTIGUITY):
            P[row, j], P[row, i] = 1, -1
        return DiscreteInterface(E=np.zeros((0, cls.nz), dtype=int), P=P, B=-physics.WHAT,
                                 names=tuple(f"cell_{i}" for i in range(cls.nz)))

    def milp(self, state: np.ndarray, goal: np.ndarray) -> MILP:
        """Body-frame goal error dynamics driven by the pushed cells, L1 tracking cost."""
        cfg, H, nz = self.config, self.horizon, self.nz
        v = Variables()
        z, face = v.add(H, nz), v.add(H, physics.N_FACES)     # z: pushed cells; face: face is pushed
        pose = [v.add(H + 1) for _ in range(3)]                # x, y, theta of the error, t = 0..H
        error = [v.add(H) for _ in range(3)]                    # |pose_t|, t = 1..H
        lower, upper, cost = np.full(v.n, -np.inf), np.full(v.n, np.inf), np.zeros(v.n)
        lower[z], upper[z], lower[face], upper[face] = 0.0, 1.0, 0.0, 1.0
        for block in pose:
            lower[block], upper[block] = -40.0, 40.0
        for block in error:
            lower[block] = 0.0

        rows = Rows(v.n)
        for block, value in zip(pose, physics.goal_error(state, goal)):
            rows.eq({block[0]: 1.0}, value)
        G = physics.body_motion_matrix()
        for t in range(H):
            for i, j in physics.CONTIGUITY:
                rows.le({z[t, j]: 1.0, z[t, i]: -1.0}, 0.0)
            for f, cells in enumerate(physics.FACE_FIRST_CELLS):
                for cell in cells:
                    rows.le({z[t, cell]: 1.0, face[t, f]: -1.0}, 0.0)
                rows.le([(face[t, f], 1.0), *((z[t, cell], -1.0) for cell in cells)], 0.0)
            rows.le({face[t, f]: 1.0 for f in range(physics.N_FACES)}, cfg.max_faces)
            for block, gains in zip(pose, G):
                rows.eq([(block[t + 1], 1.0), (block[t], -1.0), *((z[t, i], cfg.dt * g) for i, g in enumerate(gains))],
                        0.0)
        for t in range(1, H + 1):
            weight = cfg.terminal_weight if t == H else 1.0
            for block, err, w in zip(pose, error, (cfg.position_weight, cfg.position_weight, cfg.angle_weight)):
                cost[err[t - 1]] = weight * w
                rows.le({block[t]: 1.0, err[t - 1]: -1.0}, 0.0)
                rows.le({block[t]: -1.0, err[t - 1]: -1.0}, 0.0)
        cost[z] += cfg.effort_weight
        integrality = np.zeros(v.n, dtype=int)
        integrality[z], integrality[face] = 1, 1
        return MILP(cost, integrality, Bounds(lower, upper), rows.constraint())


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
