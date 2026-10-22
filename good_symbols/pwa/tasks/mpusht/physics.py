"""Collaborative Push-T: pushers move a T-shaped body through finitely many contact cells.

All eight faces of the T are pushable.  Contact positions on a face are unary ("thermometer") chains of cells
with unit spacing: the first cell of a chain applies a unit normal force, every further cell adds a unit couple,
so the net body wrench is ``WHAT @ z`` with ``WHAT`` in {0, +-1}.  Faces whose arm range crosses the centroid
axis are split into two chains, one per torque sign.  The body moves quasi-statically, ``V = K_LS * wrench``.
"""
from __future__ import annotations

import numpy as np

BAR_WIDTH, BAR_HEIGHT, STEM_WIDTH, STEM_HEIGHT = 4.0, 1.0, 1.0, 3.0

# (face, inward push normal, contact points from the centroid axis outward)
CHAINS = (
    (0, (0, -1), ((0.0, 1.357), (-1.0, 1.357), (-2.0, 1.357))),   # top bar, positive torque side
    (0, (0, -1), ((1.0, 1.357), (2.0, 1.357))),                    # top bar, negative torque side
    (1, (-1, 0), ((2.0, 1.0),)),                                   # right end of the bar
    (2, (0, 1), ((1.0, 0.357), (2.0, 0.357))),                     # bar underside, right
    (3, (-1, 0), ((0.5, 0.0),)),                                   # right side of the stem
    (3, (-1, 0), ((0.5, -1.0), (0.5, -2.0))),
    (4, (0, 1), ((0.0, -2.643),)),                                 # stem end
    (5, (1, 0), ((-0.5, 0.0), (-0.5, -1.0), (-0.5, -2.0))),        # left side of the stem
    (6, (0, 1), ((-1.0, 0.357), (-2.0, 0.357))),                   # bar underside, left
    (7, (1, 0), ((-2.0, 1.0),)),                                   # left end of the bar
)
N_FACES = 8


def _polygon_moments(polygon: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Centroid, area and squared radius of gyration (about the centroid) of a simple polygon."""
    x, y = polygon.T
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * cross.sum()
    centroid = np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) / (6 * area)
    x, y = (polygon - centroid).T
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    inertia = ((y ** 2 + y * y1 + y1 ** 2) * cross).sum() / 12 + ((x ** 2 + x * x1 + x1 ** 2) * cross).sum() / 12
    return centroid, float(area), float(abs(inertia / area))


TEE = np.array([(-BAR_WIDTH / 2, 0.0), (-STEM_WIDTH / 2, 0.0), (-STEM_WIDTH / 2, -STEM_HEIGHT),
                (STEM_WIDTH / 2, -STEM_HEIGHT), (STEM_WIDTH / 2, 0.0), (BAR_WIDTH / 2, 0.0),
                (BAR_WIDTH / 2, BAR_HEIGHT), (-BAR_WIDTH / 2, BAR_HEIGHT)])
K_LS = np.array([1.0, 1.0, 1.0 / _polygon_moments(TEE)[2]])        # limit-surface gains (translation, rotation)


def _cells():
    wrenches, first_cell, contiguity = [], [], []
    for _, normal, points in CHAINS:
        first_cell.append(len(wrenches))
        previous = np.zeros(3)
        for k, point in enumerate(points):
            cumulative = np.array([normal[0], normal[1], point[0] * normal[1] - point[1] * normal[0]], dtype=float)
            wrenches.append(cumulative - previous)
            if k > 0:
                contiguity.append((len(wrenches) - 2, len(wrenches) - 1))   # z[i+1] <= z[i]
            previous = cumulative
    return np.array(wrenches), tuple(first_cell), tuple(contiguity)


CELL_WRENCH, CHAIN_FIRST_CELL, CONTIGUITY = _cells()
WHAT = np.rint(CELL_WRENCH.T).astype(int)                          # 3 x nz net-wrench incidence
NZ = WHAT.shape[1]
FACE_FIRST_CELLS = tuple(tuple(CHAIN_FIRST_CELL[c] for c, (f, *_) in enumerate(CHAINS) if f == face)
                         for face in range(N_FACES))                # cells whose push marks the face as pushed


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def body_motion_matrix() -> np.ndarray:
    """Body-frame twist per unit of each cell, ``3 x nz``."""
    return K_LS[:, None] * CELL_WRENCH.T


def goal_error(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Goal pose relative to the body frame."""
    position = rotation(state[2]).T @ (goal[:2] - state[:2])
    return np.array([position[0], position[1], (goal[2] - state[2] + np.pi) % (2 * np.pi) - np.pi])


def step(state: np.ndarray, z: np.ndarray, dt: float) -> np.ndarray:
    x, y, theta = state
    twist = K_LS * (CELL_WRENCH.T @ np.asarray(z, dtype=float))
    world = rotation(theta) @ twist[:2]
    return np.array([x + dt * world[0], y + dt * world[1], theta + dt * twist[2]])
