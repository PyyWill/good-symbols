"""Pen geometry and the role-zone contact flow.

Two units of flow leave the source through finger roles (opposition: th+ff, medial: mf+rf, ulnar: lf), reach
axial zones of the capsule (proximal, center, distal) and end at the sink: 12 binary edges form the mode ``z``.
"""
from __future__ import annotations

import numpy as np

from ..base import DiscreteInterface

FINGERS = ("ff", "mf", "rf", "lf", "th")
ROLES = {"opposition": ("th", "ff"), "medial": ("mf", "rf"), "ulnar": ("lf",)}
ROLE_MEMBERS = tuple(tuple(FINGERS.index(f) for f in members) for members in ROLES.values())
ZONES = ("proximal", "center", "distal")
EDGES = (("s", "opposition"), ("s", "medial"), ("s", "ulnar"),
         ("opposition", "center"), ("opposition", "distal"), ("medial", "proximal"), ("medial", "center"),
         ("ulnar", "distal"), ("ulnar", "proximal"), ("proximal", "t"), ("center", "t"), ("distal", "t"))
Z_NAMES = tuple(f"{tail}_to_{head}" for tail, head in EDGES)
NZ = len(EDGES)
ARC_ROLE = np.array([0, 0, 1, 1, 2, 2])             # role-zone arcs are edges 3..8
ARC_ZONE = np.array([1, 2, 0, 1, 2, 0])

RADIUS, LENGTH = 0.008, 0.200
HALF_LENGTH = LENGTH / 2.0
SEGMENT_HALF = HALF_LENGTH - RADIUS
FINGER_CAPSULE_RADIUS = np.array([0.00705, 0.00705, 0.00705, 0.00705, 0.00918])
FINGER_CAPSULE_HALF = np.array([0.012, 0.012, 0.012, 0.012, 0.013])
FINGER_CAPSULE_CENTER = FINGER_CAPSULE_HALF.copy()
ZONE_BOUNDS = np.array([[-HALF_LENGTH, -HALF_LENGTH / 3.0], [-HALF_LENGTH / 3.0, HALF_LENGTH / 3.0],
                        [HALF_LENGTH / 3.0, HALF_LENGTH]])


def interface() -> tuple[DiscreteInterface, np.ndarray, np.ndarray]:
    """The discrete interface and the right-hand sides ``e`` and ``p``.

    ``B`` lists the mode coefficients of the coupling rows as implemented: role rows (one per member finger),
    force and axial rows per role-zone arc, and twist-support rows per zone.
    """
    nodes = ("s", *ROLES, *ZONES)
    E = np.zeros((len(nodes), NZ), dtype=int)
    for column, (tail, head) in enumerate(EDGES):
        if tail in nodes:
            E[nodes.index(tail), column] += 1
        if head in nodes:
            E[nodes.index(head), column] -= 1
    e = np.zeros(len(nodes), dtype=int)
    e[0] = 2
    P = np.zeros((1, NZ), dtype=int)
    P[0, [0, 10]] = -1                              # the opposition role or the center zone is used
    unit = np.eye(NZ, dtype=int)
    B = [unit[role] for role, members in enumerate(ROLE_MEMBERS) for _ in members]
    for arc in range(3, 9):
        B += [-unit[arc]] * 6 + [unit[arc]] * 2
    for zone in range(9, 12):
        B += [-unit[zone]] * 12
    return DiscreteInterface(E, P, np.stack(B), Z_NAMES), e, np.array([-1])


def segment_closest_points(a0, a1, b0, b1) -> tuple[np.ndarray, np.ndarray]:
    """Closest points on two finite segments."""
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc, dd, ee = float(u @ u), float(u @ v), float(v @ v), float(u @ w), float(v @ w)
    denominator = aa * cc - bb * bb
    s = 0.0 if denominator < 1e-12 else np.clip((bb * ee - cc * dd) / denominator, 0.0, 1.0)
    t = np.clip((bb * s + ee) / cc, 0.0, 1.0)
    s = np.clip((bb * t - dd) / aa, 0.0, 1.0)
    t = np.clip((bb * s + ee) / cc, 0.0, 1.0)
    return a0 + s * u, b0 + t * v


def zone_distance(axial: float, zone: int) -> float:
    lower, upper = ZONE_BOUNDS[zone]
    return float(max(lower - axial, 0.0, axial - upper))


def contact_features(tips, position, rotation, tip_rotations) -> tuple[np.ndarray, np.ndarray]:
    """Signed fingertip-capsule gaps (5) and the mean axial contact coordinate of each role (3)."""
    position, rotation = np.asarray(position), np.asarray(rotation)
    pen_a = position + rotation @ np.array([0.0, 0.0, -SEGMENT_HALF])
    pen_b = position + rotation @ np.array([0.0, 0.0, SEGMENT_HALF])
    gaps, axial = [], []
    for finger, (tip, tip_rotation) in enumerate(zip(tips, tip_rotations)):
        axis = np.asarray(tip_rotation)[:, 2]
        center, half = tip + FINGER_CAPSULE_CENTER[finger] * axis, FINGER_CAPSULE_HALF[finger] * axis
        point_finger, point_pen = segment_closest_points(center - half, center + half, pen_a, pen_b)
        gaps.append(float(np.linalg.norm(point_finger - point_pen)) - FINGER_CAPSULE_RADIUS[finger] - RADIUS)
        axial.append(float((rotation.T @ (point_pen - position))[2]))
    gaps, axial = np.asarray(gaps), np.asarray(axial)
    return gaps, np.array([np.mean(axial[list(members)]) for members in ROLE_MEMBERS])


class FlowMapper:
    """Map measured capsule contacts and forces to the closest feasible role-zone flow."""

    def __init__(self, flows: np.ndarray):
        self.flows = np.asarray(flows, dtype=int)

    def __call__(self, tips, forces, position, rotation, tip_rotations) -> np.ndarray:
        gaps, role_axial = contact_features(tips, position, rotation, tip_rotations)
        force = np.linalg.norm(np.asarray(forces), axis=1)
        role_force = np.array([force[list(m)].mean() for m in ROLE_MEMBERS])
        role_gap = np.array([max(abs(gaps[list(m)])) for m in ROLE_MEMBERS])
        costs = [sum(role_gap[ARC_ROLE[a]] / 0.003 + zone_distance(role_axial[ARC_ROLE[a]], ARC_ZONE[a]) / 0.025
                     - 0.15 * min(role_force[ARC_ROLE[a]], 3.0) for a in np.flatnonzero(flow[3:9]))
                 for flow in self.flows]
        return self.flows[int(np.argmin(costs))].copy()
