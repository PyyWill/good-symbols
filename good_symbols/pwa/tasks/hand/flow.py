"""The finger-face contact flow shared by Block and Cuboid.

Three units of flow leave the source ``s`` through fingers (or directly to the sink, an idle regrasp slot),
reach object faces through the reachable finger-face arcs and end at the sink ``t``.  The 20 binary edge
variables form the mode ``z``: ``E z = e`` is flow conservation, ``P z <= p`` requires an opposition digit and
a palm-side or proximal support, and ``B = I`` since every edge gates part of the contact dynamics.
"""
from __future__ import annotations

import numpy as np

from ..base import DiscreteInterface

EDGES = (("s", "ff"), ("s", "mf"), ("s", "rf"), ("s", "lf"), ("s", "th"),
         ("th", "palm"), ("th", "distal"), ("th", "thumb_side"), ("ff", "little_side"), ("ff", "thumb_side"),
         ("mf", "dorsal"), ("rf", "dorsal"), ("lf", "proximal"),
         ("palm", "t"), ("dorsal", "t"), ("thumb_side", "t"), ("little_side", "t"), ("distal", "t"),
         ("proximal", "t"), ("s", "t"))
NODES = ("s", "ff", "mf", "rf", "lf", "th", "palm", "dorsal", "thumb_side", "little_side", "distal", "proximal")
Z_NAMES = tuple(f"{tail}_to_{head}" for tail, head in EDGES)
NZ = len(EDGES)
ARC_FINGER = np.array([4, 4, 4, 0, 0, 1, 2, 3])           # finger-face arcs are edges 5..12
ARC_FACE = np.array([0, 4, 2, 3, 2, 1, 1, 5])
FACE_NORMALS = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
OCCUPANCY, IDLE = 13, 19                                   # first face-to-sink edge, the s -> t edge


def interface() -> tuple[DiscreteInterface, np.ndarray, np.ndarray]:
    """The discrete interface and the right-hand sides ``e`` and ``p``."""
    E = np.zeros((len(NODES), NZ), dtype=int)
    for column, (tail, head) in enumerate(EDGES):
        if tail in NODES:
            E[NODES.index(tail), column] += 1
        if head in NODES:
            E[NODES.index(head), column] -= 1
    e = np.zeros(len(NODES), dtype=int)
    e[0] = 3
    P = np.zeros((2, NZ), dtype=int)
    P[0, [0, 4]] = -1                                      # thumb or index finger in opposition
    P[1, [13, 18]] = -1                                    # palm or proximal face supported
    return DiscreteInterface(E, P, np.eye(NZ, dtype=int), Z_NAMES), e, -np.ones(2, dtype=int)


def face_distance(point: np.ndarray, face: int, half_extents: np.ndarray) -> float:
    """Distance from a body-frame point to a finite face of a box."""
    axis = int(np.flatnonzero(FACE_NORMALS[face])[0])
    normal_distance = abs(float(point[axis]) - FACE_NORMALS[face][axis] * half_extents[axis])
    tangent = [i for i in range(3) if i != axis]
    excess = np.maximum(np.abs(point[tangent]) - half_extents[tangent], 0.0)
    return float(np.hypot(normal_distance, np.linalg.norm(excess)))


class FlowMapper:
    """Map measured fingertip geometry and contact forces to the closest feasible flow."""

    def __init__(self, flows: np.ndarray, half_extents):
        self.flows = np.asarray(flows, dtype=int)
        self.half_extents = np.asarray(half_extents, dtype=float)

    def __call__(self, tips: np.ndarray, forces: np.ndarray, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        body = (rotation.T @ (np.asarray(tips) - position).T).T
        gaps = np.array([face_distance(body[f], face, self.half_extents) for f, face in zip(ARC_FINGER, ARC_FACE)])
        force = np.linalg.norm(np.asarray(forces), axis=1)
        costs = [sum(gaps[a] / 0.02 - 0.4 * min(force[ARC_FINGER[a]], 2.0) for a in np.flatnonzero(flow[5:13]))
                 + 0.3 * flow[IDLE] for flow in self.flows]
        return self.flows[int(np.argmin(costs))].copy()
