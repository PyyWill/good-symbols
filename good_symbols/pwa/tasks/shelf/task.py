"""Shelf (Packing): stack 4-8 heterogeneous rectangles into two stable stacks inside a one-sided shelf.

The robot is abstracted to the contact patch of the grasped object.  Each stage selects one primitive (grasp an
object, or place it on the floor or on another object); continuous proxies route the selected source, destination
and their sizes, and big-M rows enforce support, release and the packing order through the single opening.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds

from ...milp import MILP, Rows, Variables
from ..base import DiscreteInterface, Task
from . import physics as phys

SOURCE_COLUMNS = {o: [j for j, p in enumerate(phys.PRIMITIVES) if p[1] == o] for o in phys.OBJECTS}
DESTINATION_COLUMNS = {d: [j for j, p in enumerate(phys.PRIMITIVES) if p[2] == d] for d in phys.DESTINATIONS}
GRASP_COLUMNS = DESTINATION_COLUMNS["gripper"]
PLACE_COLUMNS = [j for j in range(phys.Z_DIM) if j not in GRASP_COLUMNS]


@dataclass(frozen=True)
class Config:
    horizon: int = 3
    time_limit: float = 8.0
    mip_rel_gap: float = 1e-5
    big_m: float = 30.0
    place_position_tol: float = 0.20
    place_angle_tol: float = 0.10
    support_ready_tol: float = 0.06
    approach_clearance: float = 0.07
    approach_slope: float = 0.075
    transport_slope: float = 0.05
    terminal_weight: float = 9.0
    stage_weight: float = 0.35
    angle_weight: float = 0.55
    effort_weight: float = 0.020
    grasp_cost: float = 0.060
    goal_mode_weight: float = 0.004


class Shelf(Task):
    name = "shelf"
    nz = phys.Z_DIM
    max_steps = 80
    Config = Config

    def reset(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return phys.sample_reset(np.random.default_rng(seed))

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return phys.step(state, action)

    def success(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return phys.success(state, goal)[0]

    def mode(self, state: np.ndarray) -> np.ndarray:
        return phys.unpack_observation(state)[1].astype(float)

    def action(self, x: np.ndarray) -> np.ndarray:
        """The first motion ``u_0`` and the primitive commanded next."""
        H, nz = self.horizon, self.nz
        u0 = x[H * nz + (H + 1) * phys.S_DIM:][:3]
        return np.r_[u0, x[nz:2 * nz] if H > 1 else x[:nz]]

    @classmethod
    def discrete_interface(cls) -> DiscreteInterface:
        P, E, B = phys.discrete_matrices()
        return DiscreteInterface(E=E, P=P, B=B, names=tuple(name for name, _, _ in phys.PRIMITIVES))

    def milp(self, observation: np.ndarray, goal: np.ndarray) -> MILP:
        state, current_z, active, valid = phys.unpack_observation(observation)
        goal_state, goal_z = phys.unpack_goal(goal)
        if not valid:
            raise ValueError("cannot plan from an invalid physical state")
        cfg, H, M, N = self.config, self.horizon, self.config.big_m, phys.N_OBJECTS
        v = Variables()
        z, s, u, du = v.add(H, phys.Z_DIM), v.add(H + 1, phys.S_DIM), v.add(H, 3), v.add(H, N, 3)
        src, dst, tgt, q = v.add(H, 6), v.add(H, 6), v.add(H, 3), v.add(H, 8)     # source/destination/target proxies
        size, err, effort = v.add(H, 4), v.add(H, N, 3), v.add(H, 3)               # err[t] is the error at t + 1
        mode_err = v.add(len(phys.OBJECTS) + len(phys.DESTINATIONS))
        c, lb, ub = np.zeros(v.n), np.full(v.n, -np.inf), np.full(v.n, np.inf)
        lb[z], ub[z] = 0.0, 1.0
        for i in range(N):
            lb[s[:, 6 * i]], ub[s[:, 6 * i]] = phys.WORLD_X
            lb[s[:, 6 * i + 1]], ub[s[:, 6 * i + 1]] = phys.WORLD_Y
            lb[s[:, 6 * i + 2]], ub[s[:, 6 * i + 2]] = -np.pi, np.pi
            lb[s[:, 6 * i + 3:6 * i + 6]], ub[s[:, 6 * i + 3:6 * i + 6]] = -20.0, 20.0
        lb[s[:, phys.GRASP_FRAME]], ub[s[:, phys.GRASP_FRAME]] = -24.0, 24.0
        limits = np.array((phys.MAX_DX, phys.MAX_DY, phys.MAX_DTHETA))
        for d, limit in enumerate(limits):
            lb[u[:, d]], ub[u[:, d]] = -limit, limit
            lb[effort[:, d]], ub[effort[:, d]] = 0.0, limit
            lb[du[:, :, d]], ub[du[:, :, d]] = -limit, limit
        lb[src], ub[src], lb[dst], ub[dst], lb[tgt], ub[tgt] = -24.0, 24.0, -24.0, 24.0, -12.0, 12.0
        lb[q], ub[q] = -12.0, 12.0
        lb[q[:, :2]], ub[q[:, :2]], lb[q[:, 5]] = -1.0, 1.0, 0.0
        lb[size], ub[size], lb[err] = 0.0, 1.0, 0.0
        lb[mode_err], ub[mode_err] = 0.0, 2.0

        rows = Rows(v.n)

        def gated(terms: dict, limit: float, t: int, columns, active_when_one: bool, offset: float = 0.0) -> None:
            """``|terms . x - offset| <= limit`` whenever the primitives in ``columns`` are selected
            (``active_when_one``) or not selected (otherwise)."""
            sign_m = M if active_when_one else -M
            for sign in (1.0, -1.0):
                rows.le([*((j, sign * a) for j, a in terms.items()), *((z[t, j], sign_m) for j in columns)],
                        (limit + M if active_when_one else limit) + sign * offset)

        P, E, B = phys.discrete_matrices()
        p = phys.availability_rhs(active)
        for d, value in enumerate(state):
            rows.eq({s[0, d]: 1.0}, value)
        for j, value in enumerate(current_z):
            rows.eq({z[0, j]: 1.0}, value)
        for t in range(H):
            rows.eq({z[t, j]: a for j, a in enumerate(E[0])}, 1)
            for row, rhs in zip(P, p):
                rows.le({z[t, j]: a for j, a in enumerate(row)}, rhs)
            # A motion exists only in a grasp mode and moves exactly the selected source object.
            for d, limit in enumerate(limits):
                for sign in (1.0, -1.0):
                    rows.le([(u[t, d], sign), *((z[t, j], -limit) for j in GRASP_COLUMNS)], 0.0)
                rows.eq([(u[t, d], -1.0), *((du[t, i, d], 1.0) for i in range(N))], 0.0)
                for i, obj in enumerate(phys.OBJECTS):
                    for sign in (1.0, -1.0):
                        rows.le([(du[t, i, d], sign), *((z[t, j], -limit) for j in SOURCE_COLUMNS[obj])], 0.0)
            for i in range(N):
                for d in range(3):
                    rows.eq({s[t + 1, 6 * i + d]: 1.0, s[t, 6 * i + d]: -1.0, du[t, i, d]: -1.0}, 0.0)
                    rows.eq({s[t + 1, 6 * i + 3 + d]: 1.0}, 0.0)
            grasp = phys.GRASP_FRAME.start
            for d in range(3):
                rows.eq({s[t + 1, grasp + d]: 1.0, src[t, d]: -1.0, u[t, d]: -1.0}, 0.0)
                gated({s[t, grasp + d]: 1.0, src[t, d]: -1.0}, 0.0, t, PLACE_COLUMNS, True)
                rows.eq({s[t + 1, grasp + 3 + d]: 1.0}, 0.0)
            # Source, target and size multiplexers.
            for i, obj in enumerate(phys.OBJECTS):
                columns = SOURCE_COLUMNS[obj]
                for d in range(6):
                    gated({src[t, d]: 1.0, s[t, 6 * i + d]: -1.0}, 0.0, t, columns, True)
                for d in range(3):
                    gated({tgt[t, d]: 1.0}, 0.0, t, columns, True, goal_state[6 * i + d])
                for k, value in enumerate(phys.PLACEMENT_DIMS[obj]):
                    gated({size[t, k]: 1.0}, 0.0, t, columns, True, value / 2)
            # Destination proxies and sizes; the floor has a continuous horizontal support coordinate.
            for destination, columns in DESTINATION_COLUMNS.items():
                if destination == "gripper":
                    for d in range(6):
                        gated({dst[t, d]: 1.0, src[t, d]: -1.0}, 0.0, t, columns, True)
                    for k in range(2):
                        gated({size[t, 2 + k]: 1.0, size[t, k]: -1.0}, 0.0, t, columns, True)
                    continue
                if destination == "shelf_floor":
                    gated({dst[t, 0]: 1.0, src[t, 0]: -1.0}, 0.0, t, columns, True)
                    for d, value in enumerate((phys.SHELF_SUPPORT_Y, 0.0, 0.0, 0.0, 0.0), start=1):
                        gated({dst[t, d]: 1.0}, 0.0, t, columns, True, value)
                    dims = (phys.SHELF_OPEN_X - phys.SHELF_BACK_X, 0.0)
                else:
                    support = phys.OBJECTS.index(destination)
                    for d in range(6):
                        gated({dst[t, d]: 1.0, s[t, 6 * support + d]: -1.0}, 0.0, t, columns, True)
                    dims = phys.PLACEMENT_DIMS[destination]
                for k, value in enumerate(dims):
                    gated({size[t, 2 + k]: 1.0}, 0.0, t, columns, True, value / 2)
            # Contact variables q = [rho_s, rho_d, cx, cy, gap, lambda_n, lambda_t, tau].
            for rho, center in ((0, src[t, 0]), (1, dst[t, 0])):
                rows.eq({q[t, 2]: 1.0, center: -1.0, q[t, rho]: -0.85}, 0.0)
            for center, half_width in ((src[t, 0], size[t, 0]), (dst[t, 0], size[t, 2])):
                for sign in (1.0, -1.0):
                    rows.le({q[t, 2]: sign, center: -sign, half_width: -1.0}, 0.0)
            for center in (src[t, 1], dst[t, 1]):
                gated({q[t, 3]: 1.0, center: -1.0}, 0.0, t, GRASP_COLUMNS, True)
            gated({q[t, 3]: 1.0, src[t, 1]: -1.0, size[t, 1]: 1.0, q[t, 4]: 1.0}, 0.0, t, GRASP_COLUMNS, False)
            gated({q[t, 3]: 1.0, dst[t, 1]: -1.0, size[t, 3]: -1.0}, 0.0, t, GRASP_COLUMNS, False)
            gated({q[t, 4]: 1.0, src[t, 1]: -1.0, size[t, 1]: 1.0, dst[t, 1]: 1.0, size[t, 3]: 1.0}, 0.0, t,
                  GRASP_COLUMNS, False)
            gated({q[t, 4]: 1.0}, 0.0, t, GRASP_COLUMNS, True)
            gated({q[t, 4]: 1.0}, phys.RELEASE_GAP, t, GRASP_COLUMNS, False)
            # A place primitive is feasible only at the selected source's target.
            for d, limit in enumerate((cfg.place_position_tol, phys.RELEASE_GAP + 0.015, cfg.place_angle_tol)):
                gated({src[t, d]: 1.0, tgt[t, d]: -1.0}, limit, t, GRASP_COLUMNS, False)
            # Single-side entry corridor: approaching its final x, a source rises to its support height.
            for i, obj in enumerate(phys.OBJECTS):
                gx, gy = goal_state[6 * i:6 * i + 2]
                half_width, half_height = 0.5 * phys.PLACEMENT_DIMS[obj][0], 0.5 * phys.PLACEMENT_DIMS[obj][1]
                switch = [*((z[t, j], M) for j in SOURCE_COLUMNS[obj]), *((z[t, j], M) for j in GRASP_COLUMNS)]
                common = [(err[t, i, 2], half_width), (effort[t, 2], half_width), *switch]
                rows.le([(s[t + 1, 6 * i + 1], -1.0), (s[t + 1, 6 * i], -cfg.approach_slope), *common],
                        2.0 * M - (gy + cfg.approach_clearance + cfg.approach_slope * gx))
                rows.le([(s[t + 1, 6 * i + 1], -1.0), (s[t + 1, 6 * i], cfg.transport_slope), *common],
                        2.0 * M - (phys.SHELF_SUPPORT_Y + half_height) + cfg.transport_slope * gx - 0.02)
            # An object is a valid support only at its goal pose, and the source footprint must fit on it.
            for destination in phys.OBJECTS:
                support = phys.OBJECTS.index(destination)
                columns = DESTINATION_COLUMNS[destination]
                for d in range(3):
                    gated({s[t, 6 * support + d]: 1.0}, cfg.place_angle_tol if d == 2 else cfg.support_ready_tol,
                          t, columns, True, goal_state[6 * support + d])
                rows.le([(size[t, 0], 1.0), (size[t, 2], -1.0), *((z[t, j], M) for j in columns)],
                        M - phys.SUPPORT_MARGIN)
            # A source may pass the entry only after every object ahead of it in the packing order is in place.
            for order, source in enumerate(phys.PACKING_ORDER):
                for predecessor in phys.PACKING_ORDER[:order]:
                    k = phys.OBJECTS.index(predecessor)
                    for d in range(3):
                        gated({s[t, 6 * k + d]: 1.0}, cfg.place_angle_tol if d == 2 else cfg.support_ready_tol,
                              t, SOURCE_COLUMNS[source], True, goal_state[6 * k + d])
            # Bounded contact wrench in a linearized friction cone.
            rows.le({q[t, 6]: 1.0, q[t, 5]: -phys.FRICTION}, 0.0)
            rows.le({q[t, 6]: -1.0, q[t, 5]: -phys.FRICTION}, 0.0)
            rows.eq({q[t, 6]: 1.0, u[t, 0]: -1.0}, 0.0)
            rows.eq({q[t, 7]: 1.0, u[t, 2]: -1.0}, 0.0)
            for sign in (1.0, -1.0):
                rows.le({u[t, 1]: sign, q[t, 5]: -1.0}, 0.0)
            c[q[t, 5]] = 1e-4
            for d in range(3):
                rows.le({u[t, d]: 1.0, effort[t, d]: -1.0}, 0.0)
                rows.le({u[t, d]: -1.0, effort[t, d]: -1.0}, 0.0)
                c[effort[t, d]] = cfg.effort_weight
            for i in range(N):
                for d in range(3):
                    for sign in (1.0, -1.0):
                        rows.le({s[t + 1, 6 * i + d]: sign, err[t, i, d]: -1.0}, sign * goal_state[6 * i + d])
                    weight = cfg.terminal_weight if t == H - 1 else cfg.stage_weight
                    c[err[t, i, d]] = weight * (cfg.angle_weight if d == 2 else 1.0)
            c[z[t, GRASP_COLUMNS]] += cfg.grasp_cost
        # Soft terminal target on the physical interface B z (no primitive-identity rows).
        for k, target in enumerate(B @ goal_z):
            for sign in (1.0, -1.0):
                rows.le([*((z[H - 1, j], sign * a) for j, a in enumerate(B[k])), (mode_err[k], -1.0)], sign * target)
            c[mode_err[k]] = cfg.goal_mode_weight
        integrality = np.zeros(v.n, dtype=int)
        integrality[z] = 1
        return MILP(c, integrality, Bounds(lb, ub), rows.constraint())
