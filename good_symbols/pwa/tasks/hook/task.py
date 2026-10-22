"""Hook: a gripper reaches around a block with an L-shaped hook and pulls it to the wall.

The MILP predicts locally affine rigid-contact motion; the chosen first gripper command is executed in
Pymunk (:mod:`.physics`).  Mode-dependent constraints are big-M rows normalized by ``M`` so that their mode
coefficients are exactly +1: they form the coupling block ``B`` of the discrete interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds

from ...milp import MILP, Rows, Terms, Variables
from ..base import DiscreteInterface, Task
from . import physics as phys

GOAL_Z = phys.mode_vector(held=True, hook_surface="arm_bottom", block_face="top", contact_active=True,
                          wall_swept_contact=True)
# contact variables q: hook coordinate, block coordinate, contact point (x, y), signed gap, impulses (n, t, wall)
Q_DIM = 8
Q_RH, Q_RB, Q_CX, Q_CY, Q_GAP, Q_LN, Q_LT, Q_LW = range(Q_DIM)
ACTION_DIM = 4                       # gripper dx, dy, dtheta, finger-gap change


@dataclass(frozen=True)
class Config:
    horizon: int = 4
    time_limit: float = 8.0
    mip_rel_gap: float = 1e-4
    big_m: float = 12.0
    max_impulse: float = 0.45
    contact_geometry_tolerance: float = 0.010
    contact_gap_tolerance: float = 0.006
    hook_contact_angle_tolerance: float = phys.HOOK_CONTACT_CHART_ANGLE
    block_contact_angle_tolerance: float = phys.BLOCK_CONTACT_CHART_ANGLE
    max_dx: float = phys.MAX_DXY
    max_dy: float = phys.MAX_DXY
    max_dtheta: float = phys.MAX_DTHETA
    max_dgap: float = phys.MAX_DGAP
    stage_goal_weight: float = 0.8
    terminal_goal_weight: float = 18.0
    stage_route_weight: float = 1.0
    terminal_route_weight: float = 10.0
    effort_weight: float = 0.010
    angle_effort_weight: float = 0.008
    gap_effort_weight: float = 0.002
    terminal_mode_weight: float = 2.0


class Hook(Task):
    name = "hook"
    nz = phys.Z_DIM
    max_steps = 60
    Config = Config

    def reset(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return phys.sample_reset(np.random.default_rng(seed))

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return phys.step(state, action)

    def success(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return phys.success(state)[0]

    def mode(self, state: np.ndarray) -> np.ndarray:
        return phys.unpack_observation(state)[1].astype(float)

    def action(self, x: np.ndarray) -> np.ndarray:
        H = self.horizon
        return x[H * phys.Z_DIM + (H + 1) * phys.S_DIM:][:ACTION_DIM].copy()      # u_0

    @classmethod
    def discrete_interface(cls) -> DiscreteInterface:
        """``B`` is read off the implemented coupling rows of the first stage."""
        task = cls(horizon=2)
        rows = task._rows(*phys.sample_reset(np.random.default_rng(0)))[1]
        coupling = [r for r, tag in enumerate(rows.tags) if tag == ("bridge", 0)]
        B = rows.matrix()[coupling][:, :phys.Z_DIM].toarray()
        if not np.allclose(B, np.rint(B)):
            raise AssertionError("coupling rows have non-integral mode coefficients")
        E, _ = phys.local_equalities()
        P, _ = phys.local_inequalities()
        return DiscreteInterface(E=E, P=P, B=np.rint(B).astype(int), names=phys.Z_NAMES)

    def milp(self, state: np.ndarray, goal: np.ndarray) -> MILP:
        return self._rows(state, goal)[0]

    def _rows(self, observation: np.ndarray, goal_state: np.ndarray) -> tuple[MILP, _GatedRows]:
        cfg, H = self.config, self.horizon
        state0, observed_z, valid = phys.unpack_observation(observation)
        goal_state = np.asarray(goal_state, dtype=float)
        if not valid:
            raise ValueError("invalid physical observation")
        if goal_state.shape != (phys.S_DIM,) or not phys.validate_mode(observed_z):
            raise ValueError("invalid goal or observed mode")
        v = Variables()
        z, s, u = v.add(H, phys.Z_DIM), v.add(H + 1, phys.S_DIM), v.add(H, ACTION_DIM)
        q, au, ey, route = v.add(H, Q_DIM), v.add(H, ACTION_DIM), v.add(H + 1), v.add(H + 1)
        c, lb, ub = np.zeros(v.n), np.full(v.n, -np.inf), np.full(v.n, np.inf)
        lb[z], ub[z] = 0.0, 1.0
        for i, (lo, hi) in _state_bounds().items():
            lb[s[:, i]], ub[s[:, i]] = lo, hi
        limits = (cfg.max_dx, cfg.max_dy, cfg.max_dtheta, cfg.max_dgap)
        max_hook_coordinate = max(value[2] for value in HOOK_SURFACE_DATA.values())
        q_bounds = {Q_RH: (-max_hook_coordinate, max_hook_coordinate), Q_RB: (-phys.BLOCK_HALF, phys.BLOCK_HALF),
                    Q_CX: phys.WORLD_X, Q_CY: phys.WORLD_Y, Q_GAP: (0.0, 4.0), Q_LN: (0.0, cfg.max_impulse),
                    Q_LT: (-cfg.max_impulse, cfg.max_impulse), Q_LW: (0.0, cfg.max_impulse)}
        for i, limit in enumerate(limits):
            lb[u[:, i]], ub[u[:, i]] = -limit, limit
            lb[au[:, i]], ub[au[:, i]] = 0.0, limit
        for i, (lo, hi) in q_bounds.items():
            lb[q[:, i]], ub[q[:, i]] = lo, hi
        lb[ey], ub[ey], lb[route], ub[route] = 0.0, 2.0, 0.0, 8.0

        rows = _GatedRows(v.n, cfg.big_m)
        for i, value in enumerate(state0):
            rows.eq({s[0, i]: 1.0}, value)
        for i, value in enumerate(observed_z):
            rows.eq({z[0, i]: 1.0}, value)
        # First-order support function of the rotated block against the wall: y_b >= h (1 + |theta_b|).
        for t in range(H + 1):
            for sign in (-1.0, 1.0):
                rows.ge({s[t, phys.B_POS.start + 1]: 1.0, s[t, phys.B_TH]: sign * phys.BLOCK_HALF},
                        phys.WALL_Y + phys.BLOCK_HALF + 0.002)
        # Binary L1 distance of the last planned mode to the goal mode (up to a constant).
        for i, value in enumerate(GOAL_Z):
            c[z[H - 1, i]] += cfg.terminal_mode_weight * (1.0 - 2.0 * value)

        E, e = phys.local_equalities()
        P, p = phys.local_inequalities()
        for t in range(H):
            for row, rhs in zip(E, e):
                rows.eq({z[t, i]: value for i, value in enumerate(row) if value}, rhs)
            for row, rhs in zip(P, p):
                rows.le({z[t, i]: value for i, value in enumerate(row) if value}, rhs)
            for i in range(ACTION_DIM):                                           # gripper kinematics
                rows.eq([(s[t + 1, i], 1.0), (s[t, i], -1.0), (u[t, i], -1.0)], 0.0)
            # Fixed polyhedral trust region: |u_xy| + (u_max / pi) |theta_h| <= u_max.
            for i, limit in ((0, cfg.max_dx), (1, cfg.max_dy)):
                for action_sign in (-1.0, 1.0):
                    for theta_sign in (-1.0, 1.0):
                        rows.le([(u[t, i], action_sign), (s[t, phys.H_TH], theta_sign * limit / np.pi)], limit)
            _add_stage(rows, cfg, t, z, s, u, q, au, c)

        _add_route_potential(rows, cfg, goal_state, H, z, s, ey, route, c)
        integrality = np.zeros(v.n, dtype=int)
        integrality[z] = 1
        return MILP(c, integrality, Bounds(lb, ub), rows.constraint()), rows


class _GatedRows(Rows):
    """Rows plus big-M constraints ``a x <= rhs`` that hold iff all ``gates`` (mode variables) are one."""

    def __init__(self, n: int, big_m: float):
        super().__init__(n)
        self.big_m = float(big_m)

    def gated_le(self, terms: Terms, rhs: float, gates, stage: int) -> int:
        if len(set(gates)) != len(gates):
            raise ValueError("gates must be distinct")
        row = {gate: 1.0 for gate in gates}
        for column, value in _accumulate(terms).items():
            row[column] = row.get(column, 0.0) + value / self.big_m
        return self.le(row, len(gates) + rhs / self.big_m, tag=("bridge", stage))

    def gated_ge(self, terms: Terms, rhs: float, gates, stage: int) -> int:
        return self.gated_le({j: -value for j, value in _accumulate(terms).items()}, -rhs, gates, stage)

    def gated_eq(self, terms: Terms, rhs: float, gates, stage: int) -> None:
        self.gated_le(terms, rhs, gates, stage)
        self.gated_ge(terms, rhs, gates, stage)


def _accumulate(terms: Terms) -> dict:
    out: dict = {}
    for column, value in (terms.items() if isinstance(terms, dict) else terms):
        out[column] = out.get(column, 0.0) + float(value)
    return out


def _surface_data(segments) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """Body-frame midpoint, unit tangent and half-length of each finite surface."""
    data = {}
    for name, (a, b, *_) in segments.items():
        length = float(np.linalg.norm(b - a))
        data[name] = (0.5 * (a + b), (b - a) / length, 0.5 * length)
    return data


HOOK_SURFACE_DATA = _surface_data(phys._surface_segments_local())
_h = phys.BLOCK_HALF
BLOCK_FACE_DATA = {"left": (np.array([-_h, 0.0]), np.array([0.0, 1.0]), _h),
                   "right": (np.array([_h, 0.0]), np.array([0.0, 1.0]), _h),
                   "bottom": (np.array([0.0, -_h]), np.array([1.0, 0.0]), _h),
                   "top": (np.array([0.0, _h]), np.array([1.0, 0.0]), _h)}
HOOK_NORMALS = {"arm_top": np.array([0.0, 1.0]), "arm_end": np.array([-1.0, 0.0]),
                "arm_bottom": np.array([0.0, -1.0]), "handle_left": np.array([-1.0, 0.0]),
                "handle_end": np.array([0.0, -1.0]), "handle_right": np.array([1.0, 0.0])}


def _state_bounds() -> dict[int, tuple[float, float]]:
    hook_x = (phys.WORLD_X[0] + phys.HOOK_LONG + 0.02, phys.WORLD_X[1] - phys.HOOK_THICKNESS / 2 - 0.02)
    return {
        phys.G_POS.start: hook_x, phys.G_POS.start + 1: (phys.WORLD_Y[0], phys.WALL_Y),
        phys.G_TH: (-np.pi, np.pi), phys.G_GAP: (phys.GRIP_CLOSED, phys.GRIP_OPEN),
        phys.H_POS.start: hook_x, phys.H_POS.start + 1: (phys.WORLD_Y[0], phys.WALL_Y), phys.H_TH: (-np.pi, np.pi),
        phys.H_VX: (-0.80, 0.80), phys.H_VY: (-0.80, 0.80), phys.H_OM: (-1.50, 1.50),
        phys.B_POS.start: (phys.WORLD_X[0] + phys.BLOCK_HALF, phys.WORLD_X[1] - phys.BLOCK_HALF),
        phys.B_POS.start + 1: (phys.WALL_Y + phys.BLOCK_HALF, phys.WORLD_Y[1] - phys.BLOCK_HALF),
        phys.B_TH: (-0.50, 0.50), phys.B_VX: (-0.80, 0.80), phys.B_VY: (-0.80, 0.80), phys.B_OM: (-1.50, 1.50),
    }


def _surface_gates(z, t: int, regime: int, surface: str) -> list[int]:
    return [z[t, regime], *(z[t, bit] for bit in phys.HOOK_SURFACES[surface])]


def _add_stage(rows: _GatedRows, cfg: Config, t: int, z, s, u, q, au, c) -> None:
    """Mode-dependent route regions, grasp, contact and wall constraints of stage ``t``."""
    released, held, separate, active = z[t, 0], z[t, 1], z[t, 2], z[t, 3]
    wall_clear, wall_contact = z[t, 14], z[t, 15]
    hx, hy, bx, by = phys.H_POS.start, phys.H_POS.start + 1, phys.B_POS.start, phys.B_POS.start + 1
    rel_x = {s[t, hx]: 1.0, s[t, bx]: -1.0}
    rel_y = {s[t, hy]: 1.0, s[t, by]: -1.0}

    # While separated, the factors select a polyhedral free-space region of the finite L: the route.
    for surface, sign in (("arm_top", 1.0), ("handle_left", -1.0)):
        gates = _surface_gates(z, t, 2, surface)
        rows.gated_ge({s[t, phys.H_TH]: sign}, phys.ROUTE_CHART_ANGLE, gates, t)
        rows.gated_ge(rel_x, phys.ROTATE_RIGHT_REL_X, gates, t)
        corridor = {**rel_y, s[t, phys.H_TH]: -sign * phys.ROTATE_Y_SLOPE}
        rows.gated_le(corridor, phys.ROUTE_LOW_CHART_REL_Y + phys.ROTATE_Y_BAND, gates, t)
        rows.gated_ge(corridor, phys.ROUTE_LOW_CHART_REL_Y - phys.ROTATE_Y_BAND, gates, t)
        rows.gated_eq({u[t, 0]: 1.0}, 0.0, gates, t)
        rows.gated_eq([(u[t, 1], 1.0), (u[t, 2], -sign * phys.ROTATE_Y_SLOPE)], 0.0, gates, t)
    gates = _surface_gates(z, t, 2, "handle_right")
    rows.gated_ge(rel_x, phys.ROUTE_RIGHT_REL_X, gates, t)
    rows.gated_le(rel_y, phys.ROUTE_LOW_CHART_REL_Y, gates, t)
    gates = _surface_gates(z, t, 2, "arm_end")
    rows.gated_ge(rel_x, phys.ROUTE_RIGHT_REL_X, gates, t)
    rows.gated_ge(rel_y, phys.ROUTE_LOW_CHART_REL_Y, gates, t)
    rows.gated_le(rel_y, phys.ROUTE_CLEAR_REL_Y, gates, t)
    rows.gated_ge({u[t, 0]: 1.0}, 0.0, gates, t)
    gates = _surface_gates(z, t, 2, "handle_end")
    rows.gated_ge(rel_y, phys.ROUTE_CLEAR_REL_Y, gates, t)
    rows.gated_ge(rel_x, phys.ROUTE_ALIGN_SWITCH_X, gates, t)
    gates = _surface_gates(z, t, 2, "arm_bottom")
    rows.gated_ge(rel_y, phys.ROUTE_CLEAR_REL_Y, [*gates, wall_clear], t)
    rows.gated_ge(rel_x, phys.ROUTE_HANDLE_REL_X, gates, t)
    rows.gated_le(rel_x, phys.ROUTE_ALIGN_SWITCH_X, gates, t)
    gates = _surface_gates(z, t, 3, "arm_bottom")                          # stable pulling contact
    rows.gated_le(rel_x, phys.ROUTE_CONTACT_REL_X + 0.16, gates, t)
    rows.gated_ge(rel_x, phys.ROUTE_CONTACT_REL_X - 0.16, gates, t)
    for surface in ("arm_end", "arm_bottom", "handle_end", "handle_right"):
        gates = _surface_gates(z, t, 2, surface)
        rows.gated_le({s[t, phys.H_TH]: 1.0}, phys.ROUTE_CHART_ANGLE, gates, t)
        rows.gated_ge({s[t, phys.H_TH]: 1.0}, -phys.ROUTE_CHART_ANGLE, gates, t)

    # Grasp and release are modes: a held hook moves with the gripper, a released one drifts and damps.
    rows.gated_ge({s[t, phys.G_GAP]: 1.0}, phys.GRIP_OPEN, [released], t)
    rows.gated_eq({s[t, phys.G_GAP]: 1.0}, phys.GRIP_CLOSED, [held], t)
    for hook_index, gripper_index in ((hx, 0), (hy, 1), (phys.H_TH, 2)):
        rows.gated_eq({s[t, hook_index]: 1.0, s[t, gripper_index]: -1.0}, 0.0, [held], t)
        rows.gated_eq({s[t + 1, hook_index]: 1.0, s[t, hook_index]: -1.0, u[t, gripper_index]: -1.0}, 0.0, [held], t)
    for velocity, i in ((phys.H_VX, 0), (phys.H_VY, 1), (phys.H_OM, 2)):
        rows.gated_eq({s[t + 1, velocity]: 1.0, u[t, i]: -1.0 / phys.CONTROL_DT}, 0.0, [held], t)
    for position, velocity in ((hx, phys.H_VX), (hy, phys.H_VY), (phys.H_TH, phys.H_OM)):
        rows.gated_eq({s[t + 1, position]: 1.0, s[t, position]: -1.0, s[t + 1, velocity]: -phys.CONTROL_DT},
                      0.0, [released], t)
        rows.gated_eq({s[t + 1, velocity]: 1.0, s[t, velocity]: -phys.VELOCITY_DAMPING}, 0.0, [released, separate], t)
    for position, velocity in ((bx, phys.B_VX), (by, phys.B_VY), (phys.B_TH, phys.B_OM)):
        rows.eq({s[t + 1, position]: 1.0, s[t, position]: -1.0, s[t + 1, velocity]: -phys.CONTROL_DT}, 0.0)
        wall = [(q[t, Q_LW], -1.0 / phys.BLOCK_MASS)] if velocity == phys.B_VY else []
        rows.gated_eq([(s[t + 1, velocity], 1.0), (s[t, velocity], -phys.VELOCITY_DAMPING), *wall], 0.0, [separate], t)

    # Contact: the selected surface pair is at zero gap; inactive contact variables vanish.
    rows.gated_eq({q[t, Q_GAP]: 1.0}, 0.0, [separate], t)
    for hook_name, bits in phys.HOOK_SURFACES.items():
        p_h, p_b = HOOK_SURFACE_DATA[hook_name][0], BLOCK_FACE_DATA[phys.CONTACT_PAIRS[hook_name]][0]
        normal = HOOK_NORMALS[hook_name]
        # gap = n_h^T (c_b - c_h), linearized once at theta_h = theta_b = 0
        terms = [(q[t, Q_GAP], 1.0), *((s[t, bx + k], -normal[k]) for k in range(2)),
                 *((s[t, hx + k], normal[k]) for k in range(2)),
                 (s[t, phys.B_TH], -float(normal @ _perp(p_b))), (s[t, phys.H_TH], float(normal @ _perp(p_h)))]
        rhs = float(normal @ (p_b - p_h))
        gates = [active, *(z[t, bit] for bit in bits)]
        rows.gated_le(terms, rhs + cfg.contact_gap_tolerance, gates, t)
        rows.gated_ge(terms, rhs - cfg.contact_gap_tolerance, gates, t)
    for i in (Q_RH, Q_RB, Q_CX, Q_CY, Q_LN, Q_LT):
        rows.gated_eq({q[t, i]: 1.0}, 0.0, [separate], t)
    rows.gated_eq({q[t, Q_GAP]: 1.0}, 0.0, [active], t)
    for theta, tolerance in ((phys.H_TH, cfg.hook_contact_angle_tolerance),
                             (phys.B_TH, cfg.block_contact_angle_tolerance)):
        rows.gated_le({s[t, theta]: 1.0}, tolerance, [active], t)
        rows.gated_ge({s[t, theta]: 1.0}, -tolerance, [active], t)
    # The static wall adds a unilateral impulse that cancels the block's normal velocity on impact.
    rows.gated_eq({q[t, Q_LW]: 1.0}, 0.0, [wall_clear], t)
    rows.gated_le({s[t, by]: 1.0}, phys.WALL_Y + phys.BLOCK_HALF + phys.WALL_SWEEP_CENTER_MARGIN, [wall_contact], t)
    rows.gated_eq({s[t + 1, phys.B_VY]: 1.0}, 0.0, [wall_contact], t)

    _add_contact_geometry(rows, cfg, t, z, s, q)
    _add_contact_dynamics(rows, t, z, s, q)
    rows.le({q[t, Q_LT]: 1.0, q[t, Q_LN]: -phys.FRICTION_MU}, 0.0)                   # friction cone
    rows.le({q[t, Q_LT]: -1.0, q[t, Q_LN]: -phys.FRICTION_MU}, 0.0)
    for i in range(ACTION_DIM):
        rows.le({u[t, i]: 1.0, au[t, i]: -1.0}, 0.0)
        rows.le({u[t, i]: -1.0, au[t, i]: -1.0}, 0.0)
        c[au[t, i]] = cfg.angle_effort_weight if i == 2 else cfg.gap_effort_weight if i == 3 else cfg.effort_weight


def _perp(p: np.ndarray) -> np.ndarray:
    return np.array([-p[1], p[0]])


def _add_contact_geometry(rows: _GatedRows, cfg: Config, t: int, z, s, q) -> None:
    """The common contact point lies on the selected hook surface and block face, inside their patches."""
    bodies = ((phys.HOOK_SURFACES, HOOK_SURFACE_DATA, phys.H_POS.start, phys.H_TH, Q_RH),
              (phys.BLOCK_FACES, BLOCK_FACE_DATA, phys.B_POS.start, phys.B_TH, Q_RB))
    for surfaces, data, position, theta, coordinate in bodies:
        for name, bits in surfaces.items():
            gates = [z[t, 3], *(z[t, bit] for bit in bits)]
            p, tangent, half_length = data[name]
            for axis, common in enumerate((Q_CX, Q_CY)):
                # c = position + p + theta J p + rho tangent, linearized at theta = 0
                terms = [(q[t, common], 1.0), (s[t, position + axis], -1.0), (s[t, theta], -_perp(p)[axis]),
                         (q[t, coordinate], -tangent[axis])]
                rows.gated_le(terms, float(p[axis]) + cfg.contact_geometry_tolerance, gates, t)
                rows.gated_ge(terms, float(p[axis]) - cfg.contact_geometry_tolerance, gates, t)
            for bound in (half_length, (1.0 - 2.0 * phys.CONTACT_PATCH_MARGIN) * half_length):
                rows.gated_le({q[t, coordinate]: 1.0}, bound, gates, t)
                rows.gated_ge({q[t, coordinate]: 1.0}, -bound, gates, t)


def _add_contact_dynamics(rows: _GatedRows, t: int, z, s, q) -> None:
    """Impulse exchange through the active hook-block contact (unilateral, sliding allowed)."""
    damp = phys.VELOCITY_DAMPING
    for hook_name, bits in phys.HOOK_SURFACES.items():
        gates = [z[t, 3], *(z[t, bit] for bit in bits)]
        hook_gates = [z[t, 0], *gates]                         # a released hook receives the reaction
        p_h, p_b = HOOK_SURFACE_DATA[hook_name][0], BLOCK_FACE_DATA[phys.CONTACT_PAIRS[hook_name]][0]
        normal = HOOK_NORMALS[hook_name]
        tangent = np.array([-normal[1], normal[0]])
        torque_b = phys.cross2(p_b, normal), phys.cross2(p_b, tangent)
        torque_h = phys.cross2(p_h, -normal), phys.cross2(p_h, -tangent)
        for axis, velocity in enumerate((phys.B_VX, phys.B_VY)):
            wall = [(q[t, Q_LW], -1.0 / phys.BLOCK_MASS)] if axis == 1 else []
            rows.gated_eq([(s[t + 1, velocity], 1.0), (s[t, velocity], -damp),
                           (q[t, Q_LN], -normal[axis] / phys.BLOCK_MASS),
                           (q[t, Q_LT], -tangent[axis] / phys.BLOCK_MASS), *wall], 0.0, gates, t)
        rows.gated_eq([(s[t + 1, phys.B_OM], 1.0), (s[t, phys.B_OM], -damp),
                       (q[t, Q_LN], -torque_b[0] / phys.BLOCK_INERTIA),
                       (q[t, Q_LT], -torque_b[1] / phys.BLOCK_INERTIA)], 0.0, gates, t)
        for axis, velocity in enumerate((phys.H_VX, phys.H_VY)):
            rows.gated_eq([(s[t + 1, velocity], 1.0), (s[t, velocity], -damp),
                           (q[t, Q_LN], normal[axis] / phys.HOOK_MASS),
                           (q[t, Q_LT], tangent[axis] / phys.HOOK_MASS)], 0.0, hook_gates, t)
        rows.gated_eq([(s[t + 1, phys.H_OM], 1.0), (s[t, phys.H_OM], -damp),
                       (q[t, Q_LN], -torque_h[0] / phys.HOOK_INERTIA),
                       (q[t, Q_LT], -torque_h[1] / phys.HOOK_INERTIA)], 0.0, hook_gates, t)
        # Active contact preserves the normal component of the patch velocity (tangential sliding is free).
        rows.gated_eq([(s[t + 1, phys.B_VX], normal[0]), (s[t + 1, phys.B_VY], normal[1]),
                       (s[t + 1, phys.B_OM], float(normal @ _perp(p_b))), (s[t + 1, phys.H_VX], -normal[0]),
                       (s[t + 1, phys.H_VY], -normal[1]), (s[t + 1, phys.H_OM], -float(normal @ _perp(p_h)))],
                      0.0, gates, t)


def _add_route_potential(rows: _GatedRows, cfg: Config, goal_state: np.ndarray, H: int, z, s, ey, route, c) -> None:
    """Block-to-wall distance and a continuous piecewise-affine remaining-path length, against procrastination."""
    hx, hy, bx, by = phys.H_POS.start, phys.H_POS.start + 1, phys.B_POS.start, phys.B_POS.start + 1
    target_y = float(goal_state[by])
    desired_x = phys.ROUTE_CONTACT_REL_X
    contact_drop = phys.ROUTE_CLEAR_REL_Y - (phys.BLOCK_HALF - float(HOOK_SURFACE_DATA["arm_bottom"][0][1]))
    rotation_tail = phys.ROUTE_CLEAR_REL_Y - phys.ROUTE_LOW_CHART_REL_Y - desired_x + contact_drop
    for t in range(H + 1):
        rows.le({s[t, by]: 1.0, ey[t]: -1.0}, target_y)
        rows.le({s[t, by]: -1.0, ey[t]: -1.0}, -target_y)
        c[ey[t]] = cfg.terminal_goal_weight if t == H else cfg.stage_goal_weight
        c[route[t]] = cfg.terminal_route_weight if t == H else cfg.stage_route_weight
        g = min(t, H - 1)
        rel_x = {s[t, hx]: 1.0, s[t, bx]: -1.0}
        minus_rel_x = {s[t, hx]: -1.0, s[t, bx]: 1.0}
        rel_y = {s[t, hy]: 1.0, s[t, by]: -1.0}
        for surface, sign in (("arm_top", 1.0), ("handle_left", -1.0)):
            rows.gated_ge({route[t]: 1.0, s[t, phys.H_TH]: -sign * phys.HOOK_HANDLE, **minus_rel_x},
                          rotation_tail - phys.HOOK_HANDLE * phys.ROUTE_CHART_ANGLE,
                          _surface_gates(z, g, 2, surface), g)
        for surface in ("handle_right", "arm_end"):
            rows.gated_ge({route[t]: 1.0, **rel_y, **minus_rel_x}, phys.ROUTE_CLEAR_REL_Y - desired_x + contact_drop,
                          _surface_gates(z, g, 2, surface), g)
        rows.gated_ge({route[t]: 1.0, **minus_rel_x}, -desired_x + contact_drop,
                      _surface_gates(z, g, 2, "handle_end"), g)
        gates = _surface_gates(z, g, 2, "arm_bottom")
        rows.gated_ge({route[t]: 1.0, **minus_rel_x}, -desired_x + contact_drop, gates, g)
        rows.gated_ge({route[t]: 1.0, **rel_x}, desired_x + contact_drop, gates, g)
        gates = _surface_gates(z, g, 3, "arm_bottom")
        rows.gated_ge({route[t]: 1.0, **minus_rel_x}, -desired_x, gates, g)
        rows.gated_ge({route[t]: 1.0, **rel_x}, desired_x, gates, g)
