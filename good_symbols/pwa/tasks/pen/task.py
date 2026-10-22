"""Pen: the Shadow Hand reorients a 200 mm capsule; contacts are organized by finger roles and axial zones.

State ``x``: joint positions and velocities, pen position, rotation error and twist, fingertip positions, the
commanded joint increments ``u`` and the latent per-arc contact forces and per-zone twist contributions (the
latter three are decisions, not measurements).  Per stage the MILP has an affine joint-servo response, the pen
twist as the sum of zone contributions, an affine fingertip model, role contact bands, arc-gated force maps and
axial intervals, zone-gated twists and three role-pair grasp Jacobians selected by the flow.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.optimize import Bounds

from ...milp import Rows, Variables
from ..hand.task import HandTask, Template
from . import geometry
from .env import CONTACT_FORCE, ERROR, NX, OBJ_OMEGA, OBJ_POS, OBJ_VEL, Q, QD, TIP_POS, U, ZONE_TWIST, PenEnv

NS = U.start                      # measured part of x; the final stage keeps only these entries
CONTROL_HZ = 60.0
ROLE_PAIRS = np.array([(1, 1, 0), (1, 0, 1), (0, 1, 1)])


@dataclass(frozen=True)
class Config:
    horizon: int = 2
    time_limit: float = 0.5
    mip_rel_gap: float = 1e-4
    terminal_weight: float = 50.0
    stage_weight: float = 0.5
    effort_weight: float = 0.01
    terminal_flow_weight: float = 0.05
    terminal_contact_weight: float = 50.0
    contact_big_m: float = 0.20
    axial_big_m: float = 0.25
    force_match_big_m: float = 5000.0
    twist_match_big_m: float = 1.0


class Pen(HandTask):
    name = "pen"
    nz = geometry.NZ
    max_steps = 250
    NX, U, ERROR = NX, U, ERROR
    Config = Config
    Env = PenEnv
    PERTURBATION = {1: (0.0, 0.0054), 2: (0.0, 0.0045), 3: (0.0, 0.0059)}
    model_config_fields = frozenset(f.name for f in fields(Config))

    @classmethod
    def discrete_interface(cls):
        return geometry.interface()[0]

    def goal(self) -> np.ndarray:
        return np.asarray(self.model["goal_z"], dtype=float)

    def goal_cost(self, cost: np.ndarray, goal: np.ndarray) -> None:
        """Terminal flow distance plus the terminal gaps of the fingers in the goal's roles."""
        super().goal_cost(cost, goal)
        fingers = {f for role in np.flatnonzero(np.asarray(goal)[:3]) for f in geometry.ROLE_MEMBERS[role]}
        for finger, column in enumerate(self.goal_gaps):
            if finger in fingers:
                cost[column] = self.config.terminal_contact_weight

    def build_template(self) -> Template:
        cfg, m, H, NZ = self.config, self.model, self.horizon, self.nz
        actuated = m["actuated_ids"].astype(int)
        passive = np.setdiff1d(np.arange(24), actuated)
        v = Variables()
        z, xs = v.add(H, NZ), v.add(H * NX + NS)
        error, effort, self.goal_gaps = v.add(H + 1, 3), v.add(H, 20), v.add(5)
        X = lambda t, j: xs[t * NX + j]

        def state_features(t):
            return ([X(t, Q.start + j) for j in actuated] + [X(t, QD.start + j) for j in actuated]
                    + [X(t, ERROR.start + a) for a in range(3)])

        def geometry_features(t):
            return [X(t, Q.start + j) for j in actuated] + [X(t, ERROR.start + a) for a in range(3)]

        interface, e, p = geometry.interface()
        gap_tolerance, axial_tolerance = float(m["contact_gap_tolerance"]), float(m["axial_tolerance"])
        rows = Rows(v.n)
        state_columns = np.arange(NS)                        # u and the latent forces and twists stay free
        state_rows = np.array([rows.eq({X(0, j): 1.0}, 0.0) for j in state_columns])
        mode_rows = np.array([rows.eq({z[0, j]: 1.0}, 0.0) for j in range(NZ)])
        for t in range(H):
            for row, rhs in zip(interface.E, e):
                rows.eq({z[t, j]: row[j] for j in np.flatnonzero(row)}, rhs)
            for row, rhs in zip(interface.P, p):
                rows.le({z[t, j]: row[j] for j in np.flatnonzero(row)}, rhs)
            # Affine response of the tendon-coupled joint-position servo to the commanded increments.
            servo_columns = [X(t, Q.start + j) for j in actuated] + [X(t, QD.start + j) for j in actuated]
            for k, joint in enumerate(actuated):
                rows.eq([(X(t + 1, Q.start + joint), 1.0), (X(t, Q.start + joint), -1.0),
                         *((col, -a) for col, a in zip(servo_columns, m["joint_state_K"][k])),
                         *((X(t, U.start + a), -b) for a, b in enumerate(m["joint_K"][k]))], m["joint_b"][k])
                rows.eq({X(t + 1, QD.start + joint): 1.0, X(t + 1, Q.start + joint): -CONTROL_HZ,
                         X(t, Q.start + joint): CONTROL_HZ}, 0.0)
                rows.add({X(t, Q.start + joint): 1.0, X(t, U.start + k): 1.0}, m["joint_lower"][k], m["joint_upper"][k])
                rows.le({X(t, U.start + k): 1.0, effort[t, k]: -1.0}, 0.0)
                rows.le({X(t, U.start + k): -1.0, effort[t, k]: -1.0}, 0.0)
            for joint in passive:
                rows.eq({X(t + 1, Q.start + joint): 1.0, X(t, Q.start + joint): -1.0}, 0.0)
                rows.eq({X(t + 1, QD.start + joint): 1.0}, 0.0)
            # The pen moves by the sum of the three zones' twist contributions.
            for component, column in enumerate(range(OBJ_POS.start, ERROR.stop)):
                rows.eq([(X(t + 1, column), 1.0), (X(t, column), -1.0),
                         *((X(t, ZONE_TWIST.start + 6 * zone + component), -1.0) for zone in range(3))], 0.0)
            for axis in range(3):
                rows.eq([(X(t + 1, OBJ_VEL.start + axis), 1.0),
                         *((X(t, ZONE_TWIST.start + 6 * zone + axis), -60.0) for zone in range(3))], 0.0)
                rows.eq([(X(t + 1, OBJ_OMEGA.start + axis), 1.0),
                         *((X(t, ZONE_TWIST.start + 6 * zone + 3 + axis), -60.0) for zone in range(3))], 0.0)
            for i in range(15):                                                        # fingertip kinematics
                rows.eq([(X(t + 1, TIP_POS.start + i), 1.0),
                         *((col, -a) for col, a in zip(geometry_features(t + 1), m["tip_C"][i]))], m["tip_b"][i])
            # A selected role brings each member finger into the capsule contact band.
            Mc = cfg.contact_big_m
            for role, members in enumerate(geometry.ROLE_MEMBERS):
                for finger in members:
                    band = [(col, a / Mc) for col, a in zip(geometry_features(t), m["gap_C"][finger])]
                    rows.le([(z[t, role], 1.0), *band], 1 + (gap_tolerance - m["gap_b"][finger]) / Mc)
                    rows.le([(z[t, role], 1.0), *((col, -a) for col, a in band)],
                            1 + (gap_tolerance + m["gap_b"][finger]) / Mc)
            # A selected role-zone arc carries a force given by its affine map, at an axial position in the zone.
            force_features = [X(t, U.start + a) for a in range(20)] + state_features(t)
            for arc in range(6):
                gate = z[t, 3 + arc]
                for axis in range(3):
                    force = X(t, CONTACT_FORCE.start + 3 * arc + axis)
                    rows.le({force: 1 / float(m["force_bound"]), gate: -1.0}, 0.0)
                    rows.le({force: -1 / float(m["force_bound"]), gate: -1.0}, 0.0)
                    K, b, M = m["lambda_K"][arc, axis], float(m["lambda_b"][arc, axis]), cfg.force_match_big_m
                    residual = [(force, 1 / M), *((col, -a / M) for col, a in zip(force_features, K))]
                    rows.le([*residual, (gate, 1.0)], 1 + b / M)
                    rows.le([*((col, -a) for col, a in residual), (gate, 1.0)], 1 - b / M)
                role = int(geometry.ARC_ROLE[arc])
                lo, hi = geometry.ZONE_BOUNDS[int(geometry.ARC_ZONE[arc])]
                Ma = cfg.axial_big_m
                axial = [(col, a / Ma) for col, a in zip(geometry_features(t), m["axial_C"][role])]
                rows.le([(gate, 1.0), *((col, -a) for col, a in axial)],
                        1 + (axial_tolerance - lo + m["axial_b"][role]) / Ma)
                rows.le([(gate, 1.0), *axial], 1 + (axial_tolerance + hi - m["axial_b"][role]) / Ma)
            # Only supported zones contribute twist.
            for zone in range(3):
                for component in range(6):
                    twist = X(t, ZONE_TWIST.start + 6 * zone + component)
                    rows.le({twist: 1 / float(m["twist_bound"]), z[t, 9 + zone]: -1.0}, 0.0)
                    rows.le({twist: -1 / float(m["twist_bound"]), z[t, 9 + zone]: -1.0}, 0.0)
            # Each pair of active roles selects its own affine grasp Jacobian for the total twist.
            twist_features = ([X(t, CONTACT_FORCE.start + j) for j in range(18)]
                              + [X(t, U.start + j) for j in range(20)]
                              + state_features(t) + [X(t, OBJ_VEL.start + j) for j in range(3)]
                              + [X(t, OBJ_OMEGA.start + j) for j in range(3)])
            M = cfg.twist_match_big_m
            for pair, roles in enumerate(ROLE_PAIRS):
                signs, selected = 2.0 * roles - 1.0, int(roles.sum())
                for component in range(6):
                    residual = [*((X(t, ZONE_TWIST.start + 6 * zone + component), 1 / M) for zone in range(3)),
                                *((col, -a / M) for col, a in zip(twist_features, m["role_twist_K"][pair, component]))]
                    gates = [(z[t, f], signs[f]) for f in range(3)]
                    bias = float(m["role_twist_b"][pair, component]) / M
                    rows.le([*residual, *gates], selected + bias)
                    rows.le([*((col, -a) for col, a in residual), *gates], selected - bias)
        for t in range(H + 1):
            for axis in range(3):
                rows.le({X(t, ERROR.start + axis): 1.0, error[t, axis]: -1.0}, 0.0)
                rows.le({X(t, ERROR.start + axis): -1.0, error[t, axis]: -1.0}, 0.0)
        # Soft terminal contact goal: |gap| of every finger, weighted by the goal's roles in goal_cost.
        for finger in range(5):
            gap = [(col, a) for col, a in zip(geometry_features(H), m["gap_C"][finger])]
            rows.le([*gap, (self.goal_gaps[finger], -1.0)], -float(m["gap_b"][finger]))
            rows.le([*((col, -a) for col, a in gap), (self.goal_gaps[finger], -1.0)], float(m["gap_b"][finger]))

        lower, upper = np.full(v.n, -np.inf), np.full(v.n, np.inf)
        lower[z], upper[z] = 0.0, 1.0
        for t in range(H + 1):
            columns = [X(t, ERROR.start + a) for a in range(3)]
            lower[columns], upper[columns] = -2.2, 2.2
        for t in range(H):
            lower[[X(t, U.start + a) for a in range(20)]] = m["control_lower"]
            upper[[X(t, U.start + a) for a in range(20)]] = m["control_upper"]
        lower[error], lower[effort], lower[self.goal_gaps] = 0.0, 0.0, 0.0
        cost = np.zeros(v.n)
        cost[effort] = cfg.effort_weight
        cost[error[:H]], cost[error[H]] = cfg.stage_weight, cfg.terminal_weight
        integrality = np.zeros(v.n, dtype=np.uint8)
        integrality[z] = 1
        return Template(cost, integrality, Bounds(lower, upper), rows.matrix(), np.asarray(rows.lower),
                        np.asarray(rows.upper), state_rows, state_columns, mode_rows)
