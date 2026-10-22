"""Shared structure of the Shadow Hand reorientation tasks (Block, Cuboid, Pen).

Each setting is a fitted local model (``data/<task>/setting_<k>.npz``) of the hand-object system around one
grasp.  The MILP has a fixed coefficient matrix: only the rows that pin the measured state ``x_0`` and mode
``z_0`` and the goal-dependent terminal cost change between steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import csr_matrix

from ...milp import MILP, Rows, Variables
from ..base import DATA_ROOT, Task
from . import flow
from .isaac import FingerFaceEnv, ShadowHandEnv

SUCCESS_TOLERANCE = 0.10          # rotation error norm


@dataclass(frozen=True)
class Template:
    cost: np.ndarray
    integrality: np.ndarray
    bounds: Bounds
    A: csr_matrix
    lower: np.ndarray
    upper: np.ndarray
    state_rows: np.ndarray        # rows pinning x_0[state_columns] and z_0
    state_columns: np.ndarray
    mode_rows: np.ndarray


class HandTask(Task):
    """A Shadow Hand task: ``state = [x, z]`` with ``x`` of size ``NX``; the action is a joint command."""

    settings = (1, 2, 3)
    NX: ClassVar[int]
    U: ClassVar[slice]                       # commanded joints within x
    ERROR: ClassVar[slice]                   # object rotation error within x
    Env: ClassVar[type[ShadowHandEnv]]
    # Evaluation resets: setting -> half-widths of the uniform joint noise and goal rotation-vector perturbation.
    PERTURBATION: ClassVar[dict[int, tuple[float, float]]]
    model_config_fields: ClassVar[frozenset[str]] = frozenset()   # Config fields stored in the setting file

    def __init__(self, setting: int = 1, perturbation: float = 1.0, **overrides):
        if setting not in self.settings:
            raise ValueError(f"{self.name} has settings {self.settings}, not {setting}")
        self.model = dict(np.load(DATA_ROOT / self.name / f"setting_{setting}.npz"))
        super().__init__(setting, **overrides)
        self.perturbation = perturbation
        self.template = self.build_template()
        self.env: ShadowHandEnv | None = None

    def default_config(self):
        """Config defaults, with the fields stored in the setting's model file taking precedence."""
        return self.Config(**{key: np.asarray(self.model[key]).item() for key in self.model_config_fields
                              if key in self.model})

    def build_template(self) -> Template:
        raise NotImplementedError

    def goal_cost(self, cost: np.ndarray, goal: np.ndarray) -> None:
        """Terminal L1 distance of the last planned mode to the goal mode (up to a constant)."""
        last = (self.horizon - 1) * self.nz
        cost[last:last + self.nz] += self.config.terminal_flow_weight * (1.0 - 2.0 * np.asarray(goal, dtype=float))

    def milp(self, state: np.ndarray, goal: np.ndarray) -> MILP:
        T = self.template
        lower, upper, cost = T.lower.copy(), T.upper.copy(), T.cost.copy()
        lower[T.state_rows] = upper[T.state_rows] = np.asarray(state)[T.state_columns]
        lower[T.mode_rows] = upper[T.mode_rows] = np.asarray(state)[self.NX:]
        self.goal_cost(cost, goal)
        return MILP(cost, T.integrality, T.bounds, LinearConstraint(T.A, lower, upper))

    def mode(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float)[self.NX:]

    def action(self, x: np.ndarray) -> np.ndarray:
        start = self.horizon * self.nz
        return x[start:start + self.NX][self.U].copy()

    def success(self, state: np.ndarray, goal: np.ndarray) -> bool:
        error = float(np.linalg.norm(np.asarray(state)[self.ERROR]))
        return not getattr(self.env, "done", False) and error <= SUCCESS_TOLERANCE

    # ------------------------------------------------------------------------------------- simulator
    def launch(self, device: str = "cuda:0") -> None:
        if self.env is None:
            self.env = self.Env(self.model, device=device)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.env.step(np.asarray(action, dtype=float))

    def goal(self) -> np.ndarray:
        return np.asarray(self.model["z_reference"][-1], dtype=float)

    def reset(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if self.env is None:
            raise RuntimeError("call launch() before reset()")
        rng = np.random.default_rng(seed)
        joint_scale, goal_scale = (scale * self.perturbation for scale in self.PERTURBATION[self.setting])
        noise = rng.uniform(-joint_scale, joint_scale, 20) if joint_scale > 0 else np.zeros(20)
        rotvec = rng.uniform(-goal_scale, goal_scale, 3) if goal_scale > 0 else np.zeros(3)
        return self.env.reset(joint_noise=noise, goal_rotvec=rotvec), self.goal()


class FingerFaceTask(HandTask):
    """Block and Cuboid: the finger-face flow selects contacts; contact arcs gate joint-to-object motion.

    State ``x``: joint positions q (24) and velocities (24), commanded targets u (20), object position, rotation
    error and twist, fingertip positions and contact-force proxies (and, for the cuboid, the previous command).
    Per stage the MILP has the flow logic, first-order joint servo dynamics, contact-gated rotation increments
    ``beta = z_arc (u - q)``, an affine fingertip model, force selection, contact-gap gates, face-occupancy and
    idle twist envelopes, and an L1 feedback reference term.
    """

    nz = flow.NZ
    NX = 110
    U, ERROR = slice(48, 68), slice(71, 74)
    Q, QD, OBJ_POS, OBJ_VEL, OBJ_OMEGA = slice(0, 24), slice(24, 48), slice(68, 71), slice(74, 77), slice(77, 80)
    TIP_POS, TIP_FORCE = slice(80, 95), slice(95, 110)
    Env = FingerFaceEnv
    model_config_fields = frozenset({"contact_gap_tolerance", "reference_weight"})
    joint_margin: ClassVar[float] = 0.10
    error_bound: ClassVar[float] = np.pi

    @classmethod
    def discrete_interface(cls):
        return flow.interface()[0]

    def tip_and_gap_models(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Affine fingertip positions and arc gaps in (actuated q, rotation error): ``C, b`` for each."""
        raise NotImplementedError

    def memory_rows(self, rows: Rows, t: int, x) -> None:
        """Extra actuator rows (cuboid: remember the previous command)."""

    def rotation_terms(self, t: int, axis: int, x) -> list[tuple[int, float]]:
        """Extra terms of the object rotation increment (cuboid: wrist motion)."""
        return []

    def feedback_rows(self, rows: Rows, t: int, z, x, reference) -> None:
        """L1 rows ``reference_t >= |u_t - feedback(x_t)|`` of the natural-motion reference."""
        raise NotImplementedError

    def bounds(self, lower: np.ndarray, upper: np.ndarray, x) -> None:
        m = self.model
        for k, joint in enumerate(self.actuated):
            lower[x[:, self.Q.start + joint]] = m["joint_lower"][k] - self.joint_margin
            upper[x[:, self.Q.start + joint]] = m["joint_upper"][k] + self.joint_margin
        lower[x[:, self.ERROR]], upper[x[:, self.ERROR]] = -self.error_bound, self.error_bound
        lower[x[:-1, self.U]], upper[x[:-1, self.U]] = m["joint_lower"], m["joint_upper"]

    def reference_weights(self) -> np.ndarray:
        return np.full(self.horizon, self.config.reference_weight)

    @property
    def actuated(self) -> np.ndarray:
        return self.model["actuated_ids"].astype(int)

    def build_template(self) -> Template:
        cfg, m, H, NZ, NX = self.config, self.model, self.horizon, self.nz, self.NX
        Q, QD, U, ERR = self.Q.start, self.QD.start, self.U.start, self.ERROR.start
        actuated = self.actuated
        passive = [j for j in range(24) if j not in set(actuated.tolist())]
        beta_arc, beta_action = m["beta_arc"].astype(int), m["beta_action"].astype(int)
        gamma, G = m["gamma"], m["G_beta"]
        tip_C, tip_b, gap_C, gap_b = self.tip_and_gap_models()
        interface, e, p = flow.interface()

        v = Variables()
        z, x, beta = v.add(H, NZ), v.add(H + 1, NX), v.add(H, len(beta_arc))
        reference, error, effort = v.add(H, 20), v.add(H + 1, 3), v.add(H, 20)
        rows = Rows(v.n)
        state_columns = np.array([j for j in range(NX) if not U <= j < self.U.stop])
        state_rows = np.array([rows.eq({x[0, j]: 1.0}, 0.0) for j in state_columns])
        mode_rows = np.array([rows.eq({z[0, j]: 1.0}, 0.0) for j in range(NZ)])
        for t in range(H):
            for row, rhs in zip(interface.E, e):
                rows.eq({z[t, j]: row[j] for j in np.flatnonzero(row)}, rhs)
            for row, rhs in zip(interface.P, p):
                rows.le({z[t, j]: row[j] for j in np.flatnonzero(row)}, rhs)
            # First-order joint servo: q_{t+1} = (1 - gamma) q_t + gamma u_t; L1 effort |u_t - q_t|.
            for k, joint in enumerate(actuated):
                rows.eq({x[t + 1, Q + joint]: 1.0, x[t, Q + joint]: -(1.0 - gamma[k]), x[t, U + k]: -gamma[k]}, 0.0)
                rows.eq({x[t + 1, QD + joint]: 1.0, x[t + 1, Q + joint]: -1.0, x[t, Q + joint]: 1.0}, 0.0)
                rows.le({x[t, U + k]: 1.0, x[t, Q + joint]: -1.0, effort[t, k]: -1.0}, 0.0)
                rows.le({x[t, U + k]: -1.0, x[t, Q + joint]: 1.0, effort[t, k]: -1.0}, 0.0)
            for joint in passive:
                rows.eq({x[t + 1, Q + joint]: 1.0, x[t, Q + joint]: -1.0}, 0.0)
                rows.eq({x[t + 1, QD + joint]: 1.0}, 0.0)
            self.memory_rows(rows, t, x)
            # Contact-arc-gated joint increments beta = z_arc (u - q), big-M normalized.
            M = cfg.beta_big_m
            for b, (arc, k) in enumerate(zip(beta_arc, beta_action)):
                joint, gate = int(actuated[k]), z[t, 5 + arc]
                rows.le({beta[t, b]: 1.0 / M, gate: -1.0}, 0.0)
                rows.le({beta[t, b]: -1.0 / M, gate: -1.0}, 0.0)
                rows.le({beta[t, b]: 1.0 / M, x[t, U + k]: -1.0 / M, x[t, Q + joint]: 1.0 / M, gate: 1.0}, 1.0)
                rows.le({beta[t, b]: -1.0 / M, x[t, U + k]: 1.0 / M, x[t, Q + joint]: -1.0 / M, gate: 1.0}, 1.0)
            # Object rotation is driven by the contact increments only (no free drift).
            for axis in range(3):
                rows.eq([(x[t + 1, ERR + axis], 1.0), (x[t, ERR + axis], -1.0),
                         *((beta[t, b], -G[axis, b]) for b in range(len(beta_arc))),
                         *self.rotation_terms(t, axis, x)], 0.0)
                rows.eq({x[t + 1, self.OBJ_OMEGA.start + axis]: 1.0, x[t + 1, ERR + axis]: 1.0,
                         x[t, ERR + axis]: -1.0}, 0.0)
                rows.eq({x[t + 1, self.OBJ_VEL.start + axis]: 1.0}, 0.0)
                rows.eq({x[t + 1, self.OBJ_POS.start + axis]: 1.0, x[t, self.OBJ_POS.start + axis]: -1.0}, 0.0)
            # Affine fingertip kinematics.
            for i in range(15):
                rows.eq([(x[t + 1, self.TIP_POS.start + i], 1.0),
                         *((x[t + 1, Q + joint], -tip_C[i, k]) for k, joint in enumerate(actuated)),
                         *((x[t + 1, ERR + a], -tip_C[i, 20 + a]) for a in range(3))], tip_b[i])
            # Only selected fingers carry a contact-force proxy.
            for finger in range(5):
                for axis in range(3):
                    force = x[t + 1, self.TIP_FORCE.start + 3 * finger + axis]
                    rows.le({force: 0.05, z[t, finger]: -1.0}, 0.0)
                    rows.le({force: -0.05, z[t, finger]: -1.0}, 0.0)
            # A selected finger-face arc requires the fingertip near that face.
            Mg = cfg.contact_gap_big_m
            for arc in range(8):
                rows.le([(z[t, 5 + arc], 1.0),
                         *((x[t, Q + joint], gap_C[arc, k] / Mg) for k, joint in enumerate(actuated)),
                         *((x[t, ERR + a], gap_C[arc, 20 + a] / Mg) for a in range(3))],
                        1.0 + (cfg.contact_gap_tolerance - gap_b[arc]) / Mg)
            # Face occupancy widens the object twist envelope about that face's normal ...
            for face in range(6):
                axis = int(np.flatnonzero(flow.FACE_NORMALS[face])[0])
                for sign in (-1.0, 1.0):
                    rows.le({x[t + 1, self.OBJ_OMEGA.start + axis]: sign / cfg.occupancy_rotation_gain,
                             z[t, flow.OCCUPANCY + face]: -1.0},
                            cfg.occupancy_base_rotation / cfg.occupancy_rotation_gain)
            # ... and an idle (two-contact regrasp) phase narrows it.
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    rows.le({x[t + 1, self.OBJ_OMEGA.start + axis]: sign / cfg.idle_rotation_limit,
                             z[t, flow.IDLE]: 1.0}, 2.0)
            self.feedback_rows(rows, t, z, x, reference)
        for t in range(H + 1):
            for axis in range(3):
                rows.le({x[t, ERR + axis]: 1.0, error[t, axis]: -1.0}, 0.0)
                rows.le({x[t, ERR + axis]: -1.0, error[t, axis]: -1.0}, 0.0)

        lower, upper = np.full(v.n, -np.inf), np.full(v.n, np.inf)
        lower[z], upper[z] = 0.0, 1.0
        self.bounds(lower, upper, x)
        lower[reference], lower[error], lower[effort] = 0.0, 0.0, 0.0
        cost = np.zeros(v.n)
        cost[reference] = self.reference_weights()[:, None]
        cost[effort] = cfg.effort_weight
        cost[error[:H]], cost[error[H]] = cfg.stage_weight, cfg.terminal_weight
        integrality = np.zeros(v.n, dtype=np.uint8)
        integrality[z] = 1
        return Template(cost, integrality, Bounds(lower, upper), rows.matrix(), np.asarray(rows.lower),
                        np.asarray(rows.upper), state_rows, state_columns, mode_rows)
