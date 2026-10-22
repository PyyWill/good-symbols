"""Cuboid: the Shadow Hand reorients a 55 x 70 x 110 mm cuboid; reachability depends on the orientation.

Compared to Block, the state also carries the previous command, the wrist joints rotate the object directly,
and the feedback reference is contact-mode dependent: each of the 28 feasible flows has its own affine map,
selected by the flow ``z`` itself (big-M on the Hamming distance).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ...milp import Rows
from ..hand.isaac import FingerFaceEnv
from ..hand.task import FingerFaceTask

PREV_U = slice(110, 130)


@dataclass(frozen=True)
class Config:
    horizon: int = 2
    time_limit: float = 0.5
    mip_rel_gap: float = 1e-4
    terminal_weight: float = 50.0
    stage_weight: float = 0.5
    reference_weight: float = 12.0
    effort_weight: float = 0.0
    terminal_flow_weight: float = 0.1
    contact_gap_tolerance: float = 0.055
    contact_gap_big_m: float = 0.20
    beta_big_m: float = 2.0
    occupancy_base_rotation: float = 0.22
    occupancy_rotation_gain: float = 0.22
    idle_rotation_limit: float = 0.22


class CuboidEnv(FingerFaceEnv):
    NX = 130
    HALF_EXTENTS = np.array([0.0275, 0.0350, 0.0550])
    RESET_TIMELINE = True

    def __init__(self, model: dict, device: str = "cuda:0"):
        super().__init__(model, device)
        self.hemisphere = model["error_hemisphere"].copy()

    def configure(self, cfg) -> None:
        """Replace the task's cube by the cuboid."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.markers import VisualizationMarkersCfg

        def spawn(goal: bool = False):
            physical = None if goal else dict(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False, disable_gravity=False, enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8, solver_velocity_iteration_count=0, sleep_threshold=0.005,
                    stabilization_threshold=0.0025, max_depenetration_velocity=1000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.22),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0,
                                                                 restitution=0.0))
            return sim_utils.CuboidCfg(size=(0.055, 0.070, 0.110), **(physical or {}),
                                       visual_material=sim_utils.PreviewSurfaceCfg(
                                           diffuse_color=(0.72, 0.20, 0.16) if goal else (0.12, 0.42, 0.72),
                                           opacity=0.35 if goal else 1.0, roughness=0.55))

        cfg.object_cfg = RigidObjectCfg(prim_path="/World/envs/env_.*/object", spawn=spawn(),
                                        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6),
                                                                                  rot=(1.0, 0.0, 0.0, 0.0)))
        cfg.goal_object_cfg = VisualizationMarkersCfg(prim_path="/Visuals/goal_marker",
                                                      markers={"goal": spawn(goal=True)})

    def rotation_error(self, rotation: np.ndarray) -> np.ndarray:
        """Twice the vector part of the relative quaternion, on the setting's hemisphere."""
        relative = Rotation.from_matrix(rotation.T @ self.goal_rotation).as_quat()
        return 2.0 * (relative if relative @ self.hemisphere >= 0.0 else -relative)[:3]


class Cuboid(FingerFaceTask):
    name = "cuboid"
    max_steps = 60
    NX = 130
    Config = Config
    Env = CuboidEnv
    joint_margin = 0.20        # tendon-coupled resets can exceed the nominal actuated limits by ~0.14 rad
    error_bound = 2.2
    # Setting 1's fitted model holds only at its nominal state: use settings 2 and 3 for perturbed evaluation.
    PERTURBATION = {1: (0.0001, 0.0011), 2: (0.0001, 0.00094), 3: (0.0001, 0.0011)}

    def tip_and_gap_models(self):
        m = self.model
        return m["tip_C"], m["tip_b"], m["gap_C"], m["gap_b"]

    def memory_rows(self, rows: Rows, t: int, x) -> None:
        for k in range(20):
            rows.eq({x[t + 1, PREV_U.start + k]: 1.0, x[t, self.U.start + k]: -1.0}, 0.0)

    def rotation_terms(self, t: int, axis: int, x) -> list[tuple[int, float]]:
        G = self.model["G_wrist"]
        return [term for k in range(2) for term in ((x[t, self.U.start + k], -G[axis, k]),
                                                     (x[t, self.Q.start + self.actuated[k]], G[axis, k]))]

    def bounds(self, lower: np.ndarray, upper: np.ndarray, x) -> None:
        super().bounds(lower, upper, x)
        lower[x[:, PREV_U]] = self.model["joint_lower"] - self.joint_margin
        upper[x[:, PREV_U]] = self.model["joint_upper"] + self.joint_margin

    def reference_weights(self) -> np.ndarray:
        """Only the executed command u_0 tracks the reference; later commands are free MPC decisions."""
        return np.r_[self.config.reference_weight, np.zeros(self.horizon - 1)]

    def feedback_rows(self, rows: Rows, t: int, z, x, reference) -> None:
        m = self.model
        features = ([x[t, PREV_U.start + k] for k in range(20)] + [x[t, self.Q.start + j] for j in self.actuated]
                    + [x[t, self.ERROR.start + a] for a in range(3)])
        feature_lower = np.concatenate((m["joint_lower"] - 0.20, m["joint_lower"] - 0.20, np.full(3, -2.2)))
        feature_upper = np.concatenate((m["joint_upper"] + 0.20, m["joint_upper"] + 0.20, np.full(3, 2.2)))
        for mode, flow in enumerate(m["mode_flows"].astype(int)):
            hamming, signs = int(flow.sum()), 2.0 * flow - 1.0       # z == flow  iff  signs . z == hamming
            for k in range(20):
                K, bias = m["mode_K"][mode, k], float(m["mode_b"][mode, k])
                low = bias + float(np.minimum(K * feature_lower, K * feature_upper).sum())
                high = bias + float(np.maximum(K * feature_lower, K * feature_upper).sum())
                M = max(float(m["joint_upper"][k]) - low, high - float(m["joint_lower"][k]), 1.0)
                residual = [(x[t, self.U.start + k], 1.0 / M), *((col, -a / M) for col, a in zip(features, K))]
                gates = [(z[t, j], signs[j]) for j in range(self.nz)]
                rows.le([*residual, (reference[t, k], -1.0 / M), *gates], hamming + bias / M)
                rows.le([*((col, -a) for col, a in residual), (reference[t, k], -1.0 / M), *gates], hamming - bias / M)
