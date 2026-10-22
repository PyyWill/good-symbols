"""IsaacLab execution of the Pen task with a rigid no-slip grasp.

The pen is kinematic: after the hand moves, the pen pose is solved from the measured fingertip motion so that
active contacts neither slip nor separate (a least-squares rigid fit, then a projection that preserves the
contact gaps).  An episode ends when the active roles lose contact.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..hand.isaac import ShadowHandEnv, matrix_to_quaternion, native, numpy, quaternion_to_matrix
from . import geometry

NX = 131
Q, QD, OBJ_POS, ERROR = slice(0, 24), slice(24, 48), slice(48, 51), slice(51, 54)
OBJ_VEL, OBJ_OMEGA, TIP_POS, U = slice(54, 57), slice(57, 60), slice(60, 75), slice(75, 95)
CONTACT_FORCE, ZONE_TWIST = slice(95, 113), slice(113, 131)


class PenEnv(ShadowHandEnv):
    """Joint increments in, ``[x, z]`` out with ``z`` the measured role-zone flow."""

    def __init__(self, model: dict, device: str = "cuda:0"):
        if not bool(model["kinematic_no_slip"]):
            raise ValueError("the pen settings require the kinematic no-slip grasp")
        super().__init__(model, device)
        self.no_slip_tolerance = float(model["no_slip_tolerance"])
        self.translation_trust = float(model["no_slip_translation_trust"])
        self.rotation_trust = float(model["no_slip_rotation_trust"])
        self.mapper = geometry.FlowMapper(model["feasible_flows"])
        self.goal_rotation = model["goal_rotation"].copy()
        self.relative_quaternion = model["error_hemisphere"].copy()
        self._has_reset = False
        self.done_reason: str | None = None
        self.max_contact_gap = 0.0

    def configure(self, cfg) -> None:
        """A kinematic, gravity-free capsule replaces the cube."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.markers import VisualizationMarkersCfg

        def capsule(goal: bool = False):
            physical = None if goal else dict(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True, enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8, solver_velocity_iteration_count=0, sleep_threshold=0.005,
                    stabilization_threshold=0.0025, max_depenetration_velocity=1000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0,
                                                                 restitution=0.0))
            return sim_utils.CapsuleCfg(radius=geometry.RADIUS, height=geometry.LENGTH, axis="Z", **(physical or {}),
                                        visual_material=sim_utils.PreviewSurfaceCfg(
                                            diffuse_color=(0.72, 0.20, 0.16) if goal else (0.12, 0.42, 0.72),
                                            opacity=0.35 if goal else 1.0, roughness=0.55))

        cfg.object_cfg = RigidObjectCfg(prim_path="/World/envs/env_.*/object", spawn=capsule(),
                                        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6),
                                                                                  rot=(1.0, 0.0, 0.0, 0.0)))
        cfg.goal_object_cfg = VisualizationMarkersCfg(prim_path="/Visuals/goal_marker",
                                                      markers={"goal": capsule(goal=True)})
        cfg.scene.replicate_physics = False

    def reset(self, *, joint_noise=None, goal_rotvec=None) -> np.ndarray:
        """Restore the setting's grasp; the goal orientation is rotated by ``goal_rotvec``."""
        import torch

        delta = Rotation.from_rotvec(np.zeros(3) if goal_rotvec is None else goal_rotvec).as_matrix()
        self.goal_rotation = self.model["goal_rotation"] @ delta
        if self._has_reset:
            self.raw.sim.reset()
        self.env.reset(seed=self.seed)
        self._has_reset = True
        raw, m = self.raw, self.model
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=raw.device)
        q, qd = tensor(m["reset_joint_pos"])[None], tensor(m["reset_joint_vel"])[None]
        raw.hand.write_joint_state_to_sim(q, qd)
        raw.hand.set_joint_position_target(q)
        raw.prev_targets[0], raw.cur_targets[0] = tensor(m["reset_prev_targets"]), tensor(m["reset_cur_targets"])
        raw.hand_dof_targets[0] = q[0]
        raw.actions[0] = tensor(m["reset_actions"])
        pose = tensor(np.r_[m["reset_object_pos"], m["reset_object_quat"]])[None]
        pose[0, :3] += raw.scene.env_origins[0]
        raw.object.write_root_pose_to_sim(pose)
        raw.object.write_root_velocity_to_sim(tensor(m["reset_object_velocity"])[None])
        raw.goal_rot[0] = torch.as_tensor(m["goal_quaternion"], dtype=raw.goal_rot.dtype, device=raw.goal_rot.device)
        raw.goal_markers.visualize(raw.goal_pos + raw.scene.env_origins, raw.goal_rot)
        self._clear_episode_buffers()
        raw.scene.write_data_to_sim()
        raw.sim.forward()
        raw._compute_intermediate_values()
        self._clear_episode_buffers()
        raw._compute_intermediate_values()
        rotation = quaternion_to_matrix(numpy(raw.object.data.root_quat_w[0]))
        self.relative_quaternion = Rotation.from_matrix(rotation.T @ self.goal_rotation).as_quat()
        self.done, self.done_reason, self.max_contact_gap = False, None, 0.0
        state = self._state()
        self._check_contact(state, self._flow(state), "reset")
        return self.observe()

    def _clear_episode_buffers(self) -> None:
        self.raw.episode_length_buf.zero_()
        self.raw.reset_goal_buf.zero_()
        self.raw.successes.zero_()

    def _state(self) -> dict:
        raw = self.raw
        raw._compute_intermediate_values()
        origin = raw.scene.env_origins[0]
        rotation = quaternion_to_matrix(numpy(raw.object.data.root_quat_w[0]))
        relative = Rotation.from_matrix(rotation.T @ self.goal_rotation).as_quat()
        if relative @ self.relative_quaternion < 0:                 # keep the quaternion sign continuous
            relative *= -1
        self.relative_quaternion = relative.copy()
        return {
            "q": numpy(raw.hand.data.joint_pos[0]), "qd": numpy(raw.hand.data.joint_vel[0]),
            "position": numpy(raw.object.data.root_pos_w[0] - origin), "rotation": rotation,
            "velocity": numpy(raw.object.data.root_lin_vel_w[0]), "omega": numpy(raw.object.data.root_ang_vel_w[0]),
            "tips": numpy(raw.hand.data.body_pos_w[0, self.fingertips] - origin),
            "tip_rotation": np.stack([quaternion_to_matrix(q)
                                      for q in numpy(raw.hand.data.body_quat_w[0, self.fingertips])]),
            "force": numpy(raw.hand.root_physx_view.get_link_incoming_joint_force()[0, self.fingertips])[:, :3],
            "error": 2 * relative[:3],
        }

    def _flow(self, state: dict) -> np.ndarray:
        return self.mapper(state["tips"], state["force"], state["position"], state["rotation"], state["tip_rotation"])

    def observe(self) -> np.ndarray:
        state = self._state()
        x = np.zeros(NX)
        x[Q], x[QD], x[OBJ_POS], x[ERROR] = state["q"], state["qd"], state["position"], state["error"]
        x[OBJ_VEL], x[OBJ_OMEGA], x[TIP_POS] = state["velocity"], state["omega"], state["tips"].reshape(-1)
        z = self._flow(state)
        for arc in np.flatnonzero(z[3:9]):                          # mean contact force of the arc's role
            members = list(geometry.ROLE_MEMBERS[geometry.ARC_ROLE[arc]])
            x[CONTACT_FORCE.start + 3 * arc:CONTACT_FORCE.start + 3 * arc + 3] = state["force"][members].mean(0)
        return np.r_[x, z]

    def step(self, increment: np.ndarray) -> np.ndarray:
        import torch

        before = self._state()
        flow = self._flow(before)
        q = numpy(self.raw.hand.data.joint_pos[0, self.actuated])
        lower, upper = self.joint_limits()
        target = q + np.asarray(increment)
        violation = max(float(np.max(lower - target)), float(np.max(target - upper)), 0.0)
        if violation > 1e-6:
            raise RuntimeError(f"joint target outside the joint limits by {violation:.3e}")
        previous = native(self.raw.prev_targets[0, self.actuated])
        alpha = float(self.raw.cfg.act_moving_average)
        action = 2 * ((target - (1.0 - alpha) * previous) / alpha - lower) / (upper - lower) - 1   # undo the smoothing
        _, _, terminated, truncated, _ = self.env.step(torch.as_tensor(action, dtype=torch.float32,
                                                                       device=self.raw.device)[None])
        terminated, truncated = bool(terminated[0].item()), bool(truncated[0].item())
        self.done = terminated or truncated
        self.done_reason = "terminated" if terminated else "truncated" if truncated else None
        self._apply_no_slip(before, flow)
        return self.observe()

    # ------------------------------------------------------------------------------- rigid no-slip
    def _apply_no_slip(self, before: dict, flow: np.ndarray) -> None:
        import torch

        after = self._state()
        fingers = self._check_contact(before, flow, "pre-step")
        if self.done:
            return
        anchors, arms, source = self._contact_anchors(before, fingers)
        target = after["tips"][fingers] + np.einsum("fij,fj->fi", after["tip_rotation"][fingers], anchors)
        matrix = np.vstack([np.column_stack((np.eye(3), -_skew(arm))) for arm in arms])
        # Anchors can be nearly collinear along the pen: regularize rotation about that line.
        ridge = np.diag([1e-6] * 3 + [1e-2] * 3)
        twist = np.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ (target - source).ravel())
        translation = twist[:3] * min(1.0, self.translation_trust / max(np.linalg.norm(twist[:3]), 1e-12))
        rotation_vector = twist[3:] * min(1.0, self.rotation_trust / max(np.linalg.norm(twist[3:]), 1e-12))
        gaps = geometry.contact_features(before["tips"], before["position"], before["rotation"],
                                         before["tip_rotation"])[0]
        predicted_rotation = Rotation.from_rotvec(rotation_vector).as_matrix() @ before["rotation"]
        position, rotation = _project_contact_pose(after, fingers, gaps[fingers], before["position"] + translation,
                                                   predicted_rotation)
        origin = numpy(self.raw.scene.env_origins[0])
        self.raw.object.write_root_pose_to_sim(torch.as_tensor(np.r_[position + origin, matrix_to_quaternion(rotation)],
                                                               dtype=torch.float32, device=self.raw.device)[None])
        self.raw.object.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=self.raw.device))
        self.raw.scene.write_data_to_sim()
        self.raw.sim.forward()
        self._check_contact(self._state(), flow, "post-step")

    @staticmethod
    def _contact_anchors(state: dict, fingers: np.ndarray):
        """Contact points in fingertip frames, their arms from the pen center, and in the world frame."""
        pen_a = state["position"] + state["rotation"] @ np.array([0.0, 0.0, -geometry.SEGMENT_HALF])
        pen_b = state["position"] + state["rotation"] @ np.array([0.0, 0.0, geometry.SEGMENT_HALF])
        local, arms, world = [], [], []
        for f in fingers:
            axis = state["tip_rotation"][f, :, 2]
            center = state["tips"][f] + geometry.FINGER_CAPSULE_CENTER[f] * axis
            half = geometry.FINGER_CAPSULE_HALF[f] * axis
            finger_axis, pen_axis = geometry.segment_closest_points(center - half, center + half, pen_a, pen_b)
            normal = (pen_axis - finger_axis) / max(np.linalg.norm(pen_axis - finger_axis), 1e-12)
            finger_surface = finger_axis + geometry.FINGER_CAPSULE_RADIUS[f] * normal
            pen_surface = pen_axis - geometry.RADIUS * normal
            contact = 0.5 * (finger_surface + pen_surface)
            local.append(state["tip_rotation"][f].T @ (contact - state["tips"][f]))
            arms.append(contact - state["position"])
            world.append(contact)
        return np.asarray(local), np.asarray(arms), np.asarray(world)

    def _check_contact(self, state: dict, flow: np.ndarray, stage: str) -> np.ndarray:
        """Fingers of the active roles; ends the episode if fewer than three or one lost contact."""
        fingers = np.unique(np.concatenate([np.asarray(geometry.ROLE_MEMBERS[r], dtype=int)
                                            for r in np.flatnonzero(np.asarray(flow)[:3])]))
        gaps = geometry.contact_features(state["tips"], state["position"], state["rotation"], state["tip_rotation"])[0]
        residual = float(np.max(np.abs(gaps[fingers])))
        self.max_contact_gap = max(self.max_contact_gap, residual)
        if len(fingers) < 3 or residual > self.no_slip_tolerance:
            self.done = True
            self.done_reason = f"no-slip contact lost ({stage}, fingers {fingers.tolist()}, gap {residual:.6f})"
        return fingers


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _project_contact_pose(state: dict, fingers, contact_gaps, position, rotation):
    """Project the rigid prediction onto poses that keep the active contact gaps (and avoid penetration)."""
    initial = np.r_[position, Rotation.from_matrix(rotation).as_rotvec()]

    def residual(pose):
        candidate = Rotation.from_rotvec(pose[3:]).as_matrix()
        gaps = geometry.contact_features(state["tips"], pose[:3], candidate, state["tip_rotation"])[0]
        return np.r_[0.03 * (pose[:3] - position) / 0.005,
                     0.03 * Rotation.from_matrix(rotation.T @ candidate).as_rotvec() / 0.10,
                     (gaps[fingers] - contact_gaps) / 0.0005, np.minimum(gaps + 0.001, 0.0) / 0.0002]

    solution = least_squares(residual, initial, max_nfev=80, xtol=1e-9, ftol=1e-9, gtol=1e-9)
    if not solution.success:
        raise RuntimeError(f"rigid no-slip projection failed: {solution.message}")
    return solution.x[:3], Rotation.from_rotvec(solution.x[3:]).as_matrix()
