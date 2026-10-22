"""IsaacLab execution for the Shadow Hand tasks (requires IsaacLab; one simulator per process).

The executed plant is IsaacLab's Shadow Hand in-hand reorientation environment with its native contact
solver; the MILP's affine contact models are only used for planning.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from .flow import FlowMapper

TASK_ID = "Isaac-Repose-Cube-Shadow-Direct-v0"
FINGERTIPS = ("robot0_ffdistal", "robot0_mfdistal", "robot0_rfdistal", "robot0_lfdistal", "robot0_thdistal")


def quaternion_to_matrix(q_wxyz) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def matrix_to_quaternion(matrix) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(matrix, dtype=float)).as_quat()[[3, 0, 1, 2]]


def numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(float)


def native(tensor) -> np.ndarray:
    """The tensor's values in its own precision (IsaacLab uses float32)."""
    return tensor.detach().cpu().numpy()


class ShadowHandEnv:
    """One IsaacLab Shadow Hand environment.  Closing it also closes the simulation app."""

    def __init__(self, model: dict, device: str = "cuda:0"):
        from isaaclab.app import AppLauncher

        self.app = AppLauncher(headless=True, enable_cameras=False, device=device).app
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  (registers the environment)
        from isaaclab_tasks.utils import parse_env_cfg

        cfg = parse_env_cfg(TASK_ID, device=device, num_envs=1, use_fabric=True)
        self.configure(cfg)
        self.model = model
        self.seed = int(model.get("environment_seed", 0))
        cfg.seed = self.seed
        cfg.scene.num_envs = 1
        cfg.scene.clone_in_fabric = False
        cfg.success_tolerance = 0.0
        cfg.sim.log_dir = str(Path(tempfile.gettempdir()) / "good_symbols_isaac")
        self.env = gym.make(TASK_ID, cfg=cfg)
        self.raw = self.env.unwrapped
        bodies = {name: i for i, name in enumerate(self.raw.hand.body_names)}
        self.fingertips = [bodies[name] for name in FINGERTIPS]
        self.actuated = model["actuated_ids"].astype(int).tolist()
        self.done = False

    def configure(self, cfg) -> None:
        """Adjust the environment configuration (e.g. replace the object) before it is created."""

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Actuated joint limits in the simulator's float32 precision (the action scaling uses them as is)."""
        return (native(self.raw.hand_dof_lower_limits[0, self.actuated]),
                native(self.raw.hand_dof_upper_limits[0, self.actuated]))

    def close(self) -> None:
        self.env.close()
        self.app.close()


class FingerFaceEnv(ShadowHandEnv):
    """Block and Cuboid: joint-position targets in, ``[x, z]`` out with ``z`` the measured contact flow."""

    NX = 110                     # x = q, qd, previous target, object pose/error/twist, fingertips, tip forces
    HALF_EXTENTS = np.full(3, 0.04)
    RESET_TIMELINE = False       # restart PhysX between episodes (clears its contact warm start)

    def __init__(self, model: dict, device: str = "cuda:0"):
        super().__init__(model, device)
        self.mapper = FlowMapper(model["feasible_flows"], self.HALF_EXTENTS)
        self.goal_rotation = model["goal_rotation"].copy()
        self.previous_target = model["initial_previous_target"].copy()
        self._has_reset = False

    def rotation_error(self, rotation: np.ndarray) -> np.ndarray:
        return Rotation.from_matrix(rotation.T @ self.goal_rotation).as_rotvec()

    def reset(self, *, joint_noise=None, goal_rotvec=None, initial_rotvec=None) -> np.ndarray:
        """Restore the setting's reference state, then perturb joints, goal and object orientation."""
        import torch

        joint_noise = np.zeros(20) if joint_noise is None else joint_noise
        goal_delta = Rotation.from_rotvec(np.zeros(3) if goal_rotvec is None else goal_rotvec).as_matrix()
        initial_delta = Rotation.from_rotvec(np.zeros(3) if initial_rotvec is None else initial_rotvec).as_matrix()
        self.goal_rotation = self.model["goal_rotation"] @ goal_delta
        if self.RESET_TIMELINE and self._has_reset:
            self.raw.sim.reset()
        self.env.reset(seed=self.seed)
        self._has_reset = True
        raw, x = self.raw, self.model["x_reference"][0]
        tensor = lambda value, like: torch.as_tensor(value, dtype=like.dtype, device=like.device)
        q, qd = raw.hand.data.joint_pos.clone(), raw.hand.data.joint_vel.clone()
        q[0], qd[0] = tensor(x[0:24], q), tensor(x[24:48], qd)
        raw.hand.write_joint_state_to_sim(q, qd)
        prior_target = self.model["initial_previous_target"]
        targets = q.clone()
        targets[0, self.actuated] = tensor(prior_target, q)
        raw.prev_targets[0] = raw.cur_targets[0] = raw.hand_dof_targets[0] = targets[0]
        raw.hand.set_joint_position_target(targets[:, self.actuated], joint_ids=self.actuated)
        root = raw.object.data.root_state_w.clone()
        root[0, 0:3] = tensor(x[68:71], root)
        root[0, 3:7] = tensor(matrix_to_quaternion(self.model["object_rotation"][0]), root)
        root[0, 7:13] = tensor(np.concatenate([x[74:77], x[77:80]]), root)
        raw.object.write_root_state_to_sim(root)
        raw.goal_rot[0] = tensor(matrix_to_quaternion(self.goal_rotation), raw.goal_rot)
        raw.goal_markers.visualize(raw.goal_pos + raw.scene.env_origins, raw.goal_rot)
        raw.episode_length_buf.zero_()
        raw.reset_goal_buf.zero_()
        raw.successes.zero_()
        raw.scene.write_data_to_sim()
        raw.sim.forward()
        raw._compute_intermediate_values()
        # Perturb after the complete reference reset.
        q, qd = raw.hand.data.joint_pos.clone(), raw.hand.data.joint_vel.clone()
        q[0, self.actuated] += tensor(joint_noise, q)
        raw.hand.write_joint_state_to_sim(q, qd)
        root = raw.object.data.root_state_w.clone()
        root[0, 3:7] = tensor(matrix_to_quaternion(self.model["object_rotation"][0] @ initial_delta), root)
        raw.object.write_root_state_to_sim(root)
        raw.goal_rot[0] = tensor(matrix_to_quaternion(self.goal_rotation), raw.goal_rot)
        raw.scene.write_data_to_sim()
        raw.sim.forward()
        raw._compute_intermediate_values()
        self.previous_target = prior_target.copy()
        return self.observe()

    def observe(self) -> np.ndarray:
        raw = self.raw
        raw._compute_intermediate_values()
        origin = raw.scene.env_origins[0]
        rotation = quaternion_to_matrix(numpy(raw.object.data.root_quat_w[0]))
        tips = numpy(raw.hand.data.body_pos_w[0, self.fingertips] - origin)
        forces = numpy(raw.hand.root_physx_view.get_link_incoming_joint_force()[0, self.fingertips])[:, :3]
        position = numpy(raw.object.data.root_pos_w[0] - origin)
        x = np.zeros(self.NX)
        x[0:24], x[24:48] = numpy(raw.hand.data.joint_pos[0]), numpy(raw.hand.data.joint_vel[0])
        x[48:68], x[68:71] = self.previous_target, position
        x[71:74] = self.rotation_error(rotation)
        x[74:77], x[77:80] = numpy(raw.object.data.root_lin_vel_w[0]), numpy(raw.object.data.root_ang_vel_w[0])
        x[80:95], x[95:110] = tips.reshape(-1), forces.reshape(-1)
        if self.NX > 110:
            x[110:130] = self.previous_target
        return np.concatenate([x, self.mapper(tips, forces, position, rotation)])

    def step(self, target: np.ndarray) -> np.ndarray:
        import torch

        lower, upper = self.joint_limits()
        action = np.clip(2.0 * (np.asarray(target) - lower) / (upper - lower) - 1.0, -1.0, 1.0)
        self.env.step(torch.as_tensor(action, dtype=torch.float32, device=self.raw.device)[None])
        self.previous_target = np.asarray(target).copy()
        return self.observe()
