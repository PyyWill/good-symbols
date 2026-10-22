"""Block: the Shadow Hand reorients an 80 mm cube to a goal orientation (IsaacLab)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...milp import Rows
from ..hand.task import FingerFaceTask


@dataclass(frozen=True)
class Config:
    horizon: int = 2
    time_limit: float = 0.1
    mip_rel_gap: float = 1e-4
    terminal_weight: float = 50.0
    stage_weight: float = 0.5
    reference_weight: float = 12.0
    effort_weight: float = 0.02
    terminal_flow_weight: float = 0.1
    contact_gap_tolerance: float = 0.055
    contact_gap_big_m: float = 0.20
    beta_big_m: float = 2.0
    occupancy_base_rotation: float = 0.22
    occupancy_rotation_gain: float = 0.22
    idle_rotation_limit: float = 0.22


class Block(FingerFaceTask):
    name = "block"
    max_steps = 30
    Config = Config
    PERTURBATION = {1: (0.001, 0.0051), 2: (0.001, 0.007), 3: (0.0001, 0.0012)}

    def tip_and_gap_models(self):
        """Ridge fits of fingertip positions and arc gaps along the setting's reference trajectory."""
        x = self.model["x_reference"]
        design = np.column_stack([x[:, self.Q][:, self.actuated], x[:, self.ERROR], np.ones(len(x))])
        regularizer = 1e-4 * np.eye(design.shape[1])
        regularizer[-1, -1] = 0.0
        fit = lambda target: np.linalg.solve(design.T @ design + regularizer, design.T @ target)
        tip, gap = fit(x[:, self.TIP_POS]), fit(self.model["assignment_gaps"])
        return tip[:-1].T, tip[-1], gap[:-1].T, gap[-1]

    def feedback_rows(self, rows: Rows, t: int, z, x, reference) -> None:
        """One affine natural-motion reference ``u = K [q, qd, error] + b`` fitted on the reference trajectory."""
        K, bias = self.model["K_reference"], self.model["b_reference"]
        Q, QD, ERR = self.Q.start, self.QD.start, self.ERROR.start
        for k in range(20):
            terms = [(x[t, self.U.start + k], 1.0),
                     *((x[t, Q + joint], -K[k, f]) for f, joint in enumerate(self.actuated)),
                     *((x[t, QD + joint], -K[k, 20 + f]) for f, joint in enumerate(self.actuated)),
                     *((x[t, ERR + a], -K[k, 40 + a]) for a in range(3))]
            rows.le([*terms, (reference[t, k], -1.0)], bias[k])
            rows.le([*((j, -a) for j, a in terms), (reference[t, k], -1.0)], -bias[k])
