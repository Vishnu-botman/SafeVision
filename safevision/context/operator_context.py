"""
Operator Context and Cabin State Estimator
Determines whether a detected worker is an active machinery cabin operator
or an external ground pedestrian, calculating G_i(t).
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from safevision.config import OperatorContextConfig
from safevision.tracking.byte_track import TrackedObject


class OperatorState(Enum):
    PEDESTRIAN = "pedestrian"         # Ground worker exposed to moving machinery
    CONFIRMED_OPERATOR = "operator"   # Inside cabin, driving/operating the machine
    UNCERTAIN = "uncertain"           # Near cabin or partially occluded (conservative: not suppressed)


class OperatorContextEstimator:
    """
    Evaluates worker-machinery spatial containment and relative kinematics
    to determine cabin operator state and context multiplier G_i(t).
    """
    def __init__(self, config: Optional[OperatorContextConfig] = None):
        self.config = config or OperatorContextConfig()

    def evaluate_operator_state(self, worker: TrackedObject,
                                machinery: List[TrackedObject]) -> Tuple[OperatorState, float, Optional[int]]:
        """
        Evaluates worker against all active machines.
        Returns:
            (OperatorState, G_i(t) multiplier, host_machine_id or None)
        """
        w_box = worker.bbox  # [x1, y1, x2, y2]
        w_area = max(1.0, (w_box[2] - w_box[0]) * (w_box[3] - w_box[1]))
        w_center = worker.center

        best_containment = 0.0
        best_machine_id = None
        best_state = OperatorState.PEDESTRIAN

        for m in machinery:
            m_box = m.bbox
            m_w = max(1.0, m_box[2] - m_box[0])
            m_h = max(1.0, m_box[3] - m_box[1])

            # Expected cabin region: upper-middle portion of machine bounding box
            rel_y1, rel_x1, rel_y2, rel_x2 = self.config.cabin_relative_box
            cabin_box = np.array([
                m_box[0] + rel_x1 * m_w,
                m_box[1] + rel_y1 * m_h,
                m_box[0] + rel_x2 * m_w,
                m_box[1] + rel_y2 * m_h
            ])

            # Calculate intersection between worker box and cabin region
            ix1 = max(w_box[0], cabin_box[0])
            iy1 = max(w_box[1], cabin_box[1])
            ix2 = min(w_box[2], cabin_box[2])
            iy2 = min(w_box[3], cabin_box[3])
            
            inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            containment_ratio = inter_area / w_area

            # Relative velocity test: operator moves with the machine!
            rel_vel = np.linalg.norm(worker.velocity - m.velocity)

            if containment_ratio > best_containment:
                best_containment = containment_ratio
                best_machine_id = m.track_id

                if containment_ratio >= self.config.cabin_containment_threshold:
                    if rel_vel <= self.config.max_relative_velocity or m.speed < 1.0:
                        best_state = OperatorState.CONFIRMED_OPERATOR
                    else:
                        best_state = OperatorState.UNCERTAIN
                elif containment_ratio >= 0.25:
                    best_state = OperatorState.UNCERTAIN
                else:
                    best_state = OperatorState.PEDESTRIAN

        # Context multiplier G_i(t)
        # Confirmed operator: suppressed (G_i = 0.0)
        # Uncertain: conservative! (G_i = 1.0, NEVER gets automatic safe status)
        # Pedestrian: standard alert (G_i = 1.0)
        if best_state == OperatorState.CONFIRMED_OPERATOR:
            multiplier = self.config.operator_multiplier
        else:
            multiplier = self.config.pedestrian_multiplier

        return best_state, multiplier, best_machine_id
