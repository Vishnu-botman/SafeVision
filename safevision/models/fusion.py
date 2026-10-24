"""
Frame-Level Detection Fusion and Cross-Detector Conflict Resolution
Performs Non-Maximum Suppression (NMS) and harmonizes dual-detector outputs.
"""

from typing import Dict, List, Tuple
import numpy as np


def compute_iou(b1: List[float], b2: List[float]) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / max(1e-6, union)


def nms(detections: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
    """Applies standard greedy Non-Maximum Suppression within identical class groups."""
    if not detections:
        return []

    # Sort descending by score
    sorted_dets = sorted(detections, key=lambda d: d['score'], reverse=True)
    kept = []

    while sorted_dets:
        chosen = sorted_dets.pop(0)
        kept.append(chosen)
        remaining = []
        for d in sorted_dets:
            if d['class_name'] == chosen['class_name']:
                if compute_iou(chosen['bbox'], d['bbox']) < iou_thresh:
                    remaining.append(d)
            else:
                remaining.append(d)
        sorted_dets = remaining

    return kept


class DetectionFusion:
    """
    Harmonizes overlapping detections across detectors:
    1. Removes duplicate worker / person boxes.
    2. Resolves PPE overlapping boundaries.
    """
    def __init__(self, iou_thresh: float = 0.45):
        self.iou_thresh = iou_thresh

    def fuse(self, worker_dets: List[Dict],
             machinery_dets: List[Dict],
             ppe_dets: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        fused_workers = nms(worker_dets, self.iou_thresh)
        fused_machinery = nms(machinery_dets, self.iou_thresh)
        fused_ppe = nms(ppe_dets, self.iou_thresh)
        return fused_workers, fused_machinery, fused_ppe
