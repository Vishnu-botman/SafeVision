"""
Motion-Adaptive Image-Space Danger Zone Generation
Dynamically computes danger zones around construction machinery that expand
in the observed direction of motion, contrasting with fixed static baselines.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from safevision.config import DangerZoneConfig
from safevision.tracking.byte_track import TrackedObject


@dataclass
class DangerZone:
    machine_id: int
    polygon: np.ndarray             # Nx2 array of polygon vertices [[x1, y1], [x2, y2], ...]
    centroid: Tuple[float, float]
    velocity: np.ndarray           # [vx, vy]
    speed: float
    is_expanding: bool
    expansion_angle_deg: float


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Standard ray-casting algorithm to test if point (x, y) is inside polygon."""
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Distance from point to a line segment."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return float(np.hypot(px - x1, py - y1))
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return float(np.hypot(px - proj_x, py - proj_y))


def point_to_polygon_distance(px: float, py: float, poly: np.ndarray) -> float:
    """Minimum Euclidean distance from point to polygon perimeter."""
    min_dist = float('inf')
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        d = point_to_segment_distance(px, py, x1, y1, x2, y2)
        if d < min_dist:
            min_dist = d
    return min_dist


class DangerZoneManager:
    """
    Constructs and evaluates motion-adaptive danger zones around machinery.
    Also provides fixed-radius static zones for baseline comparison.
    """
    def __init__(self, config: Optional[DangerZoneConfig] = None):
        self.config = config or DangerZoneConfig()

    def generate_adaptive_zone(self, machine: TrackedObject) -> DangerZone:
        """
        Creates an asymmetric danger zone that extends further in the direction
        of machine motion.
        """
        bbox = machine.bbox  # [x1, y1, x2, y2]
        w = max(10.0, bbox[2] - bbox[0])
        h = max(10.0, bbox[3] - bbox[1])
        cx, cy = machine.center
        vx, vy = float(machine.velocity[0]), float(machine.velocity[1])
        speed = float(np.hypot(vx, vy))

        # Base margin
        margin_x = w * self.config.base_margin_ratio
        margin_y = h * self.config.base_margin_ratio

        base_corners = [
            np.array([bbox[0] - margin_x, bbox[1] - margin_y]),  # Top-left
            np.array([bbox[2] + margin_x, bbox[1] - margin_y]),  # Top-right
            np.array([bbox[2] + margin_x, bbox[3] + margin_y]),  # Bottom-right
            np.array([bbox[0] - margin_x, bbox[3] + margin_y]),  # Bottom-left
        ]

        is_expanding = (speed >= self.config.min_speed_threshold)
        angle_deg = float(np.degrees(np.arctan2(vy, vx))) if is_expanding else 0.0

        if not is_expanding:
            poly = np.array(base_corners)
        else:
            # Expand vertices that face the direction of motion
            direction = np.array([vx, vy]) / max(speed, 1e-4)
            extension_length = min(300.0, speed * self.config.velocity_expansion_factor * 15.0)

            extended_points = []
            for corner in base_corners:
                # Vector from center to corner
                corner_vec = corner - np.array([cx, cy])
                proj = float(np.dot(corner_vec, direction))
                if proj > 0:
                    # Move corner along direction of motion
                    expanded_corner = corner + direction * extension_length
                    extended_points.append(expanded_corner)
                else:
                    extended_points.append(corner)

            # Insert an apex forward point for the danger cone
            lead_apex = np.array([cx, cy]) + direction * (max(w, h) * 0.7 + extension_length * 1.25)
            
            # Form convex hull or ordered polygon
            all_pts = np.vstack([base_corners, extended_points, [lead_apex]])
            # Sort points by polar angle around centroid
            center_pt = np.mean(all_pts, axis=0)
            angles = np.arctan2(all_pts[:, 1] - center_pt[1], all_pts[:, 0] - center_pt[0])
            sorted_indices = np.argsort(angles)
            poly = all_pts[sorted_indices]

        return DangerZone(
            machine_id=machine.track_id,
            polygon=poly,
            centroid=(cx, cy),
            velocity=np.array([vx, vy]),
            speed=speed,
            is_expanding=is_expanding,
            expansion_angle_deg=angle_deg
        )

    def generate_baseline_fixed_zone(self, machine: TrackedObject) -> DangerZone:
        """
        Baseline model: Fixed circular zone with constant radius, completely
        ignoring machine velocity and movement direction.
        """
        cx, cy = machine.center
        r = self.config.baseline_fixed_radius
        num_pts = 16
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        pts = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

        return DangerZone(
            machine_id=machine.track_id,
            polygon=pts,
            centroid=(cx, cy),
            velocity=machine.velocity,
            speed=float(np.linalg.norm(machine.velocity)),
            is_expanding=False,
            expansion_angle_deg=0.0
        )

    def evaluate_worker_occupancy(self, worker: TrackedObject, danger_zones: List[DangerZone]) -> Tuple[float, Optional[int]]:
        """
        Computes Z_i(t) in [0.0, 1.0]:
        - 1.0: Deep inside danger zone
        - 0.5: Exactly on perimeter
        - 0.0 - 0.49: Outside, approaching zone perimeter
        Returns:
            (Z_i(t), closest_machine_id)
        """
        if not danger_zones:
            return 0.0, None

        # Worker reference point: worker's ground contact location (bottom center of bbox)
        wx = float((worker.bbox[0] + worker.bbox[2]) / 2.0)
        wy = float(worker.bbox[3])

        max_occupancy = 0.0
        closest_mid = None

        for dz in danger_zones:
            is_inside = point_in_polygon(wx, wy, dz.polygon)
            dist_to_perim = point_to_polygon_distance(wx, wy, dz.polygon)

            if is_inside:
                # Penetration depth factor
                dist_to_center = float(np.hypot(wx - dz.centroid[0], wy - dz.centroid[1]))
                occupancy = min(1.0, 0.5 + 0.5 * (1.0 / (1.0 + 0.01 * dist_to_center)))
                occupancy = max(0.5, occupancy)
            else:
                # Proximity buffer: within 100 pixels approaches zone
                proximity_buffer = 100.0
                if dist_to_perim < proximity_buffer:
                    occupancy = 0.5 * (1.0 - (dist_to_perim / proximity_buffer))
                else:
                    occupancy = 0.0

            if occupancy > max_occupancy:
                max_occupancy = occupancy
                closest_mid = dz.machine_id

        return float(np.clip(max_occupancy, 0.0, 1.0)), closest_mid
