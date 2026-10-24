"""
SafeVision Visual Annotator
Renders translucent motion-adaptive danger zones, velocity vectors,
worker risk halos, PPE association links, and real-time HUD telemetry.
Built using PIL and NumPy with cv2 fallback support.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safevision.config import SystemConfig
from safevision.context.danger_zone import DangerZone
from safevision.context.operator_context import OperatorState
from safevision.context.ppe_association import WorkerPPEStatus
from safevision.engine.risk_engine import RiskLevel, WorkerRiskAssessment
from safevision.tracking.byte_track import TrackedObject


class VisualAnnotator:
    """
    Renders high-contrast, professional visual overlays on video frames.
    """
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # Color palette (RGB tuples for PIL)
        self.colors_rgb = {
            "safe": (46, 204, 113),          # Emerald Green
            "warning": (243, 156, 18),       # Amber / Orange
            "critical": (231, 76, 60),       # Ruby Red
            "danger_zone": (243, 156, 18),   # Orange
            "danger_zone_active": (231, 76, 60), # Red
            "worker": (52, 152, 219),        # Blue
            "machinery": (155, 89, 182),     # Purple
            "operator": (149, 165, 166),     # Muted Gray/Slate
            "helmet": (241, 196, 15),        # Yellow
            "vest": (46, 204, 113),          # Green
            "no_ppe": (231, 76, 60),         # Red
        }

    def annotate_frame(self,
                       frame_img: Image.Image,
                       workers: List[TrackedObject],
                       machinery: List[TrackedObject],
                       danger_zones: List[DangerZone],
                       ppe_statuses: Dict[int, WorkerPPEStatus],
                       assessments: Dict[int, WorkerRiskAssessment],
                       operator_states: Dict[int, Tuple[OperatorState, float, Optional[int]]],
                       frame_idx: int = 0,
                       show_danger_zones: bool = True,
                       show_ppe_boxes: bool = True,
                       show_velocity_vectors: bool = True,
                       show_hud: bool = True) -> Image.Image:
        """
        Overlays all detections, tracks, danger zones, and risk alerts onto image.
        Returns a newly annotated PIL Image.
        """
        # Ensure RGBA for alpha blending
        base = frame_img.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_base = ImageDraw.Draw(base)

        width, height = base.size

        # 1. Draw Motion-Adaptive Danger Zones (Translucent Polygon)
        if show_danger_zones:
            for dz in danger_zones:
                # Check if any worker is occupying this danger zone with alert
                is_active = any(
                    asm.associated_machine_id == dz.machine_id and asm.occupancy_Z >= 0.40
                    for asm in assessments.values()
                )
                zone_color = self.colors_rgb["danger_zone_active"] if is_active else self.colors_rgb["danger_zone"]
                alpha_fill = 70 if is_active else 40
                
                poly_pts = [(float(pt[0]), float(pt[1])) for pt in dz.polygon]
                if len(poly_pts) >= 3:
                    # Fill polygon on overlay
                    draw_overlay.polygon(poly_pts, fill=(zone_color[0], zone_color[1], zone_color[2], alpha_fill))
                    # Draw outline
                    draw_base.polygon(poly_pts, outline=zone_color + (255,), width=2)

        # 2. Draw Machinery Tracks & Velocity Vectors
        for m in machinery:
            bbox = m.bbox
            m_color = self.colors_rgb["machinery"]
            draw_base.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=m_color, width=3)
            
            # Label
            label = f"{m.class_name.upper()} #{m.track_id}"
            if m.speed > 0.5:
                label += f" | {m.speed:.1f} px/f"
            draw_base.rectangle([bbox[0], bbox[1] - 22, bbox[0] + len(label) * 8 + 12, bbox[1]], fill=m_color)
            draw_base.text((bbox[0] + 6, bbox[1] - 20), label, fill=(255, 255, 255))

            # Velocity vector arrow
            if show_velocity_vectors and m.speed > 0.8:
                cx, cy = m.center
                vx, vy = float(m.velocity[0]), float(m.velocity[1])
                arrow_len = min(120.0, m.speed * 12.0)
                unit_v = np.array([vx, vy]) / max(1e-4, m.speed)
                end_x = cx + unit_v[0] * arrow_len
                end_y = cy + unit_v[1] * arrow_len
                draw_base.line([(cx, cy), (end_x, end_y)], fill=(255, 255, 0), width=3)
                # Small circle at tip
                draw_base.ellipse([end_x - 4, end_y - 4, end_x + 4, end_y + 4], fill=(255, 255, 0))

        # 3. Draw PPE Bounding Boxes and Worker Links
        if show_ppe_boxes:
            for tid, ppe_stat in ppe_statuses.items():
                if ppe_stat.helmet_bbox is not None:
                    hb = ppe_stat.helmet_bbox
                    draw_base.rectangle([hb[0], hb[1], hb[2], hb[3]], outline=self.colors_rgb["helmet"], width=2)
                    draw_base.text((hb[0], max(0, hb[1] - 12)), "HARDHAT", fill=self.colors_rgb["helmet"])

                if ppe_stat.vest_bbox is not None:
                    vb = ppe_stat.vest_bbox
                    draw_base.rectangle([vb[0], vb[1], vb[2], vb[3]], outline=self.colors_rgb["vest"], width=2)
                    draw_base.text((vb[0], max(0, vb[1] - 12)), "VEST", fill=self.colors_rgb["vest"])

        # 4. Draw Workers with Risk Halo and Telemetry
        for w in workers:
            tid = w.track_id
            bbox = w.bbox
            asm = assessments.get(tid, None)
            ppe_stat = ppe_statuses.get(tid, None)
            op_state, G_i, _ = operator_states.get(tid, (OperatorState.PEDESTRIAN, 1.0, None))

            # Determine box color based on Risk Level or Operator State
            if op_state == OperatorState.CONFIRMED_OPERATOR:
                box_color = self.colors_rgb["operator"]
                status_txt = "CABIN OPERATOR [SUPPRESSED]"
            elif asm:
                if asm.risk_level == RiskLevel.CRITICAL:
                    box_color = self.colors_rgb["critical"]
                    status_txt = f"CRITICAL RISK ({asm.smoothed_risk_score:.2f})"
                elif asm.risk_level == RiskLevel.WARNING:
                    box_color = self.colors_rgb["warning"]
                    status_txt = f"WARNING ({asm.smoothed_risk_score:.2f})"
                else:
                    box_color = self.colors_rgb["safe"]
                    status_txt = f"SAFE ({asm.smoothed_risk_score:.2f})"
            else:
                box_color = self.colors_rgb["worker"]
                status_txt = "MONITORING"

            # Draw worker bounding box
            border_w = 4 if (asm and asm.risk_level == RiskLevel.CRITICAL) else 2
            draw_base.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=box_color, width=border_w)

            # Translucent risk halo if critical
            if asm and asm.risk_level == RiskLevel.CRITICAL:
                draw_overlay.rectangle([bbox[0] - 6, bbox[1] - 6, bbox[2] + 6, bbox[3] + 6],
                                      fill=(231, 76, 60, 50))

            # Worker Header Tag
            header = f"WORKER #{tid} | {status_txt}"
            header_w = len(header) * 7 + 14
            draw_base.rectangle([bbox[0], bbox[1] - 22, bbox[0] + header_w, bbox[1]], fill=box_color)
            draw_base.text((bbox[0] + 6, bbox[1] - 20), header, fill=(255, 255, 255))

            # PPE Compliance Sub-Tag
            if ppe_stat and op_state != OperatorState.CONFIRMED_OPERATOR:
                h_sym = "YES" if ppe_stat.has_helmet else "NO"
                v_sym = "YES" if ppe_stat.has_vest else "NO"
                ppe_tag = f"PPE: Hat [{h_sym}] Vest [{v_sym}]"
                sub_y = bbox[3] + 4
                tag_w = len(ppe_tag) * 7 + 10
                
                # Tag background
                tag_bg = (50, 50, 50, 200)
                draw_base.rectangle([bbox[0], sub_y, bbox[0] + tag_w, sub_y + 18], fill=(40, 40, 40))
                draw_base.text((bbox[0] + 5, sub_y + 2), ppe_tag, fill=(255, 255, 255))

        # Composite overlay onto base image
        result = Image.alpha_composite(base, overlay).convert("RGB")
        draw_result = ImageDraw.Draw(result)

        # 5. Draw HUD Telemetry Banner
        if show_hud:
            banner_h = 44
            draw_result.rectangle([0, 0, width, banner_h], fill=(18, 24, 38))
            
            critical_count = sum(1 for a in assessments.values() if a.risk_level == RiskLevel.CRITICAL)
            warning_count = sum(1 for a in assessments.values() if a.risk_level == RiskLevel.WARNING)
            safe_count = sum(1 for a in assessments.values() if a.risk_level == RiskLevel.SAFE)

            hud_left = f"SafeVision Pro  |  Frame #{frame_idx}  |  Workers: {len(workers)}  |  Machinery: {len(machinery)}"
            draw_result.text((15, 14), hud_left, fill=(200, 220, 240))

            # Alert counters
            right_offset = width - 360
            draw_result.text((right_offset, 14), f"SAFE: {safe_count}", fill=(46, 204, 113))
            draw_result.text((right_offset + 90, 14), f"WARNING: {warning_count}", fill=(243, 156, 18))
            draw_result.text((right_offset + 210, 14), f"CRITICAL: {critical_count}", fill=(231, 76, 60))

        return result
