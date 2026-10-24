"""
Evaluation Metrics and Ablation Study Engine
Quantifies performance improvements of Proposed SafeVision vs. Baseline System
across False Alert Rate, Detection Precision/Recall/F1, and Alert Stability.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from safevision.benchmark.scenario_generator import BenchmarkScenarioGenerator, ScenarioStep
from safevision.engine.risk_engine import RiskLevel
from safevision.pipeline import SafeVisionPipeline


@dataclass
class EvaluationSummary:
    system_name: str
    total_frames: int
    true_critical_frames: int
    false_critical_frames: int
    true_safe_frames: int
    false_safe_frames: int
    precision: float
    recall: float
    f1_score: float
    false_alert_rate_pct: float
    spurious_transitions_count: int


class AblationBenchmark:
    """
    Executes automated comparative ablation between:
    1. Baseline (Static fixed radius, single-frame decisions, no cabin operator context)
    2. Proposed SafeVision (Motion-adaptive danger zone, temporal accumulation, cabin operator gate, hysteresis)
    """
    def __init__(self):
        self.generator = BenchmarkScenarioGenerator()

    def run_scenario(self, scenario_steps: List[ScenarioStep], use_baseline: bool) -> Tuple[List[Dict], EvaluationSummary]:
        pipeline = SafeVisionPipeline(use_baseline=use_baseline)
        pipeline.reset()

        frame_records = []
        prev_level = None
        spurious_transitions = 0

        tp = 0  # correctly raised alert when unsafe
        fp = 0  # raised alert when ground truth was safe
        tn = 0  # correctly stayed safe
        fn = 0  # stayed safe when ground truth was unsafe

        for step in scenario_steps:
            custom_dets = (step.worker_detections, step.machinery_detections, step.ppe_detections)
            out = pipeline.process_frame(step.image, custom_detections=custom_dets, render_annotation=False)

            # Determine dominant risk level in this frame
            highest_level = RiskLevel.SAFE
            highest_score = 0.0
            reasons = []

            for asm in out.assessments.values():
                if asm.smoothed_risk_score > highest_score:
                    highest_score = asm.smoothed_risk_score
                    highest_level = asm.risk_level
                    reasons = asm.active_reasons

            # Check transition instability
            if prev_level is not None and highest_level != prev_level:
                spurious_transitions += 1
            prev_level = highest_level

            # Ground truth classification
            # Ground truth is unsafe if GT is "CRITICAL_PROXIMITY" or "WARNING_PPE"
            gt_is_unsafe = step.ground_truth_event in ("CRITICAL_PROXIMITY", "WARNING_PPE")
            pred_is_unsafe = (highest_level in (RiskLevel.WARNING, RiskLevel.CRITICAL))

            if pred_is_unsafe and gt_is_unsafe:
                tp += 1
            elif pred_is_unsafe and not gt_is_unsafe:
                fp += 1
            elif not pred_is_unsafe and not gt_is_unsafe:
                tn += 1
            else:
                fn += 1

            frame_records.append({
                "frame_idx": step.frame_idx,
                "gt_event": step.ground_truth_event,
                "pred_level": highest_level.value,
                "pred_score": round(highest_score, 3),
                "reasons": "; ".join(reasons)
            })

        total = len(scenario_steps)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-6, prec + rec)
        false_alert_rate = (fp / max(1, fp + tn)) * 100.0

        summary = EvaluationSummary(
            system_name="Baseline System (Static & Frame-Only)" if use_baseline else "Proposed SafeVision (Context & Adaptive)",
            total_frames=total,
            true_critical_frames=tp,
            false_critical_frames=fp,
            true_safe_frames=tn,
            false_safe_frames=fn,
            precision=round(prec, 3),
            recall=round(rec, 3),
            f1_score=round(f1, 3),
            false_alert_rate_pct=round(false_alert_rate, 2),
            spurious_transitions_count=spurious_transitions
        )

        return frame_records, summary

    def run_full_ablation(self) -> pd.DataFrame:
        """
        Runs all 5 scenarios across both Baseline and Proposed models,
        aggregating comparative metrics.
        """
        all_steps = []
        all_steps.extend(self.generator.generate_scenario_1_safe_operation(25))
        all_steps.extend(self.generator.generate_scenario_2_ppe_violation(25))
        all_steps.extend(self.generator.generate_scenario_3_imminent_danger(35))
        all_steps.extend(self.generator.generate_scenario_4_cabin_operator(30))
        all_steps.extend(self.generator.generate_scenario_5_transient_crossing(25))

        # Re-index frame sequence
        for i, s in enumerate(all_steps):
            s.frame_idx = i + 1

        _, baseline_summary = self.run_scenario(all_steps, use_baseline=True)
        _, proposed_summary = self.run_scenario(all_steps, use_baseline=False)

        data = [
            {
                "Metric / Property": "Operator Awareness (Cabin G_i)",
                "Baseline System": "None (Always alerts)",
                "Proposed SafeVision": "Cabin Containment Gated",
                "Relative Gain": "Eliminates operator false alarms"
            },
            {
                "Metric / Property": "Danger Zone Geometry",
                "Baseline System": "Fixed Static Radius (Circle)",
                "Proposed SafeVision": "Motion-Adaptive Polygon",
                "Relative Gain": "Anticipates movement vector"
            },
            {
                "Metric / Property": "Temporal Decision Logic",
                "Baseline System": "Single-Frame Instantaneous",
                "Proposed SafeVision": "Accumulation + Hysteresis",
                "Relative Gain": "Dampens spurious jitter"
            },
            {
                "Metric / Property": "False Alarm Rate (%)",
                "Baseline System": f"{baseline_summary.false_alert_rate_pct}%",
                "Proposed SafeVision": f"{proposed_summary.false_alert_rate_pct}%",
                "Relative Gain": f"-{baseline_summary.false_alert_rate_pct - proposed_summary.false_alert_rate_pct:.1f}% reduction"
            },
            {
                "Metric / Property": "Event Detection F1-Score",
                "Baseline System": f"{baseline_summary.f1_score}",
                "Proposed SafeVision": f"{proposed_summary.f1_score}",
                "Relative Gain": f"+{proposed_summary.f1_score - baseline_summary.f1_score:.3f}"
            },
            {
                "Metric / Property": "Spurious Alert Transitions",
                "Baseline System": f"{baseline_summary.spurious_transitions_count} transitions",
                "Proposed SafeVision": f"{proposed_summary.spurious_transitions_count} transitions",
                "Relative Gain": f"-{baseline_summary.spurious_transitions_count - proposed_summary.spurious_transitions_count} fewer flickers"
            }
        ]

        return pd.DataFrame(data)
