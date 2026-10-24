# Context-Aware PPE and Worker–Machinery Proximity Monitoring: Object Detection, Tracking, and Temporal Risk Scoring

**Authors:** Lerin Liju (`AM.SC.U4AIE24029`), Vishnu (`AM.SC.U4AIE24058`), and Jeevan Krishna (`AM.SC.U4AIE24059`)  
**Affiliation:** Department of Computer Science and Engineering (Artificial Intelligence), Amrita School of Computing, Amrita Vishwa Vidyapeetham, Amritapuri, India  
**Course:** Deep Learning Term Project  
**LaTeX Submission Template:** `DL_Term_project_report_template/main.tex` & `bibil.bib`

---

## Abstract
**Context:** Visual safety monitoring in dynamic industrial and construction environments is critical for preventing fatal worker-machinery collisions and enforcing Personal Protective Equipment (PPE) compliance.  
**Gap:** Current camera-based deep learning solutions operate primarily as isolated, frame-level object detectors. They treat workers, machines, helmets, and vests as independent bounding boxes without explicit identity assignment. Consequently, they fail to associate safety gear with individual personnel, ignore cabin-operator context (triggering continuous false proximity alerts for vehicle drivers), employ static circular hazard radii that neglect equipment kinematics, and suffer from high alarm jitter caused by transient detection dropouts.  
**Approach:** In this work, we propose **SafeVision**, an end-to-end framework integrating dual-detector inference, ByteTrack multi-object tracking, anatomical worker-to-PPE geometric association, motion-adaptive image-space danger zones, and an operator-aware temporal risk engine. Individual worker risk $R_i(t)$ is computed by fusing danger-zone penetration, relative machine approach motion, temporal exposure accumulation, and PPE non-compliance, governed by an operator context gate $G_i(t)$ and multi-stage state hysteresis.  
**Results:** Empirical ablation over systematic interaction scenarios demonstrates that SafeVision reduces the false alert rate from 48.04% (baseline) to 23.53% (a 24.51% absolute reduction), eliminates 100% of spurious cabin-driver proximity alerts, increases the event detection F1-score from 0.260 to 0.347, and reduces alert jitter transitions by 25%.  
**Conclusion:** By converting disjoint visual detections into worker-specific, kinematically aware, and temporally persistent safety decisions, SafeVision establishes a robust foundation for practical, deployable automated construction safety management.

---

## 1. Introduction
Visual automated surveillance powered by modern deep convolutional and transformer-based neural networks has emerged as a cornerstone of industrial occupational safety. On construction sites and heavy manufacturing facilities, the close coexistence of ground personnel and heavy moving machinery (such as excavators, wheel loaders, and cranes) creates severe hazard vectors, frequently resulting in fatal struck-by or caught-between accidents. Concurrently, mandatory compliance with Personal Protective Equipment (PPE)---specifically hardhats and high-visibility safety vests---remains a vital defense against occupational trauma.

Despite significant advances in generic object detection, deploying vision-based safety monitoring in real-world construction environments remains impeded by fundamental structural limitations. First, conventional vision systems perform isolated detection on individual video frames without maintaining temporal track identity. A helmet detected somewhere within a crowded frame does not guarantee that every worker is protected, nor does it identify which specific individual is non-compliant. Second, existing systems lack operator spatial awareness: a machine operator seated inside an excavator cabin is geometrically within the perimeter of the machinery bounding box, causing naive proximity systems to trigger deafening, false alarms continuously. Third, danger zones are conventionally represented as fixed, static circular radii, failing to account for vehicle velocity, heading, and stopping distances. Finally, instantaneous frame-level alert logic produces extreme alarm fatigue due to momentary visual occlusions and single-frame boundary crossings.

To overcome these research gaps, this paper introduces **SafeVision**, a context-aware temporal surveillance architecture that converts multi-class visual detections into structured, worker-specific safety events. SafeVision decouples perception into a dual-detector core (a Scene Detector for heavy machinery and workers, and a fine-grained PPE Detector for safety gear), unifies their detections via multi-object tracking, performs anatomical region-based PPE association, dynamically stretches hazard zones in the direction of machine travel, and computes an explainable, time-accumulated risk score $R_i(t)$ with hysteresis smoothing.

The primary contributions of this work are:
1. **Dual-Detector Fusion and Tracking Pipeline:** Integrating a coarse scene detector (trained on MOCS ontologies) with a specialized PPE detector (trained on harmonized CHV and SHWD ontologies), coupled with a pure-Python ByteTrack tracker estimating kinematic velocity vectors for both workers and equipment.
2. **Worker-Specific PPE Association with Temporal Voting:** Upper-body and torso anatomical bounding box containment tests coupled with a rolling 7-frame temporal consensus window, dampening transient occlusion noise.
3. **Motion-Adaptive Danger Zones:** Image-space hazard polygons that dynamically expand along the observed directional velocity vector of moving equipment, providing earlier warning for frontal approaches while reducing false alarms behind turning machinery.
4. **Operator-Aware Temporal Risk Engine:** Formalizing a unified risk metric $R_i(t) = G_i(t) [w_z Z_i + w_m M_i + w_t T_i + w_p P_i]$ incorporating cabin operator gating $G_i(t)$, exposure duration accumulation $T_i$, and multi-stage hysteresis.
5. **Comprehensive Ablation and Interactive Dashboard:** Demonstrating an empirical false alarm reduction of 24.51% and packaging the entire system into an open-source, interactive Streamlit demonstration dashboard.

---

## 2. Literature Review
Computer vision for construction safety has progressed rapidly from hand-crafted feature extractors to modern deep neural networks. Early approaches by Wu et al. (2019) utilized color histograms and hierarchical Support Vector Machines (H-SVM) to identify hardhats, subsequently transitioning to Faster R-CNN architectures. While pioneering, multi-stage pipelines were computationally prohibitive for real-time edge processing and struggled under harsh site illuminations.

The release of large-scale construction domain benchmarks substantially catalyzed progress. An et al. (2021) established the Moving Objects in Construction Sites (MOCS) benchmark, encompassing 41,668 images across 174 construction sites, comparing 15 object detection architectures. Their study confirmed that YOLO-style single-stage detectors provide optimal speed-accuracy trade-offs for heavy machinery and worker localization. However, MOCS is annotated solely with coarse bounding boxes, omitting PPE details, cross-frame identities, and proximity interactions.

In the PPE compliance domain, Wang et al. (2021) introduced the Construction Hardhat and Vest (CHV) dataset containing 1,330 images, achieving an 86.55% mAP using YOLOv5. Ahmad and Rahimi (2024) presented the SH17 dataset with 8,099 images spanning 17 manufacturing PPE categories, evaluating YOLOv9 models. Similarly, Liu et al. (2025) introduced specialized attention mechanisms (YOLOv10-IYA) for small helmet detection in heavy-machinery factories. Nath et al. (2020) introduced Pictor-PPE, demonstrating a unified YOLOv3 model detecting workers, helmets, and vests at 11 FPS.

Despite these achievements, a critical disconnect persists:
1. Prior works count safety gear globally per frame rather than assigning equipment to individual workers.
2. Proximity alert literature uniformly relies on static circles or 2D Euclidean distance thresholds, failing to account for equipment motion direction.
3. Workers located inside machinery cabins are routinely flagged as external pedestrians in danger.
4. Existing systems trigger alerts on single-frame threshold crossings, leading to intolerable false alarm frequencies.

SafeVision directly addresses these limitations by uniting multi-class detection with multi-object tracking, cabin-operator classification, motion-adaptive geometry, and temporal risk accumulation.

---

## 3. Methodology

### 3.1 Dataset Curation & Ontologies
- **MOCS:** Retains worker instances and 6 heavy machinery classes (excavator, dump truck, wheel loader, crane, bulldozer, roller) evaluated on 4,000 validation images partitioned by site.
- **CHV:** 1,330 multi-scale images harmonized from original color classes (Blue, Red, White, Yellow helmets) into a unified `helmet` class, with full-body `person` and torso `vest`.
- **SHWD:** 15,171 images in Pascal VOC format covering `hat` and `head_uncovered` classes under challenging occlusions.

### 3.2 Dual-Detector Perception Core
- **Model A (Scene Detector):** Predicts bounding boxes for workers and heavy equipment ($s \ge 0.40$).
- **Model B (PPE Detector):** Predicts bounding boxes for helmets, vests, and heads ($s \ge 0.35$).
- **Frame-Level Fusion:** Merges detections and applies class-specific greedy Non-Maximum Suppression ($\theta_{iou} = 0.45$).

### 3.3 Multi-Object Tracking & Kinematics (ByteTrack)
Employs pure-Python Kalman filter tracking:
$$\mathbf{x} = [c_x, c_y, s, r, v_x, v_y, v_s]^T$$
Two-stage Hungarian matching associates high-confidence detections ($s \ge 0.50$) first, then associates remaining low-confidence detections ($0.20 \le s < 0.50$) to maintain track continuity through occlusion.

### 3.4 Worker-to-PPE Association & Temporal Voting
- **Head Region:** $y \in [y_1, y_1 + 0.35h]$, $x \in [x_1 - 0.1w, x_2 + 0.1w]$
- **Torso Region:** $y \in [y_1 + 0.20h, y_1 + 0.75h]$, $x \in [x_1 - 0.1w, x_2 + 0.1w]$
- **Temporal Consensus:** Sliding-window vote over the last $K = 7$ frames to eliminate single-frame dropouts:
  $$P_i(t) = \min\Big(1.0, \; 0.60 \cdot \mathbb{I}(\neg \text{has\_helmet}) + 0.40 \cdot \mathbb{I}(\neg \text{has\_vest})\Big)$$

### 3.5 Motion-Adaptive Danger Zones
Hazard zones dynamically expand along the machine's velocity vector $\vec{v}_m$:
$$\mathbf{P}_{\text{apex}} = \mathbf{c}_m + \hat{\mathbf{v}}_m \cdot \Big(\max(w_m, h_m) \cdot 0.70 + \|\vec{v}_m\| \cdot \kappa \cdot 15.0\Big)$$
where $\kappa = 2.5$. Evaluates worker ground contact via ray-casting to determine $Z_i(t) \in [0.0, 1.0]$.

### 3.6 Cabin-Operator Context Gate $G_i(t)$
- Tests worker containment within the cabin window ($[0.05h, 0.20w, 0.55h, 0.80w]$) and relative velocity $\|\vec{v}_w - \vec{v}_m\| \le 2.0\text{ px/f}$.
- $G_i(t) = 0.0$ for confirmed cabin operators (suppressing false proximity alerts).
- $G_i(t) = 1.0$ for ground workers or uncertain cases (conservative safety rule).

### 3.7 Temporal Risk Formulation
$$R_i(t) = G_i(t) \cdot \Big[ w_z Z_i(t) + w_m M_i(t) + w_t T_i(t) + w_p P_i(t) \Big]$$
Weights: $w_z = 0.35, w_m = 0.25, w_t = 0.20, w_p = 0.20$.  
Temporal smoothing: $\tilde{R}_i(t) = 0.65 R_i(t) + 0.35 \tilde{R}_i(t-1)$.  
Hysteresis: Requires 3 consecutive frames to escalate alert (`SAFE` $\to$ `WARNING` $\to$ `CRITICAL`), and 5 consecutive frames to de-escalate.

---

## 4. Performance Evaluation & Empirical Ablation

| Metric / System Property | Baseline System (Static & Frame-Only) | Proposed SafeVision (Context & Adaptive) | Improvement |
| :--- | :---: | :---: | :--- |
| **Cabin Operator Gate $G_i(t)$** | None (Constant False Alarm) | Cabin Containment Gated | **100% suppression of driver false alarms** |
| **Danger Zone Geometry** | Fixed Circle ($120\text{px}$) | Motion-Adaptive Dynamic Polygon | **Anticipates movement direction** |
| **Temporal Decision Logic** | Frame-Instantaneous | Accumulation + Hysteresis | **Dampens 1-frame boundary jitters** |
| **False Alarm Rate (%)** | **48.04%** | **23.53%** | **-24.51% absolute reduction** |
| **Event Detection Precision** | 0.165 | **0.231** | **+0.066 improvement** |
| **Event Detection Recall** | 0.615 | **0.731** | **+0.116 improvement** |
| **Event Detection F1-Score** | **0.260** | **0.347** | **+0.087 improvement (+33.5% relative)** |
| **Spurious State Transitions** | 4 transitions | **3 transitions** | **25% smoother temporal stability** |

---

## 5. Conclusions
SafeVision resolves the critical limitations of isolated frame-level detectors in construction safety. By combining dual-detector perception, ByteTrack multi-object tracking, anatomical PPE association, motion-adaptive hazard zones, and cabin operator gating, the system reduces false alarms by 24.51% while increasing hazard event F1-score to 0.347. The complete software and interactive Streamlit web dashboard are deployed and fully verified.
