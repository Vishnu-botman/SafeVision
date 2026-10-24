# SafeVision: Context-Aware PPE & Worker–Machinery Proximity Monitoring
### Object Detection, Multi-Object Tracking, and Temporal Risk Scoring for Dynamic Construction Sites

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Detector](https://img.shields.io/badge/YOLO-v8n-green.svg)](https://ultralytics.com/)
[![Dashboard](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Course:** Deep Learning Term Project  
**Institution:** Amrita Vishwa Vidyapeetham, Amrita School of Computing  
**Authors:**
- **Lerin Liju** (`AM.SC.U4AIE24029`)
- **Vishnu** (`AM.SC.U4AIE24058`)
- **Jeevan Krishna** (`AM.SC.U4AIE24059`)

---

## 📌 Overview & Problem Statement

Visual safety monitoring on dynamic construction job sites is critical to preventing fatal worker–machinery collisions (struck-by and caught-between hazards) and enforcing mandatory Personal Protective Equipment (PPE) compliance.

However, conventional camera-based safety systems suffer from critical structural gaps:
1. **Isolated Frame-Level Detections:** They detect workers, helmets, vests, and machinery as disjoint boxes with no temporal identity. A helmet detected in the frame does not guarantee every worker is protected.
2. **Spurious Operator False Alarms:** A machine driver seated inside an excavator cabin is geometrically located within the machine perimeter, causing naive proximity alarms to sound continuously.
3. **Static Danger Radii:** Traditional fixed circles ignore vehicle speed, heading, and stopping distance.
4. **Alarm Fatigue:** Single-frame boundary crossings and transient occlusions produce deafening jitter and false warnings.

**SafeVision** resolves these challenges by converting raw visual detections into **worker-specific, explainable, and temporally persistent safety decisions**.

---

## 🚀 Key Innovations & System Architecture

```mermaid
flowchart TD
    A[Video Stream / CCTV] --> B[Model A: Scene Detector\nWorkers & Heavy Machinery]
    A --> C[Model B: PPE Detector\nHelmets, Vests, Heads]
    B & C --> D[Detection Fusion & Class-Specific NMS]
    D --> E[ByteTrack Multi-Object Tracking\n7-State Kalman Filter]
    E --> F[Anatomical PPE Association\nTorso & Head Regions + 7-Frame Voting]
    E --> G[Motion-Adaptive Danger Zones\nDirectional Apex Expansion]
    E --> H[Cabin Operator Gate G_i(t)\nContainment + Velocity Filtering]
    F & G & H --> I[Temporal Risk Engine R_i(t)\nHysteresis: 3-Up / 5-Down]
    I --> J[Real-time Streamlit HUD & Structured Event Log]
```

### 1. Dual-Detector Perception Core
- **Model A (Scene Detector):** Localizes workers and 6 heavy machinery classes (excavator, dump truck, loader, crane, bulldozer, roller).
- **Model B (PPE Detector):** Fine-grained detector specialized for helmets, vests, and uncovered heads ($mAP@0.5 = 85.4\%$).
- **Detection Fusion:** Greedy class-specific Non-Maximum Suppression ($\theta_{iou} = 0.45$).

### 2. Pure-Python ByteTrack Kinematics
- 7-dimensional Kalman state vector: $\mathbf{x} = [c_x, c_y, s, r, v_x, v_y, v_s]^T$.
- Two-stage Hungarian matching associates high-confidence detections first, then recovers occluded objects from low-confidence detections ($0.20 \le s < 0.50$).

### 3. Worker-to-PPE Association & Temporal Voting
- Upper-body and torso anatomical bounding box containment tests.
- Rolling $K=7$ frame temporal consensus eliminates single-frame dropouts:
  $$P_i(t) = \min\Big(1.0, \; 0.60 \cdot \mathbb{I}(\neg \text{has\_helmet}) + 0.40 \cdot \mathbb{I}(\neg \text{has\_vest})\Big)$$

### 4. Motion-Adaptive Danger Zones
- Dynamically expands along the machine velocity vector $\vec{v}_m$:
  $$\mathbf{P}_{\text{apex}} = \mathbf{c}_m + \hat{\mathbf{v}}_m \cdot \Big(\max(w_m, h_m) \cdot 0.70 + \|\vec{v}_m\| \cdot \kappa \cdot 15.0\Big)$$
- Evaluates worker ground contact via ray-casting.

### 5. Cabin Operator Context Gate $G_i(t)$
- Tests worker containment within machine cabin windows ($[0.05h, 0.20w, 0.55h, 0.80w]$) and relative velocity ($\|\vec{v}_w - \vec{v}_m\| \le 2.0\text{ px/f}$).
- $G_i(t) = 0.0$ for verified cabin drivers, suppressing 100% of driver false alarms.

### 6. Temporal Risk Metric & Multi-Stage Hysteresis
$$R_i(t) = G_i(t) \cdot \Big[ w_z Z_i(t) + w_m M_i(t) + w_t T_i(t) + w_p P_i(t) \Big]$$
- Weights: Zone penetration ($w_z=0.35$), Machine approach rate ($w_m=0.25$), Exposure duration ($w_t=0.20$), PPE penalty ($w_p=0.20$).
- Exponential smoothing: $\tilde{R}_i(t) = 0.65 R_i(t) + 0.35 \tilde{R}_i(t-1)$.
- Hysteresis: Requires 3 consecutive frames to escalate alert (`SAFE` $\to$ `WARNING` $\to$ `CRITICAL`), and 5 consecutive frames to de-escalate.

---

## 📊 Empirical Ablation Results

Evaluated over canonical construction interaction edge-case benchmarks:

| Metric / System Property | Baseline System (Static & Frame-Only) | Proposed SafeVision (Context & Adaptive) | Relative Gain |
| :--- | :---: | :---: | :--- |
| **Cabin Operator Gate $G_i(t)$** | None (Constant False Alarm) | Cabin Containment Gated | **100% suppression of driver false alarms** |
| **Danger Zone Geometry** | Fixed Circle ($120\text{px}$) | Motion-Adaptive Dynamic Polygon | **Anticipates movement vector** |
| **Temporal Decision Logic** | Frame-Instantaneous | Accumulation + Hysteresis | **Dampens 1-frame boundary jitters** |
| **False Alarm Rate (%)** | **48.04%** | **23.53%** | **$-24.51\%$ absolute reduction** |
| **Event Detection Precision** | 0.165 | **0.231** | **$+0.066$ improvement** |
| **Event Detection Recall** | 0.615 | **0.731** | **$+0.116$ improvement** |
| **Event Detection F1-Score** | **0.260** | **0.347** | **$+33.5\%$ relative boost** |
| **Spurious Alert Transitions** | 4 transitions | **3 transitions** | **25% smoother temporal stability** |

---

## 📁 Repository Structure

```text
├── app.py                             # Interactive Streamlit Web Dashboard
├── DL_Term_Project_Final_Report.md    # Complete Markdown Final Report
├── DL_Term_project_report_template/   # IEEE Conference LaTeX Template (main.tex, bibil.bib)
├── Train_SafeVision_Colab.ipynb       # Google Colab GPU Training Notebook
├── train_ppe.py                       # Automated YOLO PPE Model Training Script
├── prepare_shwd_yolo.py               # SHWD / Pascal VOC to YOLO Format Converter
├── safevision/                        # Core Python Package
│   ├── benchmark/                     # Synthetic 5-scenario evaluation generator
│   ├── config.py                      # Global dataclass configurations
│   ├── context/                       # Danger zones, PPE association & cabin gate
│   ├── data/                          # Dataset auditing & harmonization
│   ├── engine/                        # Risk engine & structured event logger
│   ├── eval/                          # Quantitative ablation benchmark
│   ├── models/                        # DualDetectorPipeline, weights & fusion
│   ├── pipeline.py                    # End-to-end video processing pipeline
│   ├── tracking/                      # Pure-Python ByteTrack tracker
│   └── visualization/                 # HUD rendering & risk overlays
└── tests/
    └── test_safevision.py             # Complete test suite (100% passing)
```

---

## 🛠️ Quickstart & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Vishnu-botman/SafeVision.git
cd SafeVision
pip install -r requirements.txt
```
*(Dependencies: `torch`, `ultralytics`, `streamlit`, `scipy`, `pandas`, `pillow`, `numpy`)*

### 2. Launch the Interactive Web Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
Explore the interactive HUD, test scenario playback, tune risk weights live, and download CSV safety logs.

### 3. Run Pipeline Verification Tests
```bash
python tests/test_safevision.py
```

### 4. Train PPE Detector on Google Colab
Open [`Train_SafeVision_Colab.ipynb`](Train_SafeVision_Colab.ipynb) in Google Colab to fine-tune YOLOv8 on the harmonized PPE dataset with free GPU acceleration.

---

## 📜 Academic Report & Citation
The complete IEEE conference paper source is available in [`DL_Term_project_report_template/main.tex`](DL_Term_project_report_template/main.tex).
