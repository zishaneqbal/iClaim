# 🚗 iClaim: Vehicle Damage Intelligence

<div align="center">

### **Autonomous CNN Damage Classification • Explainable Grad-CAM • Actuarial Claim Analytics**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-MobileNetV2-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Pytest-80%20Passed-10b981?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

<br/>

> **Decision-Support Tool for Motor Insurance Operations**: Accelerates preliminary claim triaging and damage validation from vehicle photographs while providing explainable AI heatmaps and actuarial repair estimates.

[⚡ Quick Start](#-quick-start) • [📸 How It Works](#-how-it-works) • [📊 Dashboard Analytics](#-management-dashboard) • [🧠 ML Pipeline](#-machine-learning-architecture) • [🧪 Testing](#-test-suite)

</div>

---

## 🌟 Overview & Architecture

**iClaim** bridges the gap between deep computer vision and insurance claims processing. Rather than acting as a black-box model, it combines:
1. **Computer Vision Head**: MobileNetV2 transfer learning trained on multi-class vehicle panel damage.
2. **Explainable AI (XAI)**: Grad-CAM gradient saliency maps pinpointing where the model focused.
3. **Automated Actuarial Triage**: Multi-signal severity scoring (class prior + confidence + activation area ratio) mapped into automated claim priority routing (Low / Medium / High / Manual Review).
4. **Segment-Aware Cost Estimation**: Indicative INR repair cost benchmarking across 8 vehicle segments (from Two-Wheelers to Luxury EVs and Heavy Commercial).

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                   iClaim Operational Vision Pipeline                   │
   └────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       [ 📸 Vehicle Image ] ──► (Quality & Blur Gate) ──► [ Laplacian Variance >= 60 ]
                                    │
                                    ▼
                        [ MobileNetV2 CNN Head ]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        { Class Probabilities }             { Convolutional Saliency }
        (Scratch, Dent, Crack, Glass, Body)           │
                  │                                   ▼
                  │                         [ Grad-CAM Heatmap ]
                  │                         (Activated Area Ratio)
                  └─────────────────┬─────────────────┘
                                    │
                                    ▼
                    [ ⚖️ Rule-Based Severity Engine ]
                    (Class Prior + Confidence + Area)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
       [ 🎯 Claim Priority ]               [ 💰 Indicative Repair Cost ]
     (Low / Medium / High / Review)       (Base Class Range x Vehicle Multiplier)
                                    │
                                    ▼
                   [ 📊 Operations Management Dashboard ]
```

---

## 🚀 Key Features

| Capability | Technical Details | Operational Impact |
| :--- | :--- | :--- |
| 📸 **Damage Classification** | Deep MobileNetV2 transfer learning trained across 5 critical damage types. | Instant damage identification in **<25 ms** per photograph. |
| 🔬 **Grad-CAM Explainability** | Layer-by-layer gradient activation tracking with JET colormap alpha overlay. | Transparent inspection: claims adjusters can see exactly which panels triggered the classification. |
| 🛡️ **Optical QA & Blur Gate** | Laplacian variance edge-filtering rejects blurry or corrupt images (<60 score). | Prevents hallucinated or unconfident assessments on degraded photos. |
| ⚖️ **Severity Triage** | Multi-factor weighted formulation deriving Minor, Moderate, or Severe ratings. | Standardizes repair complexity evaluation across adjuster teams. |
| 🎯 **Workflow Priority Routing** | Rule-driven routing: Low, Medium, High, or Mandatory Manual Review. | Fast-tracks minor scratch claims while routing ambiguous or severe claims to senior appraisers. |
| 💰 **Vehicle-Segment Costs** | Actuarial multiplier matrix across 8 vehicle segments (Hatchbacks to Heavy Commercial). | Generates realistic, segment-adjusted INR repair exposure ranges. |
| 📊 **Modern Telemetry Dashboard** | Multi-tab Plotly interface with rounded caps, drilldown Sunburst, and timelines. | Provides portfolio executives with frequency, risk exposure, and intake velocity insights. |

---

## 🏷️ Damage Taxonomy & Actuarial Matrix

### 1. Recognized Damage Classes
| Category | Typical Impact Area | Base Severity | Operational Routing |
| :--- | :--- | :--- | :--- |
| **Scratch** | Clear-coat and surface paint abrasions | Minor | Fast-Track Auto Approval |
| **Dent** | Sheet metal and panel deformations | Moderate | PDR / Body Panel Repair |
| **Crack** | Structural plastics, bumpers, fiber composites | Moderate | Component Replacement |
| **Broken Glass** | Windshields, windows, lamp assemblies | Moderate | Glazing Fast-Track |
| **Major Body Damage** | Crumple-zone, chassis, multi-panel structural collapse | Severe | Physical Adjuster Inspection |

### 2. Vehicle Segment Multiplier Matrix
| Segment | Multiplier | Benchmark Description |
| :--- | :---: | :--- |
| 🏍️ **Two-Wheeler / Bike** | `0.45x` | Compact fairing and exterior panel repairs |
| 🚗 **Hatchback** | `0.85x` | Standard entry-level passenger bodywork |
| 🚙 **Sedan** | `1.00x` | Baseline actuarial benchmark |
| 🛞 **SUV** | `1.25x` | Higher ground clearance and reinforced panels |
| 🚚 **Commercial / Light Truck** | `1.35x` | Fleet bodywork and utility panels |
| ⚡ **Electric Vehicle (EV)** | `1.45x` | Specialized sensor calibration and high-voltage safety protocols |
| 🚛 **Heavy Commercial** | `1.85x` | Heavy chassis and commercial fabrication |
| 💎 **Luxury / Premium** | `2.20x` | OEM aluminum alloys, specialized paint, and advanced ADAS |

---

## 🛠 Tech Stack

```
Backend & Machine Learning:
  • Python 3.13            - Core runtime
  • TensorFlow 2.15 & Keras- Deep learning backbone & custom preprocessing layers
  • MobileNetV2            - ImageNet-pretrained transfer head
  • OpenCV & Pillow        - Computer vision, blur analysis, and heatmap overlay
  • NumPy & Pandas         - Tensor manipulation and tabular claim analytics
  • PyYAML                 - Centralized configuration management

Frontend & Visual Telemetry:
  • Streamlit 1.40+        - Interactive web application framework
  • Plotly 5.18+           - Interactive charts with custom HTML tooltips and sunburst drilldowns
  • Custom CSS Animations  - Automotive HUD reticle brackets, laser scanlines, and glowing LED pills

Quality & Engineering:
  • Pytest                 - 80 automated unit & regression test cases
  • GitHub Actions CI      - Automated multi-OS build and test verification
```

---

## 📂 Project Structure

```
iClaim/
├── app/
│   ├── app.py                 # Main application portal (Hero & system overview)
│   ├── pages/
│   │   ├── 1_Claim.py         # Image intake, CNN forward pass, HUD & Grad-CAM
│   │   └── 2_Dashboard.py     # 4-tab management telemetry dashboard
│   └── components/
│       ├── charts.py          # Plotly visualizations (Sunburst, timeline, cost bars)
│       └── ui.py              # Telemetry CSS, HUD inspection frames, and metric cards
├── src/
│   ├── config.py              # YAML config loader (single source of truth)
│   ├── data_loader.py         # Dataset preparation, stratified splits & class weights
│   ├── model.py               # MobileNetV2, ResNet50, and EfficientNet architectures
│   ├── train.py               # Two-stage transfer learning (head + fine-tuning)
│   ├── evaluate.py            # Test set evaluation and metrics reporter
│   ├── predict.py             # Inference pipeline with error isolation
│   ├── gradcam.py             # Feature map gradient computation & heatmap overlay
│   ├── severity.py            # Rule-based multi-signal severity assessment
│   ├── analytics.py           # Claim logging, priority rules, and cost estimation
│   └── utils.py               # Laplacian blur validation and image helpers
├── tests/                     # 80-test automated suite (no ImageNet download required)
├── models/                    # Trained model weights (.keras) and class_names.json
├── data/                      # Dataset directories and synthetic claim registers
├── reports/                   # Training curves, confusion matrices, and results.md
├── config.yaml                # Single source of truth configuration
├── requirements.txt           # Python package dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/zishaneqbal/iClaim.git
cd iClaim

# Create and activate a virtual environment
py -m venv .venv
.venv\Scripts\Activate.ps1    # On Windows
# source .venv/bin/activate   # On macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application

```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`. The application runs immediately using synthetic portfolio data!

### 3. Model Training & Evaluation (Optional)

If you wish to train the MobileNetV2 transfer head locally:

```bash
# 1. Prepare raw images into stratified splits (train/val/test)
python -m src.data_loader --prepare

# 2. Train the CNN model (saves weights to models/best_model.keras)
python -m src.train --config config.yaml

# 3. Evaluate metrics on the held-out test split
python -m src.evaluate
```

---

## 📸 Application Walkthrough

### 1. Home Command Center (`app.py`)
- **Cybernetic Hero Banner**: Displays real-time model status via pulsing LED indicator, backbone architecture specs, and live project stats.
- **Quick-Launch Action Cards**: Direct routing into Claim Assessment or the Portfolio Dashboard.
- **Inspection Pipeline Flow**: Interactive 4-stage visual breakdown of the end-to-end AI assessment lifecycle.
- **Damage Taxonomy Explorer**: Visual cards explaining the 5 recognized damage categories and handling protocols.

### 2. Claim Assessment Bay (`1_Claim.py`)
- **AI Vision Inspection HUD**: Uploaded images and Grad-CAM attention heatmaps are displayed in cybernetic camera viewports with animated holographic scanning lines and reticle brackets.
- **Confidence & Blur Telemetry**: Automatic Laplacian sharpness scoring warns if an image is blurry (<60).
- **Actuarial Summary Dossier**: Calculates indicative repair ranges in INR, checks claimed amounts against expected intervals, and assigns triage priority.

### 3. Operations Management Dashboard (`2_Dashboard.py`)
Organized into 4 interactive analysis tabs:
- **📊 Overview & Vehicle Mix**: Damage frequency, severity breakdown, vehicle segment distribution, and priority queues.
- **💰 Cost & Exposure Ranges**: Floating range bars showing average low, midpoint, and high repair costs with hover details.
- **📈 Intake Velocity & Timeline**: Dual-axis spline chart displaying weekly claim intake volume alongside cumulative financial exposure in INR.
- **🔬 Interactive Hierarchy (Sunburst)**: Multi-level drilldown (`Vehicle Segment ➔ Damage Type ➔ Severity Level`). Click any segment to zoom in; click the center to zoom out.

---

## 🧪 Test Suite

The comprehensive test suite runs without requiring pre-trained weights or GPU downloads:

```bash
pytest tests -v
```

---

## ⚠️ Limitations & Governance

- **Decision-Support Scope**: Outputs are preliminary recommendations. They are intended to augment, not replace, licensed insurance adjusters.
- **Indicative Actuarial Assumptions**: Repair cost estimates in INR are benchmark approximations based on standard industry distributions, not binding repairer quotes.
- **Optical Variations**: Severe occlusions, extreme lighting glare, or non-vehicle inputs can affect prediction confidence; such instances are automatically routed to **Manual Review**.
- **Data Privacy**: The demonstration interface operates offline and does not transmit customer or policyholder data to external cloud services.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**iClaim** · Vehicle Damage Intelligence & Claim Analytics

[⬆ Back to Top](#-iclaim-vehicle-damage-intelligence)

</div>
