# iClaim

Vehicle Damage Intelligence — CNN-based preliminary damage assessment with insurance claim analytics.

[![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/zishaneqbal/iClaim/ci.yml?label=CI)](.github/workflows/ci.yml)

> **Decision-support tool** for insurance staff to get a fast, consistent first look at vehicle damage photographs. Supports — and does not replace — professional inspection and final claim approval.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 📸 **Damage Classification** | CNN (MobileNetV2 transfer head) classifies damage type: scratch, dent, crack, broken glass, major body damage. |
| ⚖️ **Severity Assessment** | Rule-based severity scoring → Minor / Moderate / Severe, with Grad-CAM heatmaps for explainability. |
| 🎯 **Claim Priority** | Priority routing (Low / Medium / High / Manual review) based on severity + confidence. |
| 📊 **Management Dashboard** | Damage frequency, severity distribution, indicative repair-cost ranges (INR), and claim priorities. |
| 🧱 **Modular ML Pipeline** | `src/` package: data loading, training, evaluation, prediction, Grad-CAM, severity, analytics. |
| 🧪 **Tested** | Pytest suite covering data, model, Grad-CAM, severity, analytics, prediction, and image validation. |
| ⚙️ **Config-Driven** | All hyperparameters, paths, class names, cost ranges, and thresholds in `config.yaml`. |

---

## 🛠 Tech Stack

**Backend / ML**
- Python 3.13 · TensorFlow 2.15 · Keras · MobileNetV2 · scikit-learn · NumPy · Pandas · OpenCV · Pillow · PyYAML

**Frontend / Viz**
- Streamlit · Plotly · Matplotlib · Seaborn

**Testing / CI**
- Pytest · GitHub Actions

---

## 📂 Project Structure

```
iClaim/
├── app/
│   ├── app.py                 # Streamlit home page (hero + overview)
│   ├── pages/
│   │   ├── 1_Claim.py         # Upload photo + claim details → prediction
│   │   └── 2_Dashboard.py     # Management analytics view
│   └── components/
│       ├── __init__.py
│       ├── ui.py              # Shared UI components (cards, badges, section headers)
│       └── charts.py          # Plotly/Matplotlib charts for the dashboard
├── src/
│   ├── __init__.py
│   ├── config.py              # YAML config loader (single source of truth)
│   ├── data_loader.py         # Dataset audit, stratified split, class weights
│   ├── model.py               # MobileNetV2 transfer learning, save/load (.keras)
│   ├── train.py               # Head training + fine-tuning
│   ├── evaluate.py            # Test-set metrics + reports/results.md
│   ├── predict.py             # Image prediction + Grad-CAM overlay
│   ├── gradcam.py             # Grad-CAM heatmap generation
│   ├── severity.py            # Rule-based severity scoring
│   ├── analytics.py           # Claim aggregation, priority, cost ranges
│   └── utils.py               # Image validation, bytes/array helpers
├── tests/                     # Pytest suite (no ImageNet download needed)
├── .streamlit/
│   └── config.toml            # Streamlit theme / server settings
├── docs/
│   ├── DATASET_GUIDE.md       # Dataset download & preparation steps
│   ├── REPORT_OUTLINE.md      # Academic report outline
│   └── synopsis.md            # Project synopsis
├── models/                    # Trained models (gitignored; retrain locally)
├── data/                      # Datasets (raw/processed gitignored)
├── reports/                   # Evaluation results & figures
├── notebooks/                 # Exploration notebooks
├── config.yaml                # Central project configuration
├── requirements.txt           # Python dependencies
├── SECURITY.md                # Security policy
├── LICENSE
└── README.md
```

---

## 🗺 Roadmap

| Phase | Focus |
|-------|-------|
| 0 | Code scaffold (`src/`, `app/`, tests, docs) |
| 1 | Dataset audit, splits, baseline CNN |
| 2 | Transfer learning, tuning, Grad-CAM, metrics |
| 3 | Severity/priority tuning, app polish with real model |
| 4 | Report, slides, demo recording |

---

## ⚠️ Limitations & Disclaimer

- **Decision-support only.** Outputs are preliminary and must not be the sole basis for any insurance claim approval or settlement.
- Repair cost ranges are **indicative assumptions** (INR), not real insurer quotes.
- Severity is rule-based unless severity-labelled data is available.
- Sample claim data is **synthetic** and does not reflect real insurer data.
- Image quality (lighting, angle, occlusion, blur) affects predictions.
- The Streamlit app has **no authentication**; do not expose it publicly with real claim or customer data.

See [`SECURITY.md`](SECURITY.md) for the full security policy.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

## 🙋‍♂️ Acknowledgements

- TensorFlow / Keras team for MobileNetV2
- Streamlit for the rapid dashboard framework
- Plotly & Matplotlib for visualisations

---

<div align="center">

**iClaim** · Vehicle Damage Intelligence · Academic Project

[⬆ Back to top](#iclaim)

</div>
## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/zishaneqbal/iClaim.git
cd iClaim
```

### 2. Create a virtual environment & install dependencies

```bash
# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the app (works immediately with synthetic sample claims)

```bash
streamlit run app/app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

### 4. Enable image prediction (optional, needs a trained model)

The dashboard works with synthetic claims out of the box. To classify real vehicle photos:

```bash
# Prepare the dataset (see docs/DATASET_GUIDE.md for download steps)
python -m src.data_loader --prepare

# Train the model (writes models/best_model.keras)
python -m src.train

# Evaluate on the held-out test set
python -m src.evaluate
```

### 5. Run the test suite

```bash
pytest tests -q
```

---

## 📸 How It Works

### Claim page (`1_Claim.py`)
1. Upload a vehicle damage photograph (JPG/PNG/WebP, ≤10 MB).
2. Enter claim details (claim ID, vehicle type, policy type).
3. The CNN predicts the damage type and confidence.
4. Severity is scored (rule-based) and a Grad-CAM heatmap highlights the damage region.
5. A processing priority (Low / Medium / High / Manual review) is suggested.
6. Result is logged for analytics.

### Dashboard page (`2_Dashboard.py`)
- Damage frequency by class
- Severity distribution
- Indicative repair-cost ranges (INR)
- Claim priority breakdown
- Synthetic sample claims when no model is trained

---

## ⚙️ Configuration

All paths, hyperparameters, class names, cost ranges, and thresholds live in [`config.yaml`](config.yaml). Edit it instead of hard-coding values:

```yaml
data:
  class_names: [scratch, dent, crack, broken_glass, major_body_damage]
  image_size: [224, 224]

model:
  backbone: mobilenetv2
  dropout: 0.3

training:
  epochs_head: 10
  epochs_finetune: 10
  batch_size: 32

analytics:
  cost_ranges_inr:            # indicative ranges only
    scratch: { minor: [2000, 6000], ... }
  currency: INR
```

---

## 🧪 Testing

The test suite uses randomly initialised models (no ImageNet download required) and runs on every push via GitHub Actions.

```bash
pytest tests -q
```

| Test file | Covers |
|-----------|--------|
| `test_data_loader.py` | Stratified splits, duplicates, aliases, class weights |
| `test_model.py` | All backbones, freeze/unfreeze, `.keras` save/load |
| `test_gradcam.py` | Heatmap range, area ratio, overlay size |
| `test_severity.py` | Score range, thresholds, clamping |
| `test_analytics.py` | Priority rules, cost ranges, aggregations |
| `test_predict.py` | `predict()` contract, bytes/array inputs, overlay |
| `test_image_validation.py` | Extension, size, pixel limits, blur, grayscale/RGBA |

---
