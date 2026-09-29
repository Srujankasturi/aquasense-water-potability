# AquaSense AI - Intelligent Water Potability Prediction System

AquaSense AI is a full-stack, machine learning-powered water quality screening platform designed to assess the safety and drinkability of water samples based on 9 core physicochemical parameters conforming to **WHO & EPA** drinking water quality guidelines.

---

## Key Features

- **Trained Machine Learning Benchmark**: Evaluated across 4 distinct candidate models:
  - **Random Forest Classifier (Active Champion)**: 400-tree bagging ensemble with balanced class weights. Chosen automatically as the model with the highest cross-validated F1 (`0.5432 ± 0.0247`, 5-fold on the training split), also the best CV ROC-AUC (`0.6938`).
  - **XGBoost Classifier**: Gradient-boosted trees with cost-sensitive class weighting; best single-split test F1 (`0.5388`) but a lower cross-validated F1 (`0.5213 ± 0.0234`).
  - **Decision Tree Classifier**: Hierarchical interpretable rule partitioner (`max_depth = 8`).
  - **Logistic Regression**: Standardized linear baseline.
- **ML Verdict + Guideline Checks**: The Potable / Not Potable verdict and its probability come straight from the selected trained model. WHO/EPA guideline checks run alongside as warnings, a 0-100 guideline score and treatment recommendations; they never change the model's verdict. If the model says potable but a threshold is breached, the report says so explicitly.
- **Modern Full-Stack Dashboard**:
  - `main.html`: Live interactive simulation rig, instant preset samples, and responsive circular probability gauges.
  - `predict.html`: Full 9-parameter submission with live parameter validation badges and modal diagnostics.
  - `analysis.html`: 4-Model benchmark comparison cards, confusion matrix ($N=656$), and dataset EDA.
  - `result.html`: Parameter audit, WHO/EPA compliance tables, and prescribed purification steps (RO, Carbon Filtration, pH Neutralization).
- **Labelled Offline Mode**: If the backend cannot be reached, `frontend/app.js` shows a rules-only estimate built from the same WHO/EPA thresholds, clearly marked "Offline estimate (rules only - not ML)". Invalid input is rejected by the API (HTTP 400) rather than silently replaced by defaults.

---

## Evaluated Model Benchmark (5-fold CV on the training split; test columns = held-out set, $N = 656$)

| Model Architecture | CV F1 (mean ± std) | Accuracy | Precision | Recall (Potable) | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| XGBoost | 0.5213 ± 0.0234 | 65.55% | 56.41% | 51.56% | 0.5388 | 0.6646 | Candidate |
| **Random Forest** | **0.5432 ± 0.0247** | **64.33%** | **55.24%** | **45.31%** | **0.4979** | **0.6630** | 🟢 **Active Model** |
| Decision Tree | 0.4466 ± 0.0563 | 64.48% | 59.35% | 28.52% | 0.3852 | 0.6099 | Candidate |
| Logistic Regression | 0.4155 ± 0.0283 | 52.44% | 41.46% | 53.12% | 0.4658 | 0.5474 | Baseline |


### How the champion is chosen
`train_and_analyze.py` runs 5-fold stratified cross-validation on the **training split only** and picks the model with the highest mean F1 (ties broken by mean ROC-AUC). The held-out test set plays no part in that choice and is scored exactly once, so the test columns above are an unbiased estimate. The gaps between the top models are small (about one standard deviation), and everything is limited by a noisy dataset (about 65% accuracy), so treat the ranking as indicative rather than definitive. The frontend reads the champion from `/api/metadata`, so retraining updates the whole app.

---

## Physicochemical Parameters Evaluated

1. **pH Level** (breach if < 6.5 or > 8.5; marginal outside 6.8 - 8.2)
2. **Hardness** (breach > 330 mg/L; marginal > 280)
3. **Total Dissolved Solids (TDS)** (breach > 26,000 ppm; marginal > 20,000)
4. **Chloramines** (breach > 8.5 ppm; marginal > 7.5)
5. **Sulfate** (breach > 450 mg/L; marginal > 350)
6. **Electrical Conductivity** (breach > 650 uS/cm)
7. **Total Organic Carbon (TOC)** (breach > 20 ppm; marginal > 16)
8. **Trihalomethanes (THM)** (breach > 80 ug/L, EPA limit)
9. **Turbidity** (breach > 5.0 NTU; marginal > 4.0)

These thresholds are defined once, in `WATER_STANDARDS` in `app.py`; the frontend receives them from `/api/metadata`.

---

## Project Structure

```
AquaSense/
├── app.py                      # Flask backend API server & static file host
├── train_and_analyze.py        # Modular ML training, EDA, and evaluation pipeline
├── requirements.txt            # Python package dependencies
├── requirements-dev.txt        # Test dependencies (pytest)
├── run.bat                     # Windows quick launch script
├── tests/                      # pytest suite for the API and guideline logic
├── .gitignore                  # Git ignore rules (.venv, cache, etc.)
│
├── aquasense_artifacts/        # Serialized models and evaluation data
│   ├── water_potability_model.joblib   # Active champion model (highest CV F1)
│   ├── xgboost.joblib
│   ├── random_forest.joblib
│   ├── decision_tree.joblib
│   ├── logistic_regression.joblib
│   ├── candidate_models.json
│   ├── model_metrics.json
│   ├── analysis_summary.json
│   └── feature_importance.json
│
├── dataset/
│   └── water_potability.csv    # 3,276 water quality records
│
└── frontend/                   # Web application interface
    ├── main.html               # Home dashboard & quick simulator
    ├── predict.html            # Parameter prediction form
    ├── analysis.html           # Model telemetry & benchmark comparisons
    ├── result.html             # Detailed diagnostic report & recommendations
    ├── app.js                  # Frontend client engine & API connector
    ├── theme.css               # "Hydrographic chart" design tokens (light + dark), atmosphere, components, motion
    ├── tailwind-config.js      # Maps Tailwind token names to the CSS variables in theme.css
    ├── DESIGN.md               # Design system: palette, type, motion, accessibility rules
    └── assets/contours.svg     # Generated contour-line artwork (CSS mask)
```

---

## Frontend Design

The interface is designed as a **hydrographic chart**: warm chart paper and ink by day, a "night survey" dark theme by night (it follows the OS setting and the toggle in the header), with contour-line atmosphere, double-ruled plates and one signal accent for breaches. Fonts are Young Serif, Familjen Grotesk and Sometype Mono. All colours are CSS variables, contrast is checked at 4.5:1 or better in both themes, and motion respects `prefers-reduced-motion`. See `frontend/DESIGN.md` for the full system and how to change it.

---

## Live Demo & Deployment

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Srujankasturi/aquasense-water-potability)

The repo ships a `render.yaml` blueprint (free web service, `gunicorn app:app`, health check at `/api/health`). Click the button, sign in with GitHub, and confirm; Render builds and serves the app at a public `onrender.com` URL. On the free tier the service sleeps when idle, so open it a minute before presenting.

Any other host works the same way: install `requirements.txt` and run `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4`.

---

## Getting Started

### 1. Prerequisites
- Python 3.11+ installed on your system.

### 2. Setup Virtual Environment & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/Srujankasturi/aquasense-water-potability.git
cd aquasense-water-potability

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Application
- **Windows (Double-click)**: Simply run `run.bat`.
- **Command Line**:
  ```bash
  python app.py
  ```
- Open your browser to **`http://127.0.0.1:5000`**.

### 4. Retrain Models (Optional)
To retrain and evaluate all 4 candidate models:
```bash
python train_and_analyze.py
```

### 5. Run the Tests
```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

> **Model files and library versions:** the `.joblib` files in `aquasense_artifacts/` are pickles and only load with the scikit-learn / xgboost versions they were trained with (`requirements.txt`). On startup the server verifies every model and logs a clear message if one is unusable (`/api/health` then reports `degraded`). On a newer Python / library stack, retrain with `python train_and_analyze.py`; the benchmark numbers below will shift slightly with the library versions.

---

## API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/predict` | `POST` | ML verdict + probability, WHO/EPA guideline checks, score and recommendations (HTTP 400 with per-field errors on invalid input) |
| `/api/metadata` | `GET` | Returns champion model metadata (chosen by cross-validated F1)
| `/api/models` | `GET` | Returns 4-model benchmark comparisons and feature importances |
| `/api/analysis` | `GET` | Returns dataset exploratory analysis summary |
| `/api/health` | `GET` | Server health status |

---


