"""
AquaSense AI - Flask Backend Server
Serves the AquaSense frontend and provides water potability ML prediction and analysis APIs.
Supports 4 models: XGBoost, Random Forest, Decision Tree, Logistic Regression.

The Potable / Not Potable verdict and its probability come straight from the selected
ML model. WHO/EPA guideline checks are reported alongside as warnings, a compliance
score and treatment recommendations; they never change the model's verdict.
"""

import os
import json
import math
import uuid
import datetime
import joblib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Configuration & Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "aquasense_artifacts")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config['JSON_SORT_KEYS'] = False
if hasattr(app, "json"):
    app.json.sort_keys = False
CORS(app)

# Feature definitions aligned with model training pipeline
FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

# ========================================================
# SINGLE SOURCE OF TRUTH: WHO / EPA REFERENCE GUIDELINES
# ========================================================
# critical_*    -> a breach is flagged "critical" (strictly beyond the limit)
# acceptable_*  -> beyond this the parameter is "acceptable" (marginal) but not a breach
# The frontend receives this dict through /api/metadata, so thresholds are defined once.
WATER_STANDARDS = {
    "ph": {
        "name": "pH Level", "short_name": "pH Level", "unit": "pH", "feature": "ph",
        "aliases": ["ph", "pH", "PH"], "decimals": 2,
        "critical_min": 6.5, "critical_max": 8.5,
        "acceptable_min": 6.8, "acceptable_max": 8.2,
        "critical_penalty": 15, "acceptable_penalty": 8,
    },
    "hardness": {
        "name": "Hardness", "short_name": "Hardness", "unit": "mg/L", "feature": "Hardness",
        "aliases": ["hardness", "Hardness"], "decimals": 1,
        "critical_max": 330, "acceptable_max": 280,
        "critical_penalty": 15, "acceptable_penalty": 8,
        "critical_label": "Extreme Hardness (> 330 mg/L)",
    },
    "solids": {
        "name": "Total Solids (TDS)", "short_name": "Total Solids", "unit": "ppm", "feature": "Solids",
        "aliases": ["solids", "Solids"], "decimals": 0,
        "critical_max": 26000, "acceptable_max": 20000,
        "critical_penalty": 24, "acceptable_penalty": 10,
        "critical_label": "High Total Dissolved Solids (> 26,000 ppm)",
    },
    "chloramines": {
        "name": "Chloramines", "short_name": "Chloramines", "unit": "ppm", "feature": "Chloramines",
        "aliases": ["chloramines", "Chloramines"], "decimals": 2,
        "critical_max": 8.5, "acceptable_max": 7.5,
        "critical_penalty": 22, "acceptable_penalty": 8,
        "critical_label": "High Chloramines / Disinfection Excess (> 8.5 ppm)",
    },
    "sulfate": {
        "name": "Sulfate", "short_name": "Sulfate", "unit": "mg/L", "feature": "Sulfate",
        "aliases": ["sulfate", "Sulfate"], "decimals": 1,
        "critical_max": 450, "acceptable_max": 350,
        "critical_penalty": 22, "acceptable_penalty": 10,
        "critical_label": "Elevated Sulfates (> 450 mg/L)",
    },
    "conductivity": {
        "name": "Conductivity", "short_name": "Conductivity", "unit": "μS/cm", "feature": "Conductivity",
        "aliases": ["conductivity", "Conductivity"], "decimals": 1,
        "critical_max": 650, "acceptable_max": None,
        "critical_penalty": 14, "acceptable_penalty": 0,
        "critical_label": "High Electrical Conductivity (> 650 μS/cm)",
    },
    "organic": {
        "name": "Organic Carbon", "short_name": "Org. Carbon", "unit": "ppm", "feature": "Organic_carbon",
        "aliases": ["organic", "carbon", "Organic_carbon"], "decimals": 2,
        "critical_max": 20.0, "acceptable_max": 16.0,
        "critical_penalty": 16, "acceptable_penalty": 8,
        "critical_label": "Elevated Total Organic Carbon (> 20 ppm)",
    },
    "trihalo": {
        "name": "Trihalomethanes", "short_name": "Trihalomethanes", "unit": "μg/L", "feature": "Trihalomethanes",
        "aliases": ["trihalo", "thm", "Trihalomethanes"], "decimals": 2,
        "critical_max": 80.0, "acceptable_max": None,
        "critical_penalty": 18, "acceptable_penalty": 0,
        "critical_label": "Exceeds Trihalomethanes Limit (EPA 80 μg/L)",
    },
    "turbidity": {
        "name": "Turbidity", "short_name": "Turbidity", "unit": "NTU", "feature": "Turbidity",
        "aliases": ["turbidity", "Turbidity"], "decimals": 2,
        "critical_max": 5.0, "acceptable_max": 4.0,
        "critical_penalty": 22, "acceptable_penalty": 8,
        "critical_label": "High Turbidity / Cloudiness (> 5.0 NTU)",
    },
}

# Parameters grouped by the treatment that addresses them
RO_PARAMS = {"solids", "sulfate", "conductivity"}
GAC_PARAMS = {"turbidity", "organic", "trihalo"}

# Model keys, accepted aliases and friendly names
MODEL_ALIASES = {
    "xgboost": "xgboost", "xgb": "xgboost",
    "random_forest": "random_forest", "randomforest": "random_forest", "rf": "random_forest",
    "decision_tree": "decision_tree", "decisiontree": "decision_tree", "dt": "decision_tree",
    "logistic_regression": "logistic_regression", "logisticregression": "logistic_regression",
    "logistic": "logistic_regression", "lr": "logistic_regression",
}

MODEL_BASE_NAMES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
}

MODEL_NOTES = {
    "random_forest": "Ensemble Bagging",
    "decision_tree": "Rule-Based",
    "logistic_regression": "Linear Baseline",
}

# 1. LOAD TRAINED MODELS
MODELS = {}
model_file_map = {
    "xgboost": "xgboost.joblib",
    "random_forest": "random_forest.joblib",
    "decision_tree": "decision_tree.joblib",
    "logistic_regression": "logistic_regression.joblib"
}

def _smoke_test(pipeline):
    """Run one prediction so pickles from an incompatible scikit-learn/xgboost fail at startup, not per request."""
    sample = {"ph": 7.0, "Hardness": 190.0, "Solids": 20000.0, "Chloramines": 7.0, "Sulfate": 330.0,
              "Conductivity": 420.0, "Organic_carbon": 14.0, "Trihalomethanes": 66.0, "Turbidity": 4.0}
    pipeline.predict_proba(pd.DataFrame([sample]))


for key, filename in model_file_map.items():
    model_path = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(model_path):
        try:
            candidate = joblib.load(model_path)
            _smoke_test(candidate)
            MODELS[key] = candidate
            print(f"[AquaSense] Loaded model: {key} from {filename}")
        except Exception as e:
            print(f"[AquaSense] Error loading {key}: {type(e).__name__}: {e}")
            print(f"[AquaSense]   -> '{filename}' is unusable with the installed library versions. "
                  f"Install the versions in requirements.txt or retrain: python train_and_analyze.py")

# 2. LOAD METADATA AND ARTIFACTS
def load_json_artifact(filename, default=None):
    path = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[AquaSense] Error reading {filename}: {e}")
    return default

CANDIDATE_MODELS = load_json_artifact("candidate_models.json", {})
MODEL_METRICS = load_json_artifact("model_metrics.json", [])
ANALYSIS_SUMMARY = load_json_artifact("analysis_summary.json", {})
FEATURE_IMPORTANCES = load_json_artifact("feature_importance.json", {})
_CORRELATION_RAW = load_json_artifact("correlation_matrix.json", {})


def build_correlation(raw):
    """Feature-by-feature Pearson matrix (from the training script) plus each feature's correlation with Potability."""
    if not raw or "columns" not in raw:
        return {}
    cols, idx, data = raw["columns"], raw["index"], raw["data"]
    labels = [c for c in cols if c != "Potability"]
    value = lambda r, c: data[idx.index(r)][cols.index(c)]
    return {
        "labels": labels,
        "matrix": [[value(r, c) for c in labels] for r in labels],
        "with_potability": {f: value("Potability", f) for f in labels} if "Potability" in idx else {},
    }


CORRELATION = build_correlation(_CORRELATION_RAW)


def model_key_from_name(name):
    """'Random Forest' -> 'random_forest'."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def pick_champion():
    """
    Champion = highest cross-validated F1 among the loaded models, matching train_and_analyze.py
    (artifacts without CV fields fall back to test F1; with no metrics, any loaded model).
    """
    best_key, best_score = None, (-1.0, -1.0)
    for m in MODEL_METRICS:
        key = model_key_from_name(m.get("Model", ""))
        score = (m.get("CV-F1-Mean", m.get("F1-Score", 0)), m.get("CV-ROC-AUC-Mean", m.get("ROC-AUC", 0)))
        if key in MODELS and score > best_score:
            best_key, best_score = key, score
    if best_key is None:
        best_key = "xgboost" if "xgboost" in MODELS else next(iter(MODELS), "xgboost")
    return best_key


CHAMPION_KEY = pick_champion()


def display_name(model_key):
    base = MODEL_BASE_NAMES.get(model_key, model_key)
    if model_key == CHAMPION_KEY:
        return f"{base} (Champion · Best CV F1)"
    note = MODEL_NOTES.get(model_key)
    return f"{base} ({note})" if note else base


# ========================================================
# INPUT PARSING & VALIDATION
# ========================================================

def parse_inputs(data):
    """
    Extract the 9 parameters from the request payload.
    Returns (inputs, errors). A value of 0 is kept as 0; missing / non-numeric /
    non-finite / negative values (and pH outside 0-14) are reported as errors.
    """
    inputs, errors = {}, {}
    for key, std in WATER_STANDARDS.items():
        raw = None
        for alias in std["aliases"]:
            if data.get(alias) is not None:
                raw = data[alias]
                break
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            errors[key] = "This value is required."
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[key] = "Must be a number."
            continue
        if not math.isfinite(value):
            errors[key] = "Must be a finite number."
        elif value < 0:
            errors[key] = "Cannot be negative."
        elif key == "ph" and value > 14:
            errors[key] = "pH must be between 0 and 14."
        else:
            inputs[key] = value
    return inputs, errors


def resolve_model_key(raw):
    """Map a user-supplied model name to a model key, or None if unknown."""
    if raw is None or str(raw).strip() == "":
        return CHAMPION_KEY
    normalized = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return MODEL_ALIASES.get(normalized)


# ========================================================
# WHO / EPA GUIDELINE EVALUATION (informational - never changes the ML verdict)
# ========================================================

def classify_parameter(key, value):
    """Return (status, violation_label) for one parameter using WATER_STANDARDS."""
    std = WATER_STANDARDS[key]
    crit_min = std.get("critical_min")
    below = crit_min is not None and value < crit_min
    if below or value > std["critical_max"]:
        if key == "ph":
            label = "Acidic pH (< 6.5)" if below else "Alkaline pH (> 8.5)"
        else:
            label = std["critical_label"]
        return "critical", label

    acc_min, acc_max = std.get("acceptable_min"), std.get("acceptable_max")
    if (acc_min is not None and value < acc_min) or (acc_max is not None and value > acc_max):
        return "acceptable", None
    return "optimal", None


def evaluate_parameters(inputs):
    """
    Evaluate individual water parameters against WHO/EPA reference guidelines.
    Returns (violations, param_status_list, score, flagged_keys).
    """
    violations = []
    param_status_list = []
    flagged = set()
    score = 94

    for key, std in WATER_STANDARDS.items():
        value = inputs[key]
        status, label = classify_parameter(key, value)
        if status == "critical":
            violations.append(label)
            flagged.add(key)
            if key == "ph":
                score -= min(35, std["critical_penalty"] + abs(7.0 - value) * 10)
            else:
                score -= std["critical_penalty"]
        elif status == "acceptable":
            score -= std["acceptable_penalty"]
        param_status_list.append({
            "key": key,
            "name": std["short_name"],
            "value": round(value, std["decimals"]),
            "unit": std["unit"],
            "status": status,
        })

    ph, chloramines = inputs["ph"], inputs["chloramines"]
    # Synergistic chemical hazard penalties
    if chloramines > 7.2 and (ph < 6.8 or ph > 8.2):
        violations.append("Chloramine / pH Equilibrium Disruption")
        flagged.add("ph")
        score -= 10
    if inputs["solids"] > 22000 and inputs["sulfate"] > 360:
        violations.append("Synergistic Mineral Salinity (TDS + Sulfate)")
        flagged.update({"solids", "sulfate"})
        score -= 12
    if inputs["organic"] > 15 and inputs["trihalo"] > 65:
        violations.append("Disinfection Byproduct Precursor Hazard (TOC + THM)")
        flagged.update({"organic", "trihalo"})
        score -= 14

    # Round half up (like JavaScript's Math.round) so the offline estimate in app.js matches exactly;
    # Python's round() would round 62.5 down to 62.
    score = max(5, min(98, int(math.floor(score + 0.5))))
    return violations, param_status_list, score, flagged


def generate_recommendations(is_potable, violations, flagged):
    """Generate treatment and action recommendations from the ML verdict and guideline checks."""
    recommendations = []

    if is_potable and not violations:
        recommendations.append({
            "title": "Conforms to Reference Guidelines",
            "desc": "The model classifies this sample as potable and all parameters conform to WHO & EPA reference thresholds (laboratory confirmation advised).",
            "icon": "task_alt",
            "type": "success"
        })
        recommendations.append({
            "title": "Routine Disinfection Monitoring",
            "desc": "Maintain residual disinfectant between 0.2 and 4.0 ppm to prevent secondary bacterial contamination during storage.",
            "icon": "sanitizer",
            "type": "info"
        })
        return recommendations

    if is_potable:
        recommendations.append({
            "title": "Guideline Warning · Model Predicts Potable",
            "desc": f"The model predicts this sample is potable, but {len(violations)} WHO/EPA guideline threshold(s) are exceeded. Confirm with laboratory testing before consumption.",
            "icon": "warning",
            "type": "warning"
        })
    else:
        desc = (
            f"The model classifies this sample as not potable and {len(violations)} guideline threshold(s) are exceeded. Treatment recommended before consumption."
            if violations
            else "The model classifies this sample as not potable, although no single guideline threshold is exceeded. Laboratory testing and filtration recommended."
        )
        recommendations.append({
            "title": "Screening Alert · Corrective Treatment Advised",
            "desc": desc,
            "icon": "dangerous",
            "type": "danger"
        })

    if flagged & RO_PARAMS:
        recommendations.append({
            "title": "Reverse Osmosis (RO) Demineralization",
            "desc": "Total Dissolved Solids, sulfates or conductivity exceed guideline levels. Demineralize using multi-stage RO membranes.",
            "icon": "filter_alt",
            "type": "warning"
        })
    if flagged & GAC_PARAMS:
        recommendations.append({
            "title": "Coagulation & Activated Carbon Filtering",
            "desc": "Reduce suspended colloidal particles and organic carbon precursors with granulated activated carbon (GAC) filtering.",
            "icon": "layers",
            "type": "warning"
        })
    if "ph" in flagged:
        recommendations.append({
            "title": "pH Neutralization Buffering",
            "desc": "Inject stabilizing buffering agents (calcite/soda ash) to return pH to the reference 6.5 – 8.5 range.",
            "icon": "tune",
            "type": "warning"
        })
    return recommendations


# ========================================================
# FRONTEND STATIC ROUTES
# (all other files in frontend/ are served by Flask's static folder)
# ========================================================

@app.route("/")
@app.route("/index.html")
@app.route("/main.html")
def serve_main():
    return send_from_directory(FRONTEND_DIR, "main.html")

@app.route("/predict.html")
def serve_predict():
    return send_from_directory(FRONTEND_DIR, "predict.html")

@app.route("/analysis.html")
def serve_analysis():
    return send_from_directory(FRONTEND_DIR, "analysis.html")

@app.route("/result.html")
def serve_result():
    return send_from_directory(FRONTEND_DIR, "result.html")


# ========================================================
# API ENDPOINTS
# ========================================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint showing loaded models and status."""
    return jsonify({
        "status": "online" if MODELS else "degraded",
        "service": "AquaSense AI Water Quality Prediction System",
        "version": "2.5.0",
        "models_available": list(MODELS.keys()),
        "champion_model": CHAMPION_KEY
    })

@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Returns candidate model benchmarks, water standards and active model metadata."""
    champ_metrics = {}
    champ_name = MODEL_BASE_NAMES.get(CHAMPION_KEY, CHAMPION_KEY)
    for m in MODEL_METRICS:
        if model_key_from_name(m.get("Model", "")) == CHAMPION_KEY:
            champ_name = m.get("Model", champ_name)
            champ_metrics = {
                "accuracy": m.get("Accuracy"),
                "precision": m.get("Precision"),
                "recall": m.get("Recall"),
                "f1": m.get("F1-Score"),
                "roc_auc": m.get("ROC-AUC"),
                "cv_f1_mean": m.get("CV-F1-Mean"),
                "cv_f1_std": m.get("CV-F1-Std")
            }
            break

    champ_info = CANDIDATE_MODELS.get(champ_name, {})
    return jsonify({
        "success": True,
        "metadata": {
            "model_name": champ_name,
            "model_key": CHAMPION_KEY,
            "model_display_name": champ_info.get("display_name", champ_name),
            "metrics": champ_metrics,
            "candidate_models": CANDIDATE_MODELS,
            "feature_names": FEATURES,
            "standards": WATER_STANDARDS
        }
    })

@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns all 4 models and their evaluation metrics."""
    return jsonify({
        "success": True,
        "models": CANDIDATE_MODELS,
        "benchmark_table": MODEL_METRICS,
        "features": FEATURES,
        "feature_importances": FEATURE_IMPORTANCES
    })

@app.route("/api/analysis", methods=["GET"])
def get_analysis():
    """Returns full dataset exploratory data analysis and correlation."""
    return jsonify({
        "success": True,
        "summary": ANALYSIS_SUMMARY,
        "feature_importances": FEATURE_IMPORTANCES,
        "correlation": CORRELATION
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Accepts 9 physicochemical parameters and a model selection. The verdict and probability
    come from the ML model; WHO/EPA guideline checks are returned as supporting information.
    """
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            data = request.form.to_dict()
        if not data:
            return jsonify({"success": False, "error": "No input payload provided"}), 400

        inputs, errors = parse_inputs(data)
        if errors:
            return jsonify({"success": False, "error": "Invalid input", "errors": errors}), 400

        model_key = resolve_model_key(data.get("model"))
        if model_key is None:
            return jsonify({
                "success": False,
                "error": f"Unknown model '{data.get('model')}'.",
                "valid_models": sorted(set(MODEL_ALIASES.values()))
            }), 400

        pipeline = MODELS.get(model_key)
        if pipeline is None:
            return jsonify({"success": False, "error": f"Model '{model_key}' is not loaded on the server."}), 503

        # Prepare DataFrame matching pipeline column names
        row = pd.DataFrame([{WATER_STANDARDS[k]["feature"]: inputs[k] for k in WATER_STANDARDS}])

        # ML model inference - this is the verdict
        prediction = int(pipeline.predict(row)[0])
        probs = pipeline.predict_proba(row)[0]
        potable_prob = float(probs[list(pipeline.classes_).index(1)])
        is_potable = prediction == 1

        potable_pct = int(round(potable_prob * 100))
        risk_pct = 100 - potable_pct
        confidence = potable_pct if is_potable else risk_pct

        # WHO & EPA guideline evaluation (informational)
        violations, param_status_list, guideline_score, flagged = evaluate_parameters(inputs)
        recommendations = generate_recommendations(is_potable, violations, flagged)

        return jsonify({
            "success": True,
            "engine": "ml",
            "prediction": prediction,
            "label": "Potable" if is_potable else "Not Potable",
            "probability": round(potable_prob, 4),
            "model_name": display_name(model_key),
            "model_key": model_key,
            "diagnostic": {
                "is_potable": is_potable,
                "score": guideline_score,
                "guideline_violations_count": len(violations),
                "confidence": confidence,
                "risk": risk_pct,
                "probabilities": {
                    "potable": potable_pct,
                    "non_potable": risk_pct
                }
            },
            "inputs": inputs,
            "params": inputs,
            "parameters": param_status_list,
            "violations": violations,
            "flags": violations,
            "recommendations": recommendations,
            "disclaimer": "This prediction is an ML-based screening aid and should not replace certified laboratory water testing.",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_id": f"RUN-{uuid.uuid4().hex[:6].upper()}"
        })

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"success": False, "error": "Internal server error while computing the prediction."}), 500


# ========================================================
# SERVER ENTRY POINT
# ========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n========================================================")
    print("  AquaSense AI Flask Server Running")
    print(f"  Open in Browser: http://127.0.0.1:{port}")
    print(f"  Active Models: {', '.join(MODELS.keys())}  (champion: {CHAMPION_KEY})")
    print("========================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
