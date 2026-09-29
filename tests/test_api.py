"""
API and guideline-logic tests for AquaSense.

Run from the project root:  python -m pytest tests -q
Requires the trained artifacts in aquasense_artifacts/ to be loadable with the installed
scikit-learn / xgboost versions (see requirements.txt); if they were pickled by a different
version, retrain with `python train_and_analyze.py`.
"""

import os

import numpy as np
import pandas as pd
import pytest

import app as aquasense
import train_and_analyze as training

CLEAN = dict(ph=7.25, hardness=185, solids=14200, chloramines=6.8, sulfate=310,
             conductivity=420, organic=9.8, trihalo=58.0, turbidity=2.85)
INDUSTRIAL = dict(ph=4.80, hardness=380, solids=42000, chloramines=11.8, sulfate=520,
                  conductivity=890, organic=26.5, trihalo=124.0, turbidity=8.6)


@pytest.fixture()
def client():
    return aquasense.app.test_client()


def post(client, payload):
    return client.post("/api/predict", json=payload)


def model_row(inputs):
    return pd.DataFrame([{aquasense.WATER_STANDARDS[k]["feature"]: v for k, v in inputs.items()}])


class StubPipeline:
    """Minimal stand-in for a trained pipeline with a fixed output."""
    classes_ = np.array([0, 1])

    def __init__(self, potable_prob):
        self.p = potable_prob

    def predict(self, X):
        return np.array([1 if self.p >= 0.5 else 0])

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]])


# ---------------------------------------------------------------- input handling

def test_zero_value_is_kept_not_replaced_by_default(client):
    r = post(client, {**CLEAN, "turbidity": 0, "ph": 0})
    assert r.status_code == 200
    inputs = r.get_json()["inputs"]
    assert inputs["turbidity"] == 0.0
    assert inputs["ph"] == 0.0


@pytest.mark.parametrize("field,value", [
    ("ph", "abc"), ("ph", 15), ("ph", -1), ("solids", -5), ("turbidity", "nan"), ("sulfate", "inf"), ("hardness", ""),
])
def test_invalid_values_return_400_with_field_error(client, field, value):
    r = post(client, {**CLEAN, field: value})
    body = r.get_json()
    assert r.status_code == 400
    assert body["success"] is False
    assert field in body["errors"]


def test_missing_fields_are_reported(client):
    r = post(client, {"ph": 7})
    assert r.status_code == 400
    assert set(r.get_json()["errors"]) == set(aquasense.WATER_STANDARDS) - {"ph"}


def test_unknown_model_returns_400(client):
    r = post(client, {**CLEAN, "model": "neural_net"})
    assert r.status_code == 400
    assert "valid_models" in r.get_json()


def test_empty_payload_returns_400(client):
    assert client.post("/api/predict", json={}).status_code == 400


def test_model_aliases_and_default(client):
    assert post(client, {**CLEAN, "model": "RF"}).get_json()["model_key"] == "random_forest"
    assert post(client, {**CLEAN, "model": "Decision Tree"}).get_json()["model_key"] == "decision_tree"
    assert post(client, CLEAN).get_json()["model_key"] == aquasense.CHAMPION_KEY


def test_alternate_field_names_are_accepted(client):
    payload = {"pH": 7.0, "Hardness": 180, "Solids": 15000, "Chloramines": 7, "Sulfate": 300,
               "Conductivity": 400, "Organic_carbon": 10, "Trihalomethanes": 60, "Turbidity": 3}
    assert post(client, payload).status_code == 200


# ---------------------------------------------------------------- ML is the verdict

@pytest.mark.parametrize("model_key", list(aquasense.MODELS))
@pytest.mark.parametrize("sample", [CLEAN, INDUSTRIAL], ids=["clean", "industrial"])
def test_verdict_and_probability_come_from_the_model(client, model_key, sample):
    pipeline = aquasense.MODELS[model_key]
    row = model_row(sample)
    expected_pred = int(pipeline.predict(row)[0])
    expected_prob = float(pipeline.predict_proba(row)[0][list(pipeline.classes_).index(1)])

    body = post(client, {**sample, "model": model_key}).get_json()
    assert body["engine"] == "ml"
    assert body["prediction"] == expected_pred
    assert body["label"] == ("Potable" if expected_pred else "Not Potable")
    assert body["probability"] == pytest.approx(round(expected_prob, 4))
    assert body["diagnostic"]["probabilities"]["potable"] == int(round(expected_prob * 100))
    assert body["diagnostic"]["probabilities"]["potable"] + body["diagnostic"]["probabilities"]["non_potable"] == 100


def test_guidelines_never_override_the_model_verdict(client, monkeypatch):
    monkeypatch.setitem(aquasense.MODELS, "xgboost", StubPipeline(0.80))
    body = post(client, {**INDUSTRIAL, "model": "xgboost"}).get_json()
    assert body["label"] == "Potable"                       # model says potable...
    assert body["probability"] == 0.8                       # ...with its own, unblended probability
    assert body["diagnostic"]["guideline_violations_count"] > 0
    titles = [r["title"] for r in body["recommendations"]]
    assert titles[0].startswith("Guideline Warning")        # ...but the breaches are still surfaced
    assert "Reverse Osmosis (RO) Demineralization" in titles
    assert "pH Neutralization Buffering" in titles


def test_model_not_potable_without_violations(client, monkeypatch):
    monkeypatch.setitem(aquasense.MODELS, "xgboost", StubPipeline(0.30))
    body = post(client, {**CLEAN, "model": "xgboost"}).get_json()
    assert body["label"] == "Not Potable"
    assert body["violations"] == []
    assert body["diagnostic"]["confidence"] == 70
    assert "no single guideline threshold" in body["recommendations"][0]["desc"]


def test_industrial_sample_reports_violations_and_treatments(client):
    body = post(client, INDUSTRIAL).get_json()
    assert len(body["violations"]) >= 5
    assert body["diagnostic"]["score"] < 60
    titles = [r["title"] for r in body["recommendations"]]
    assert "Coagulation & Activated Carbon Filtering" in titles


def test_unloaded_model_returns_503(client, monkeypatch):
    monkeypatch.delitem(aquasense.MODELS, "decision_tree")
    assert post(client, {**CLEAN, "model": "decision_tree"}).status_code == 503


def test_internal_errors_do_not_leak_details(client, monkeypatch):
    class Broken(StubPipeline):
        def predict(self, X):
            raise RuntimeError("secret internal detail")
    monkeypatch.setitem(aquasense.MODELS, "xgboost", Broken(0.5))
    r = post(client, {**CLEAN, "model": "xgboost"})
    assert r.status_code == 500
    assert "secret" not in r.get_data(as_text=True)


# ---------------------------------------------------------------- guideline bands

def base_inputs(**overrides):
    return {**CLEAN, **overrides}


def status_of(key, value):
    _, params, _, _ = aquasense.evaluate_parameters(base_inputs(**{key: value}))
    return next(p["status"] for p in params if p["key"] == key)


@pytest.mark.parametrize("key,value,expected", [
    ("sulfate", 350, "optimal"), ("sulfate", 350.1, "acceptable"),
    ("sulfate", 450, "acceptable"), ("sulfate", 450.1, "critical"),
    ("ph", 6.8, "optimal"), ("ph", 6.79, "acceptable"),
    ("ph", 6.5, "acceptable"), ("ph", 6.49, "critical"),
    ("ph", 8.2, "optimal"), ("ph", 8.5, "acceptable"), ("ph", 8.51, "critical"),
    ("conductivity", 650, "optimal"), ("conductivity", 650.1, "critical"),
    ("trihalo", 80, "optimal"), ("trihalo", 80.1, "critical"),
    ("turbidity", 4.0, "optimal"), ("turbidity", 5.0, "acceptable"), ("turbidity", 5.01, "critical"),
    ("hardness", 330, "acceptable"), ("hardness", 330.5, "critical"),
])
def test_guideline_band_boundaries(key, value, expected):
    assert status_of(key, value) == expected


def test_clean_sample_has_no_violations_and_full_score():
    violations, params, score, flagged = aquasense.evaluate_parameters(CLEAN)
    assert violations == [] and flagged == set()
    assert all(p["status"] == "optimal" for p in params)
    assert score == 94


def test_synergy_rules_flag_combined_hazards():
    violations, _, _, flagged = aquasense.evaluate_parameters(
        base_inputs(solids=23000, sulfate=380, organic=16, trihalo=70))
    assert any("Salinity" in v for v in violations)
    assert any("Precursor" in v for v in violations)
    assert {"solids", "sulfate", "organic", "trihalo"} <= flagged


def test_score_rounds_half_up_like_javascript():
    # pH 6.35 -> critical penalty 15 + 6.5 = 21.5; TDS 21195.89 -> -10 => 94 - 31.5 = 62.5 -> 63
    _, _, score, _ = aquasense.evaluate_parameters(base_inputs(ph=6.35, solids=21195.89))
    assert score == 63


# ---------------------------------------------------------------- metadata & static routes

def test_metadata_exposes_standards_and_matching_champion(client):
    meta = client.get("/api/metadata").get_json()["metadata"]
    assert set(meta["standards"]) == set(aquasense.WATER_STANDARDS)
    assert meta["model_key"] == aquasense.CHAMPION_KEY
    health = client.get("/api/health").get_json()
    assert health["champion_model"] == aquasense.CHAMPION_KEY


def test_champion_is_highest_cross_validated_f1_among_loaded_models():
    cv = {aquasense.model_key_from_name(m["Model"]): m["CV-F1-Mean"] for m in aquasense.MODEL_METRICS}
    loaded = {k: v for k, v in cv.items() if k in aquasense.MODELS}
    assert aquasense.CHAMPION_KEY == max(loaded, key=loaded.get)


def test_metadata_reports_cv_scores_for_champion(client):
    metrics = client.get("/api/metadata").get_json()["metadata"]["metrics"]
    assert 0 < metrics["cv_f1_mean"] < 1 and metrics["cv_f1_std"] >= 0


def test_app_and_training_script_agree_on_champion():
    ranked = aquasense.MODEL_METRICS
    assert aquasense.model_key_from_name(training.select_champion(ranked)) == aquasense.CHAMPION_KEY
    with open(os.path.join(aquasense.ARTIFACTS_DIR, "best_model.txt"), encoding="utf-8") as f:
        assert aquasense.model_key_from_name(f.read().strip()) == aquasense.CHAMPION_KEY


def test_exported_metrics_include_cross_validation_fields():
    for record in aquasense.MODEL_METRICS:
        assert {"CV-F1-Mean", "CV-F1-Std", "CV-ROC-AUC-Mean"} <= set(record)
    for candidate in aquasense.CANDIDATE_MODELS.values():
        assert candidate["cv_f1_mean"] is not None
    assert sum(1 for c in aquasense.CANDIDATE_MODELS.values() if c["is_active"]) == 1


# ---------------------------------------------------------------- champion selection rule

def fake_record(name, cv_f1, cv_auc, test_f1, test_auc=0.6):
    return {"Model": name, "CV-F1-Mean": cv_f1, "CV-ROC-AUC-Mean": cv_auc, "F1-Score": test_f1, "ROC-AUC": test_auc}


def test_select_champion_uses_cv_not_test_f1():
    records = [fake_record("A", cv_f1=0.50, cv_auc=0.6, test_f1=0.70),   # best on the test set only
               fake_record("B", cv_f1=0.55, cv_auc=0.6, test_f1=0.40)]
    assert training.select_champion(records) == "B"


def test_select_champion_breaks_ties_with_cv_roc_auc():
    records = [fake_record("A", 0.55, cv_auc=0.60, test_f1=0.5),
               fake_record("B", 0.55, cv_auc=0.66, test_f1=0.5)]
    assert training.select_champion(records) == "B"


def test_select_champion_falls_back_to_test_metrics_without_cv():
    records = [{"Model": "A", "F1-Score": 0.4, "ROC-AUC": 0.6}, {"Model": "B", "F1-Score": 0.5, "ROC-AUC": 0.6}]
    assert training.select_champion(records) == "B"


def test_analysis_exposes_a_real_correlation_matrix(client):
    corr = client.get("/api/analysis").get_json()["correlation"]
    n = len(corr["labels"])
    assert n == 9 and len(corr["matrix"]) == 9 and all(len(row) == 9 for row in corr["matrix"])
    assert all(corr["matrix"][i][i] == pytest.approx(1.0) for i in range(n))
    assert all(corr["matrix"][i][j] == pytest.approx(corr["matrix"][j][i], abs=1e-3) for i in range(n) for j in range(n))
    # the dataset has no strong single predictor: this is what caps model accuracy near 65%
    assert max(abs(v) for v in corr["with_potability"].values()) < 0.05


def test_static_files_and_404(client):
    assert client.get("/").status_code == 200
    assert client.get("/predict.html").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert client.get("/does-not-exist.js").status_code == 404
