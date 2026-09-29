"""
AquaSense AI - Model Training & Evaluation Pipeline
===================================================
Automates end-to-end dataset exploration, preprocessing, machine learning pipeline
training, holdout validation, feature importance extraction, and artifact serialization.

Candidate Models:
  1. XGBoost (Gradient-boosted trees)
  2. Random Forest (Ensemble Bagging)
  3. Decision Tree (Interpretable Hierarchical Rules)
  4. Logistic Regression (Linear Sigmoidal Baseline)

The active champion is chosen by 5-fold stratified cross-validation on the TRAINING split only
(highest mean F1, ties broken by mean ROC-AUC). The held-out test set is scored once and plays no
part in the choice, so the reported test metrics are an unbiased estimate.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==============================================================================
# PIPELINE CONFIGURATION & CONSTANTS
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "water_potability.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "aquasense_artifacts")

FEATURES: List[str] = [
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
TARGET: str = "Potability"

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

MODEL_ORDER: List[str] = [
    "XGBoost",
    "Random Forest",
    "Decision Tree",
    "Logistic Regression"
]

MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "XGBoost": {
        "safe_key": "xgboost",
        "display_name": "XGBoost Classifier (Gradient Boosting)",
        "threshold": 0.50
    },
    "Random Forest": {
        "safe_key": "random_forest",
        "display_name": "Random Forest Classifier (Ensemble Bagging)",
        "threshold": 0.50
    },
    "Decision Tree": {
        "safe_key": "decision_tree",
        "display_name": "Decision Tree Classifier (Hierarchical Rules)",
        "threshold": 0.50
    },
    "Logistic Regression": {
        "safe_key": "logistic_regression",
        "display_name": "Logistic Regression (Linear Sigmoid Baseline)",
        "threshold": 0.50
    }
}


# ==============================================================================
# 1. DATA INGESTION & EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

def load_dataset(filepath: str) -> pd.DataFrame:
    """Loads the water potability dataset from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    df = pd.read_csv(filepath)
    print("=" * 70)
    print("  AQUASENSE AI - MACHINE LEARNING DATASET & MODEL BENCHMARK")
    print("=" * 70)
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def perform_exploratory_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes distribution statistics, missing value profiles, and correlations."""
    missing_counts = df[FEATURES].isna().sum().to_dict()
    missing_pcts = (df[FEATURES].isna().mean() * 100).round(2).to_dict()
    total_missing = int(df[FEATURES].isna().sum().sum())

    target_counts = df[TARGET].value_counts().to_dict()
    non_potable_count = int(target_counts.get(0, 0))
    potable_count = int(target_counts.get(1, 0))
    total_samples = len(df)

    potable_pct = round((potable_count / total_samples) * 100, 2)
    non_potable_pct = round((non_potable_count / total_samples) * 100, 2)

    print("\n[EDA] Missing Value Profile:")
    for col in FEATURES:
        print(f"  - {col:16s}: {missing_counts[col]:4d} missing ({missing_pcts[col]:5.2f}%)")
    print(f"  Total missing cells across cohort: {total_missing:,}")

    print("\n[EDA] Target Class Distribution:")
    print(f"  - Class 0 (Non-Potable): {non_potable_count:,} ({non_potable_pct}%)")
    print(f"  - Class 1 (Potable)    : {potable_count:,} ({potable_pct}%)")

    # Feature distribution metrics
    stats_summary = {}
    for col in FEATURES:
        s = df[col]
        stats_summary[col] = {
            "mean": float(round(s.mean(), 2)),
            "std": float(round(s.std(), 2)),
            "median": float(round(s.median(), 2)),
            "min": float(round(s.min(), 2)),
            "max": float(round(s.max(), 2)),
            "q25": float(round(s.quantile(0.25), 2)),
            "q75": float(round(s.quantile(0.75), 2)),
            "skew": float(round(s.skew(), 2))
        }

    corr_df = df[FEATURES + [TARGET]].corr(numeric_only=True).round(4)

    return {
        "missing_counts": missing_counts,
        "missing_pcts": missing_pcts,
        "total_missing": total_missing,
        "potable_count": potable_count,
        "non_potable_count": non_potable_count,
        "potable_pct": potable_pct,
        "non_potable_pct": non_potable_pct,
        "feature_statistics": stats_summary,
        "correlation_matrix": corr_df
    }


# ==============================================================================
# 2. PIPELINE FACTORY
# ==============================================================================

def build_model_pipelines(class_imbalance_ratio: float) -> Dict[str, Pipeline]:
    """
    Constructs scikit-learn preprocessing and estimator pipelines for the 4 candidate models.
    Median imputation is encapsulated strictly inside each pipeline to prevent data leakage.
    """
    return {
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=class_imbalance_ratio,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])
    }


# ==============================================================================
# 3. TRAINING & HOLDOUT EVALUATION
# ==============================================================================

def train_and_evaluate(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[Dict[str, Pipeline], List[Dict[str, Any]]]:
    """Fits each candidate pipeline on training data and computes test set metrics."""
    trained_models: Dict[str, Pipeline] = {}
    metrics_list: List[Dict[str, Any]] = []

    print("\n[TRAINING] Fitting Models on Stratified Training Split:")
    for name in MODEL_ORDER:
        pipeline = pipelines[name]
        print(f"  -> Training {name}...")
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

        # Predictions & Probabilities
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate standard evaluation metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred)

        cm_dict = {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1])
        }

        record = {
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "ConfusionMatrix": cm_dict
        }
        metrics_list.append(record)
        print(f"     Acc: {acc*100:5.2f}% | Prec: {prec*100:5.2f}% | Rec: {rec*100:5.2f}% | F1: {f1:.4f} | AUC: {auc:.4f}")

    return trained_models, metrics_list


# ==============================================================================
# 4. FEATURE IMPORTANCE EXTRACTION
# ==============================================================================

def cross_validate_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Stratified k-fold cross-validation on the TRAINING split only (the test set is never touched).
    Pipelines are cloned internally, so the later final fit is unaffected. Because imputation and
    scaling live inside each pipeline, they are re-fit per fold (no leakage across folds).
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: Dict[str, Dict[str, float]] = {}

    print(f"\n[CV] {CV_FOLDS}-fold stratified cross-validation on the training split:")
    for name in MODEL_ORDER:
        scores = cross_validate(
            pipelines[name], X_train, y_train,
            cv=cv, scoring={"f1": "f1", "roc_auc": "roc_auc"}
        )
        results[name] = {
            "CV-F1-Mean": round(float(np.mean(scores["test_f1"])), 4),
            "CV-F1-Std": round(float(np.std(scores["test_f1"])), 4),
            "CV-ROC-AUC-Mean": round(float(np.mean(scores["test_roc_auc"])), 4)
        }
        r = results[name]
        print(f"  -> {name:20s} F1 = {r['CV-F1-Mean']:.4f} ± {r['CV-F1-Std']:.4f} | ROC-AUC = {r['CV-ROC-AUC-Mean']:.4f}")
    return results


def select_champion(metrics_list: List[Dict[str, Any]]) -> str:
    """
    Champion = candidate with the highest cross-validated F1 (ties broken by cross-validated ROC-AUC).
    Selection uses training-split CV only, never the held-out test metrics. Records without CV fields
    fall back to their test F1 / ROC-AUC.
    """
    best = max(
        metrics_list,
        key=lambda r: (r.get("CV-F1-Mean", r["F1-Score"]), r.get("CV-ROC-AUC-Mean", r["ROC-AUC"]))
    )
    return best["Model"]


def extract_feature_importances(trained_models: Dict[str, Pipeline]) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts tree Gini/gain importances and linear regression coefficients."""
    importance_data: Dict[str, List[Dict[str, Any]]] = {}

    # XGBoost
    xgb_model = trained_models["XGBoost"].named_steps["model"]
    xgb_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": xgb_model.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    importance_data["xgboost"] = xgb_df.to_dict(orient="records")

    # Random Forest
    rf_model = trained_models["Random Forest"].named_steps["model"]
    rf_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    importance_data["random_forest"] = rf_df.to_dict(orient="records")

    # Decision Tree
    dt_model = trained_models["Decision Tree"].named_steps["model"]
    dt_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": dt_model.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    importance_data["decision_tree"] = dt_df.to_dict(orient="records")

    # Logistic Regression
    lr_model = trained_models["Logistic Regression"].named_steps["model"]
    lr_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": lr_model.coef_[0]
    }).sort_values(by="Coefficient", key=abs, ascending=False).reset_index(drop=True)
    importance_data["logistic_regression"] = lr_df.to_dict(orient="records")

    return importance_data


# ==============================================================================
# 5. ARTIFACT PERSISTENCE
# ==============================================================================

def export_artifacts(
    artifacts_dir: str,
    total_samples: int,
    class_ratio: float,
    eda_summary: Dict[str, Any],
    trained_models: Dict[str, Pipeline],
    metrics_list: List[Dict[str, Any]],
    importance_data: Dict[str, List[Dict[str, Any]]],
    champion: str
) -> None:
    """Serializes models, JSON schemas, and tabular benchmarks for production serving."""
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Save Trained Model Binaries (.joblib)
    for model_name, pipeline in trained_models.items():
        meta = MODEL_METADATA[model_name]
        filename = f"{meta['safe_key']}.joblib"
        joblib.dump(pipeline, os.path.join(artifacts_dir, filename), compress=3)

    # Champion model = highest cross-validated F1 (see select_champion)
    joblib.dump(trained_models[champion], os.path.join(artifacts_dir, "water_potability_model.joblib"), compress=3)

    # 2. Save Benchmark Metrics (CSV & JSON)
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(artifacts_dir, "model_metrics.csv"), index=False)
    with open(os.path.join(artifacts_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=4)

    # 3. Save Candidate Models Dictionary (Ordered for frontend consumption)
    candidate_models_dict = {}
    for r in metrics_list:
        m_name = r["Model"]
        meta = MODEL_METADATA[m_name]
        candidate_models_dict[m_name] = {
            "name": m_name,
            "display_name": meta["display_name"],
            "accuracy": r["Accuracy"],
            "precision": r["Precision"],
            "recall": r["Recall"],
            "f1": r["F1-Score"],
            "roc_auc": r["ROC-AUC"],
            "cv_f1_mean": r.get("CV-F1-Mean"),
            "cv_f1_std": r.get("CV-F1-Std"),
            "cv_roc_auc_mean": r.get("CV-ROC-AUC-Mean"),
            "threshold": meta["threshold"],
            "is_active": m_name == champion,
            "confusion_matrix": r["ConfusionMatrix"]
        }

    with open(os.path.join(artifacts_dir, "candidate_models.json"), "w", encoding="utf-8") as f:
        json.dump(candidate_models_dict, f, indent=4)

    # 4. Save Feature Registry & Best Model Name
    with open(os.path.join(artifacts_dir, "features.json"), "w", encoding="utf-8") as f:
        json.dump(FEATURES, f, indent=4)

    with open(os.path.join(artifacts_dir, "best_model.txt"), "w", encoding="utf-8") as f:
        f.write(f"{champion}\n")

    # 5. Save Analysis & Dataset Summary
    analysis_summary = {
        "total_samples": total_samples,
        "total_features": len(FEATURES),
        "features": FEATURES,
        "missing_values": eda_summary["missing_counts"],
        "missing_percentages": eda_summary["missing_pcts"],
        "total_missing_cells": eda_summary["total_missing"],
        "potable_samples": eda_summary["potable_count"],
        "not_potable_samples": eda_summary["non_potable_count"],
        "potable_percentage": eda_summary["potable_pct"],
        "not_potable_percentage": eda_summary["non_potable_pct"],
        "feature_statistics": eda_summary["feature_statistics"],
        "class_imbalance_ratio": round(class_ratio, 4),
        "best_model": champion
    }
    with open(os.path.join(artifacts_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(analysis_summary, f, indent=4)

    # 6. Save Correlation Matrix
    corr_df = eda_summary["correlation_matrix"]
    corr_df.to_csv(os.path.join(artifacts_dir, "correlation_matrix.csv"))
    corr_df.to_json(os.path.join(artifacts_dir, "correlation_matrix.json"), orient="split")

    # 7. Save Feature Importances
    pd.DataFrame(importance_data["xgboost"]).to_csv(
        os.path.join(artifacts_dir, "xgboost_feature_importance.csv"), index=False
    )
    pd.DataFrame(importance_data["random_forest"]).to_csv(
        os.path.join(artifacts_dir, "random_forest_feature_importance.csv"), index=False
    )
    with open(os.path.join(artifacts_dir, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(importance_data, f, indent=4)

    print(f"\n[ARTIFACTS] Exported all pipeline binaries and JSON artifacts to '{artifacts_dir}'.")


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================

def main() -> None:
    """Executes the full machine learning training, evaluation, and artifact pipeline."""
    # 1. Ingest Data & Run Exploratory Analysis
    df = load_dataset(DATASET_PATH)
    eda_summary = perform_exploratory_analysis(df)

    # 2. Stratified Train / Test Partitioning
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    class_ratio = float((y_train == 0).sum() / (y_train == 1).sum())

    print(f"\n[SPLIT] Partition Sizes: Train={len(X_train):,}, Test={len(X_test):,}")
    print(f"[SPLIT] Stratified Class Imbalance (Neg/Pos Ratio): {class_ratio:.4f}")

    # 3. Build & Train Model Pipelines
    pipelines = build_model_pipelines(class_ratio)

    # Model selection uses cross-validation on the training split only
    cv_results = cross_validate_models(pipelines, X_train, y_train)

    # Final fit on the full training split; the test set is scored exactly once
    trained_models, metrics_list = train_and_evaluate(pipelines, X_train, X_test, y_train, y_test)
    for record in metrics_list:
        record.update(cv_results[record["Model"]])

    # 4. Display Formatted Benchmark Summary
    benchmark_df = pd.DataFrame(metrics_list)[
        ["Model", "CV-F1-Mean", "CV-F1-Std", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    ]
    champion = select_champion(metrics_list)
    print("\n" + "=" * 78)
    print(f"  MODEL BENCHMARK (CV = {CV_FOLDS}-fold on train; other columns = held-out test, N = {len(X_test):,})")
    print("=" * 78)
    print(benchmark_df.to_string(index=False))
    print("=" * 78)
    print(f"Active Champion Model: {champion} (highest cross-validated F1; test set not used for selection)")

    # 5. Extract Feature Importances
    importance_data = extract_feature_importances(trained_models)

    # 6. Export All Artifacts
    export_artifacts(
        artifacts_dir=ARTIFACTS_DIR,
        total_samples=len(df),
        class_ratio=class_ratio,
        eda_summary=eda_summary,
        trained_models=trained_models,
        metrics_list=metrics_list,
        importance_data=importance_data,
        champion=champion
    )
    print("\n[OK] Pipeline execution completed successfully.\n")


if __name__ == "__main__":
    main()
