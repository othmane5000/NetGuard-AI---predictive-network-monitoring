"""
NETGUARD AI - Classification Training
========================================

Models built:
  1. Logistic Regression baseline           (manual NumPy + sklearn, lambda=0)
  2. Regularized Logistic Regression         (manual NumPy + sklearn, L2, tuned lambda/C)

SPLIT STRATEGY (important, explicitly stated assumption):
We split by DEVICE, not by row. If we split randomly by row, readings
from the same device (highly autocorrelated in time) would appear in both
train and test sets, letting the model "memorize" a device's specific
behaviour rather than learning generalizable failure patterns. This is a
subtle but critical form of leakage in time-series-per-entity data.
80% of devices -> train, 20% of devices -> test.

Feature scaling: StandardScaler (z-score) is required for gradient
descent to converge well and for L2 regularization to penalize all
features fairly (otherwise large-scale features like `traffic_in` would
dominate the penalty term over small-scale ones like `packet_loss`).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.feature_engineering.build_features import build_feature_table, get_feature_columns, TARGET_COLUMN
from src.models.manual_implementations import ManualLogisticRegression
from src.evaluation.metrics import classification_report_manual, get_confusion_matrix, get_roc_curve

BASE = Path(__file__).resolve().parents[2]


def device_split(df, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    devices = np.array(df["device_id"].unique(), dtype=object)
    devices = devices[rng.permutation(len(devices))]
    n_test = max(1, int(len(devices) * test_size))
    test_devices = set(devices[:n_test])
    train_mask = ~df["device_id"].isin(test_devices)
    return df[train_mask].copy(), df[~train_mask].copy()


def main():
    raw = pd.read_csv(BASE / "data" / "raw" / "network_telemetry.csv", parse_dates=["timestamp"])
    features_df = build_feature_table(raw)
    feat_cols = get_feature_columns(features_df)

    train_df, test_df = device_split(features_df)
    print(f"Train devices: {train_df['device_id'].nunique()} ({len(train_df)} rows) | "
          f"Test devices: {test_df['device_id'].nunique()} ({len(test_df)} rows)")

    X_train_raw = train_df[feat_cols].values.astype(float)
    X_test_raw = test_df[feat_cols].values.astype(float)
    y_train = train_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    results = {}

    # ---------- Model 1: Logistic Regression baseline ----------
    print("\n=== Model 1: Logistic Regression Baseline ===")
    manual_base = ManualLogisticRegression(learning_rate=0.5, n_iterations=1500, lambda_reg=0.0)
    manual_base.fit(X_train, y_train)
    proba_manual_base = manual_base.predict_proba(X_test)
    pred_manual_base = (proba_manual_base >= 0.5).astype(int)
    rep = classification_report_manual(y_test, pred_manual_base, proba_manual_base)
    print("Manual (NumPy):", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()})
    results["logreg_baseline_manual"] = rep

    sk_base = LogisticRegression(penalty=None, max_iter=2000)
    sk_base.fit(X_train, y_train)
    proba_sk_base = sk_base.predict_proba(X_test)[:, 1]
    pred_sk_base = sk_base.predict(X_test)
    rep_sk = classification_report_manual(y_test, pred_sk_base, proba_sk_base)
    print("sklearn        :", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep_sk.items()})
    results["logreg_baseline_sklearn"] = rep_sk

    # ---------- Model 2: Regularized Logistic Regression ----------
    print("\n=== Model 2: Regularized Logistic Regression (L2) ===")
    # sklearn tuning via GridSearchCV to pick C (C = 1/lambda)
    grid = GridSearchCV(
        LogisticRegression(penalty="l2", max_iter=2000, class_weight="balanced"),
        param_grid={"C": [0.01, 0.1, 1, 10]},
        scoring="recall", cv=3,
    )
    grid.fit(X_train, y_train)
    best_C = grid.best_params_["C"]
    print(f"Best C selected by CV (optimizing recall): {best_C}")

    sk_reg = grid.best_estimator_
    proba_sk_reg = sk_reg.predict_proba(X_test)[:, 1]
    pred_sk_reg = sk_reg.predict(X_test)
    rep_sk_reg = classification_report_manual(y_test, pred_sk_reg, proba_sk_reg)
    print("sklearn (reg)  :", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep_sk_reg.items()})
    results["logreg_regularized_sklearn"] = rep_sk_reg

    manual_reg = ManualLogisticRegression(learning_rate=0.5, n_iterations=1500, lambda_reg=1.0 / best_C)
    manual_reg.fit(X_train, y_train)
    proba_manual_reg = manual_reg.predict_proba(X_test)
    pred_manual_reg = (proba_manual_reg >= 0.5).astype(int)
    rep_manual_reg = classification_report_manual(y_test, pred_manual_reg, proba_manual_reg)
    print("Manual (reg)   :", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep_manual_reg.items()})
    results["logreg_regularized_manual"] = rep_manual_reg

    # ---------- Save best production model (sklearn regularized: used by app) ----------
    models_dir = BASE / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(sk_reg, models_dir / "classification_model.joblib")
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(feat_cols, models_dir / "feature_columns.joblib")

    cm = get_confusion_matrix(y_test, pred_sk_reg)
    fpr, tpr, _ = get_roc_curve(y_test, proba_sk_reg)
    np.save(models_dir / "confusion_matrix.npy", cm)
    np.savez(models_dir / "roc_curve.npz", fpr=fpr, tpr=tpr)

    # coefficient-based feature importance for explainability layer
    coef_importance = pd.Series(sk_reg.coef_[0], index=feat_cols).sort_values(key=abs, ascending=False)
    coef_importance.to_csv(models_dir / "feature_importance.csv", header=["coefficient"])

    with open(BASE / "reports" / "classification_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved production classification model + scaler + metrics to {models_dir}")
    print("\nTop 10 risk indicators (by |coefficient|):")
    print(coef_importance.head(10))

    return results


if __name__ == "__main__":
    main()
