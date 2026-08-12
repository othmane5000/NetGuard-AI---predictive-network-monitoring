"""
NETGUARD AI - Regression Training (Model 3)
==============================================

Target: a continuous DEGRADATION SCORE (0-100) rather than failure
probability. This demonstrates the regression half of the course
material (separate from the classification failure prediction).

Degradation score definition (explicitly stated assumption, since no
dataset provides a ground-truth "degradation score" directly):
    degradation_score = weighted combination of normalized stress
    indicators (packet loss, interface errors, temperature, CPU, latency)
    scaled to 0-100.
This is a deterministic, documented formula (not a learned target from
somewhere else), used as an interpretable composite health index.

FORECASTING FRAMING (avoids trivial/circular regression):
If we predicted the degradation_score AT time t from the raw metrics AT
time t, the model would trivially "solve" a closed-form formula it can
see all the inputs to (R^2 -> 1.0, an unrealistic and uninteresting
result for a portfolio project). Instead, per the project brief's
suggestion to predict a FUTURE metric, this script predicts the
degradation score H=6 HOURS AHEAD using only features available up to
and including the current time t. This is a genuine forecasting task:
the model must learn how current stress trends translate into future
degradation, and cannot simply "read off" the answer from present-time
inputs. Rows near the end of each device's history (where t+H doesn't
exist yet) are dropped - this is standard practice for supervised
forecasting datasets and does not introduce leakage.
"""

FORECAST_HORIZON_HOURS = 6

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.feature_engineering.build_features import build_feature_table, get_feature_columns, TARGET_COLUMN
from src.models.manual_implementations import ManualLinearRegression
from src.evaluation.metrics import regression_report

BASE = Path(__file__).resolve().parents[2]


def compute_degradation_score(df):
    """Deterministic composite index, 0-100, higher = worse."""
    def norm(s, cap):
        return (s.clip(lower=0, upper=cap) / cap * 100)

    score = (
        0.30 * norm(df["packet_loss"], 15) +
        0.25 * norm(df["interface_errors"], 20) +
        0.20 * norm(df["temperature"] - 30, 60) +
        0.15 * norm(df["cpu_usage"], 100) +
        0.10 * norm(df["latency"], 200)
    )
    return score.clip(0, 100)


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
    raw = raw.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
    raw["degradation_score_now"] = compute_degradation_score(raw)

    features_df = build_feature_table(raw)

    # Build the FUTURE target: degradation score H hours ahead, per device.
    features_df = features_df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
    features_df["degradation_score_future"] = (
        features_df.groupby("device_id")["timestamp"]
        .transform(lambda s: s)  # no-op, keeps alignment explicit
    )
    future_scores = raw.groupby("device_id")["degradation_score_now"].shift(-FORECAST_HORIZON_HOURS)
    features_df["degradation_score_future"] = future_scores.values

    # Drop rows with no future value available (end of each device's history)
    features_df = features_df.dropna(subset=["degradation_score_future"]).reset_index(drop=True)

    feat_cols = get_feature_columns(features_df)
    feat_cols = [c for c in feat_cols if c != "degradation_score_future"]

    train_df, test_df = device_split(features_df)
    X_train_raw = train_df[feat_cols].values.astype(float)
    X_test_raw = test_df[feat_cols].values.astype(float)
    y_train = train_df["degradation_score_future"].values
    y_test = test_df["degradation_score_future"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    results = {}

    print("=== Model 3a: Linear Regression (manual, no regularization) ===")
    # NOTE: the target (degradation score, 0-100) is NOT standardized, only
    # the features are. This means gradients are larger than in the
    # classification case (target range 100x bigger than a 0/1 label), so
    # a smaller learning rate is required for gradient descent to converge
    # instead of diverging/overflowing - a direct, hands-on illustration
    # of why learning-rate selection matters in gradient descent.
    manual_lr = ManualLinearRegression(learning_rate=0.02, n_iterations=3000, lambda_reg=0.0)
    manual_lr.fit(X_train, y_train)
    pred_manual = manual_lr.predict(X_test)
    rep_manual = regression_report(y_test, pred_manual)
    print({k: round(v, 4) for k, v in rep_manual.items()})
    results["linear_regression_manual"] = rep_manual

    sk_lr = LinearRegression()
    sk_lr.fit(X_train, y_train)
    pred_sk = sk_lr.predict(X_test)
    rep_sk = regression_report(y_test, pred_sk)
    print("sklearn:", {k: round(v, 4) for k, v in rep_sk.items()})
    results["linear_regression_sklearn"] = rep_sk

    print("\n=== Model 3b: Ridge (regularized) Regression ===")
    manual_ridge = ManualLinearRegression(learning_rate=0.02, n_iterations=3000, lambda_reg=5.0)
    manual_ridge.fit(X_train, y_train)
    pred_manual_ridge = manual_ridge.predict(X_test)
    rep_manual_ridge = regression_report(y_test, pred_manual_ridge)
    print({k: round(v, 4) for k, v in rep_manual_ridge.items()})
    results["ridge_regression_manual"] = rep_manual_ridge

    sk_ridge = Ridge(alpha=5.0)
    sk_ridge.fit(X_train, y_train)
    pred_sk_ridge = sk_ridge.predict(X_test)
    rep_sk_ridge = regression_report(y_test, pred_sk_ridge)
    print("sklearn:", {k: round(v, 4) for k, v in rep_sk_ridge.items()})
    results["ridge_regression_sklearn"] = rep_sk_ridge

    models_dir = BASE / "models"
    joblib.dump(sk_ridge, models_dir / "regression_model.joblib")
    joblib.dump(scaler, models_dir / "regression_scaler.joblib")
    joblib.dump(feat_cols, models_dir / "regression_feature_columns.joblib")
    joblib.dump(FORECAST_HORIZON_HOURS, models_dir / "forecast_horizon.joblib")

    with open(BASE / "reports" / "regression_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved production regression model to {models_dir}")
    return results


if __name__ == "__main__":
    main()
