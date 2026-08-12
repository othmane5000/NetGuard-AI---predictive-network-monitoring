"""
NETGUARD AI - Prediction Pipeline
=====================================
Loads the trained production models + scaler, and produces a full
RiskAssessment per device from its latest known telemetry reading.
This is the module the Streamlit dashboard calls.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.feature_engineering.build_features import build_feature_table, get_feature_columns
from src.prediction.risk_engine import build_risk_assessment, DEFAULT_THRESHOLDS

BASE = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE / "models"


class NetGuardPredictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.clf = joblib.load(self.models_dir / "classification_model.joblib")
        self.scaler = joblib.load(self.models_dir / "scaler.joblib")
        self.feature_columns = joblib.load(self.models_dir / "feature_columns.joblib")

        self.reg_model = None
        self.reg_scaler = None
        self.reg_feature_columns = None
        reg_path = self.models_dir / "regression_model.joblib"
        if reg_path.exists():
            self.reg_model = joblib.load(reg_path)
            self.reg_scaler = joblib.load(self.models_dir / "regression_scaler.joblib")
            self.reg_feature_columns = joblib.load(self.models_dir / "regression_feature_columns.joblib")

    def build_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return build_feature_table(raw_df)

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        X = features_df[self.feature_columns].values.astype(float)
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def predict_degradation_score(self, features_df: pd.DataFrame):
        if self.reg_model is None:
            return None
        cols = [c for c in self.reg_feature_columns if c in features_df.columns]
        X = features_df[cols].values.astype(float)
        X_scaled = self.reg_scaler.transform(X)
        return self.reg_model.predict(X_scaled)

    def assess_all_devices(self, raw_df: pd.DataFrame, thresholds=None,
                            latest_only=True) -> pd.DataFrame:
        """
        Returns one row per device (latest snapshot by default) with:
        failure_probability, risk_level, prediction, indicators, actions.
        """
        features_df = self.build_features(raw_df)
        proba = self.predict_proba(features_df)
        features_df = features_df.copy()
        features_df["failure_probability"] = proba

        if latest_only:
            idx = features_df.groupby("device_id")["timestamp"].idxmax()
            snapshot = features_df.loc[idx].reset_index(drop=True)
        else:
            snapshot = features_df

        coef = self.clf.coef_[0]
        X_all = snapshot[self.feature_columns].values.astype(float)
        X_scaled = self.scaler.transform(X_all)

        rows = []
        for i, row in snapshot.iterrows():
            assessment = build_risk_assessment(
                device_id=row["device_id"],
                probability=row["failure_probability"],
                feature_values=X_scaled[i] if latest_only else X_scaled[snapshot.index.get_loc(i)],
                feature_names=self.feature_columns,
                coefficients=coef,
                thresholds=thresholds,
            )
            rows.append({
                "device_id": row["device_id"],
                "device_type": row.get("device_type", self._infer_type(row)),
                "timestamp": row["timestamp"],
                "cpu_usage": row["cpu_usage"],
                "memory_usage": row["memory_usage"],
                "temperature": row["temperature"],
                "packet_loss": row["packet_loss"],
                "latency": row["latency"],
                "interface_errors": row["interface_errors"],
                "failure_probability": assessment.failure_probability,
                "risk_level": assessment.risk_level,
                "prediction": assessment.prediction,
                "indicators": assessment.indicators,
                "recommended_actions": assessment.recommended_actions,
            })

        result = pd.DataFrame(rows)
        return result

    @staticmethod
    def _infer_type(row):
        for c in row.index:
            if c.startswith("dtype_") and row[c] == 1:
                return c.replace("dtype_", "")
        return "unknown"


if __name__ == "__main__":
    raw = pd.read_csv(BASE / "data" / "raw" / "network_telemetry.csv", parse_dates=["timestamp"])
    predictor = NetGuardPredictor()
    assessment = predictor.assess_all_devices(raw)
    print(assessment[["device_id", "device_type", "failure_probability", "risk_level", "prediction"]]
          .sort_values("failure_probability", ascending=False).to_string(index=False))
