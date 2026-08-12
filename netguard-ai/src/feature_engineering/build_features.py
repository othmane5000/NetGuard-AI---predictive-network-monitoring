"""
NETGUARD AI - Feature Engineering
===================================

DATA LEAKAGE POLICY (read this first)
----------------------------------------
The target `failure` at time t must be predicted using ONLY information
that would be available at time t in a real monitoring system:
  - the CURRENT raw metrics at time t (cpu_usage, temperature, ... at t)
  - HISTORICAL metrics strictly BEFORE t (rolling windows, trends, lags)

The following columns from the raw dataset are EXCLUDED from the feature
set because they would leak the label or are not causal inputs:
  - `failure_type`      -> only exists when failure happened (leaks label)
  - `failure` itself     -> obviously leaks (it's the target)
  - `failure_probability_true` -> generator's internal ground-truth prob,
                                    never present in the real raw csv anyway.

All rolling/trend features use `min_periods=1` and are computed with
`closed='left'`-equivalent logic (i.e. the window for row t includes rows
up to and including t only when using CURRENT values, and up to t-1 when
we need a "before this reading" trend). We use pandas `.rolling()` on
values INCLUDING the current reading for moving averages (this is
standard - "CPU moving average" means "recent CPU behaviour including
now", which is legitimately available at prediction time), and explicit
`.shift(1)` for anything meant to represent "the trend leading up to
now" so today's failure can't leak into its own trend feature.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROLL_WINDOW = 6   # 6-hour rolling window
TREND_WINDOW = 12  # 12-hour trend window


def _device_sorted(df):
    return df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)


def add_rolling_features(df):
    """Moving averages of current + recent readings (available at time t)."""
    df = _device_sorted(df)
    g = df.groupby("device_id")

    for col in ["cpu_usage", "temperature", "packet_loss", "latency", "interface_errors"]:
        df[f"{col}_ma{ROLL_WINDOW}"] = (
            g[col].transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).mean())
        )
        df[f"{col}_std{ROLL_WINDOW}"] = (
            g[col].transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=1).std().fillna(0))
        )
    return df


def add_trend_features(df):
    """
    Trend = current value minus the value TREND_WINDOW steps ago (shifted),
    representing the direction/speed of change leading INTO the current
    reading. Uses only past values, so no leakage.
    """
    df = _device_sorted(df)
    g = df.groupby("device_id")

    for col in ["cpu_usage", "packet_loss", "latency", "interface_errors", "bandwidth_usage"]:
        lagged = g[col].transform(lambda s: s.shift(TREND_WINDOW))
        df[f"{col}_trend"] = (df[col] - lagged).fillna(0)

    # Traffic growth (%) over the trend window
    for col in ["traffic_in", "traffic_out"]:
        lagged = g[col].transform(lambda s: s.shift(TREND_WINDOW))
        df[f"{col}_growth_pct"] = ((df[col] - lagged) / (lagged.abs() + 1e-3)).fillna(0)
        df[f"{col}_growth_pct"] = df[f"{col}_growth_pct"].clip(-5, 5)

    return df


def add_spike_and_anomaly_features(df):
    """Binary/continuous indicators for sudden spikes vs. the device's own
    recent baseline (z-score against the rolling mean/std computed above)."""
    df = df.copy()

    df["cpu_spike"] = (
        (df["cpu_usage"] - df["cpu_usage_ma6"]) / (df["cpu_usage_std6"] + 1e-3)
    ).clip(-5, 5)
    df["temperature_anomaly"] = (
        (df["temperature"] - df["temperature_ma6"]) / (df["temperature_std6"] + 1e-3)
    ).clip(-5, 5)
    df["cpu_spike_flag"] = (df["cpu_spike"] > 2).astype(int)
    df["temperature_anomaly_flag"] = (df["temperature_anomaly"] > 2).astype(int)
    return df


def add_device_age_and_history_features(df):
    df = df.copy()
    df["uptime_days"] = df["uptime_hours"] / 24.0
    df["long_uptime_flag"] = (df["uptime_days"] > 200).astype(int)
    df["recent_failure_flag"] = (df["previous_failures"] > 0).astype(int)
    return df


def add_error_and_loss_rates(df):
    """Composite ratios that a network engineer would actually look at."""
    df = df.copy()
    df["error_rate"] = df["interface_errors"] / (df["active_connections"] + 1)
    df["loss_over_latency"] = df["packet_loss"] / (df["latency"] + 1)
    return df


def encode_categoricals(df):
    """One-hot encode device_type (kept simple/interpretable rather than
    embeddings, consistent with the linear-model focus of this project)."""
    df = pd.get_dummies(df, columns=["device_type"], prefix="dtype")
    return df


LEAKY_COLUMNS = ["failure_type", "failure_probability_true"]
NON_FEATURE_COLUMNS = ["timestamp", "device_id"]  # identifiers, not model inputs
TARGET_COLUMN = "failure"


def build_feature_table(raw_df):
    df = raw_df.copy()
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_spike_and_anomaly_features(df)
    df = add_device_age_and_history_features(df)
    df = add_error_and_loss_rates(df)

    for c in LEAKY_COLUMNS:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = encode_categoricals(df)
    return df


def get_feature_columns(df):
    """Everything except identifiers and the target."""
    exclude = set(NON_FEATURE_COLUMNS + [TARGET_COLUMN])
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    raw = pd.read_csv(base / "data" / "raw" / "network_telemetry.csv", parse_dates=["timestamp"])
    features = build_feature_table(raw)

    out = base / "data" / "processed" / "features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)

    feat_cols = get_feature_columns(features)
    print(f"Built {len(feat_cols)} feature columns for {len(features):,} rows")
    print("Feature columns:", feat_cols)
    print(f"Saved to: {out}")
