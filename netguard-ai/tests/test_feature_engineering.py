import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from src.data_processing.generate_data import generate_dataset
from src.feature_engineering.build_features import build_feature_table, get_feature_columns, LEAKY_COLUMNS, TARGET_COLUMN


def _sample_raw():
    return generate_dataset(n_devices=3, n_steps=48, seed=1)


def test_generated_dataset_has_expected_columns():
    df = _sample_raw()
    for col in ["timestamp", "device_id", "device_type", "cpu_usage", "failure", "failure_type"]:
        assert col in df.columns


def test_failure_rate_is_reasonable():
    df = _sample_raw()
    rate = df["failure"].mean()
    assert 0.0 <= rate <= 0.5  # sanity bound, not exact (stochastic)


def test_no_leaky_columns_in_feature_table():
    raw = _sample_raw()
    features = build_feature_table(raw)
    for leaky_col in LEAKY_COLUMNS:
        assert leaky_col not in features.columns


def test_feature_columns_exclude_identifiers_and_target():
    raw = _sample_raw()
    features = build_feature_table(raw)
    feat_cols = get_feature_columns(features)
    assert "device_id" not in feat_cols
    assert "timestamp" not in feat_cols
    assert TARGET_COLUMN not in feat_cols


def test_rolling_features_do_not_use_future_data():
    """A rolling mean at row t should never differ if we truncate the
    series after t (i.e. it must not depend on rows after t)."""
    raw = _sample_raw()
    device_id = raw["device_id"].iloc[0]
    device_df = raw[raw["device_id"] == device_id].sort_values("timestamp").reset_index(drop=True)

    cutoff = len(device_df) // 2
    truncated = device_df.iloc[:cutoff].copy()

    full_feats = build_feature_table(device_df)
    trunc_feats = build_feature_table(truncated)

    full_value = full_feats.iloc[cutoff - 1]["cpu_usage_ma6"]
    trunc_value = trunc_feats.iloc[cutoff - 1]["cpu_usage_ma6"]
    assert np.isclose(full_value, trunc_value), "Rolling feature leaks future information"


def test_previous_failures_excludes_current_row():
    raw = _sample_raw()
    # first row per device must have previous_failures == 0
    first_rows = raw.sort_values(["device_id", "timestamp"]).groupby("device_id").head(1)
    assert (first_rows["previous_failures"] == 0).all()
