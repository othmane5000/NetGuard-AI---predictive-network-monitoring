import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from src.prediction.risk_engine import get_risk_level, build_risk_assessment, DEFAULT_THRESHOLDS


def test_risk_level_boundaries():
    assert get_risk_level(0.0) == "LOW"
    assert get_risk_level(0.29) == "LOW"
    assert get_risk_level(0.30) == "MEDIUM"
    assert get_risk_level(0.59) == "MEDIUM"
    assert get_risk_level(0.60) == "HIGH"
    assert get_risk_level(0.79) == "HIGH"
    assert get_risk_level(0.80) == "CRITICAL"
    assert get_risk_level(0.99) == "CRITICAL"


def test_custom_thresholds():
    custom = {"LOW": (0, 0.5), "MEDIUM": (0.5, 0.7), "HIGH": (0.7, 0.9), "CRITICAL": (0.9, 1.01)}
    assert get_risk_level(0.45, custom) == "LOW"
    assert get_risk_level(0.85, custom) == "HIGH"


def test_build_risk_assessment_high_risk_has_actions():
    feature_names = ["packet_loss", "cpu_usage", "temperature"]
    coefficients = np.array([2.5, 1.8, 0.3])
    feature_values = np.array([2.0, 1.5, 0.1])  # standardized, all positive -> risk up

    assessment = build_risk_assessment(
        device_id="TEST-01", probability=0.9,
        feature_values=feature_values, feature_names=feature_names,
        coefficients=coefficients,
    )
    assert assessment.risk_level == "CRITICAL"
    assert len(assessment.indicators) > 0
    assert len(assessment.recommended_actions) > 0


def test_build_risk_assessment_low_risk_no_alarm_actions():
    feature_names = ["packet_loss", "cpu_usage"]
    coefficients = np.array([2.5, 1.8])
    feature_values = np.array([-1.0, -1.0])  # negative -> pushes risk down

    assessment = build_risk_assessment(
        device_id="TEST-02", probability=0.05,
        feature_values=feature_values, feature_names=feature_names,
        coefficients=coefficients,
    )
    assert assessment.risk_level == "LOW"
