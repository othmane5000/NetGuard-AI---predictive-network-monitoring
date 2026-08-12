"""
NETGUARD AI - Risk Engine & Explainability
=============================================

Converts a raw failure probability into a business-readable risk level,
and produces a simple, interpretable explanation of WHY a device is at
risk, based on the logistic regression's own coefficients (no black-box
explainability library needed for a linear model - the coefficients ARE
the explanation, which is one of the practical advantages of using
logistic regression here instead of an opaque model).

RISK THRESHOLDS - LIMITATIONS (explicitly stated)
----------------------------------------------------
The default thresholds below (30/60/80) are a reasonable, configurable
starting point, not a scientifically derived optimum. In a real
deployment they should be tuned against the organization's actual
tolerance for false alarms vs missed failures (e.g. via cost-sensitive
analysis on the ROC curve / precision-recall tradeoff), and they may need
to differ per device type or criticality tier (a core switch and an
access point do not carry the same operational risk at the same failure
probability).
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

DEFAULT_THRESHOLDS = {
    "LOW": (0.0, 0.30),
    "MEDIUM": (0.30, 0.60),
    "HIGH": (0.60, 0.80),
    "CRITICAL": (0.80, 1.01),
}

# Human-readable labels + recommended actions per dominant indicator.
# IMPORTANT: these map to OBSERVABLE metrics only. We deliberately do NOT
# claim to identify a specific failing physical component (e.g. "the
# power supply will fail") because the dataset contains no component-level
# information - only aggregate telemetry. See project constraint #9.
INDICATOR_META = {
    "packet_loss":        ("Packet Loss", "Inspect physical links / switch ports for errors and congestion."),
    "packet_loss_ma6":     ("Packet Loss (trend)", "Packet loss has been elevated over the last 6h - check upstream links."),
    "interface_errors":   ("Interface Errors", "Inspect network interfaces and cabling for faults."),
    "interface_errors_ma6": ("Interface Errors (trend)", "Rising interface errors over 6h - schedule interface diagnostics."),
    "temperature":         ("Temperature", "Check cooling / airflow / ambient conditions around the device."),
    "temperature_anomaly": ("Temperature Anomaly", "Sudden temperature spike detected - check cooling system urgently."),
    "cpu_usage":           ("CPU Usage", "Investigate high CPU processes; consider load balancing."),
    "cpu_spike":           ("CPU Spike", "Sudden CPU spike detected - check for traffic surge or process fault."),
    "memory_usage":        ("Memory Usage", "Monitor for memory leaks; consider a scheduled restart if trend continues."),
    "latency":             ("Latency", "Investigate network path congestion or routing issues."),
    "bandwidth_usage":     ("Bandwidth Utilization", "Link approaching capacity - consider traffic shaping or capacity upgrade."),
    "active_connections":  ("Connection Count", "Unusually high connection count - check for abnormal client activity."),
    "previous_failures":   ("Recent Failure History", "Device has a recent failure history - prioritize inspection."),
    "maintenance_days":    ("Maintenance Overdue", "Device is overdue for scheduled maintenance."),
    "uptime_hours":        ("Long Uptime", "Long uptime without restart - consider a scheduled maintenance window."),
}


def get_risk_level(probability, thresholds=None):
    thresholds = thresholds or DEFAULT_THRESHOLDS
    for level, (lo, hi) in thresholds.items():
        if lo <= probability < hi:
            return level
    return "CRITICAL"


@dataclass
class RiskAssessment:
    device_id: str
    failure_probability: float
    risk_level: str
    prediction: str
    indicators: List[Dict] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


def explain_prediction(feature_values: np.ndarray, feature_names: List[str],
                        coefficients: np.ndarray, top_n=5):
    """
    Contribution of feature j to the logit = coefficient_j * standardized_value_j.
    (feature_values must already be STANDARDIZED, i.e. scaler-transformed,
    so contributions are comparable across features regardless of original
    units - this is exactly what the model coefficients operate on.)

    Returns the top_n features by absolute contribution, signed (positive
    contribution = pushes risk UP, negative = pushes risk DOWN).
    """
    contributions = feature_values * coefficients
    order = np.argsort(-np.abs(contributions))[:top_n]

    explained = []
    for idx in order:
        name = feature_names[idx]
        contrib = contributions[idx]
        label, action = INDICATOR_META.get(name, (name.replace("_", " ").title(), None))
        explained.append({
            "feature": name,
            "label": label,
            "contribution": float(contrib),
            "direction": "increases risk" if contrib > 0 else "decreases risk",
            "action": action,
        })
    return explained


def severity_label(contribution, max_abs_contribution):
    """Map a contribution magnitude to a HIGH/MEDIUM/LOW-style tag for
    display, relative to the strongest driver for this device."""
    if max_abs_contribution <= 0:
        return "LOW"
    ratio = abs(contribution) / max_abs_contribution
    if ratio > 0.8:
        return "VERY HIGH"
    if ratio > 0.5:
        return "HIGH"
    if ratio > 0.25:
        return "MEDIUM"
    return "LOW"


def build_risk_assessment(device_id, probability, feature_values, feature_names,
                           coefficients, thresholds=None, top_n=4) -> RiskAssessment:
    risk_level = get_risk_level(probability, thresholds)
    explained = explain_prediction(feature_values, feature_names, coefficients, top_n=top_n)

    positive_drivers = [e for e in explained if e["contribution"] > 0]
    max_abs = max([abs(e["contribution"]) for e in explained], default=0)

    indicators = []
    actions = []
    for e in positive_drivers:
        indicators.append({
            "label": e["label"],
            "severity": severity_label(e["contribution"], max_abs),
        })
        if e["action"] and e["action"] not in actions:
            actions.append(e["action"])

    if risk_level in ("HIGH", "CRITICAL"):
        prediction = "Potential network degradation / failure risk"
    elif risk_level == "MEDIUM":
        prediction = "Early warning signs of degradation"
    else:
        prediction = "Device operating within normal parameters"

    if not actions:
        actions = ["Continue routine monitoring - no urgent action required."]

    return RiskAssessment(
        device_id=device_id,
        failure_probability=float(probability),
        risk_level=risk_level,
        prediction=prediction,
        indicators=indicators or [{"label": "No significant risk drivers", "severity": "LOW"}],
        recommended_actions=actions,
    )
