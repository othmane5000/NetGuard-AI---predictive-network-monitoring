import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from src.evaluation.metrics import classification_report_manual, regression_report


def test_classification_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)

    manual = classification_report_manual(y_true, y_pred)

    assert np.isclose(manual["accuracy"], accuracy_score(y_true, y_pred))
    assert np.isclose(manual["precision"], precision_score(y_true, y_pred, zero_division=0))
    assert np.isclose(manual["recall"], recall_score(y_true, y_pred, zero_division=0))
    assert np.isclose(manual["f1_score"], f1_score(y_true, y_pred, zero_division=0))


def test_regression_metrics_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    rep = regression_report(y_true, y_pred)
    assert np.isclose(rep["mae"], 0.0)
    assert np.isclose(rep["rmse"], 0.0)
    assert np.isclose(rep["r2"], 1.0)


def test_regression_metrics_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    rep = regression_report(y_true, y_pred)
    assert np.isclose(rep["mae"], (0 + 0 + 2) / 3)
