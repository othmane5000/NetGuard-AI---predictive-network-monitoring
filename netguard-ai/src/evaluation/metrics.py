"""
NETGUARD AI - Evaluation Metrics
===================================

Classification metrics are computed manually from the confusion matrix
(to show understanding of what each metric means) and cross-checked with
sklearn. Regression metrics likewise.

WHY RECALL MATTERS MORE HERE
------------------------------
In network monitoring, a False Negative (predicting "no failure" when a
device actually fails) means an outage goes undetected until it happens -
potentially causing service downtime. A False Positive (predicting
failure risk when the device is actually fine) just costs an engineer a
few minutes of unnecessary investigation. The cost of the two error types
is asymmetric, so this project prioritizes RECALL (catching real
failures) over raw accuracy, while still tracking precision so alert
fatigue is visible and can be tuned via the classification threshold.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
)


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def classification_report_manual(y_true, y_pred, y_proba=None):
    """
    accuracy  = (TP+TN) / total                -> overall correctness
    precision = TP / (TP+FP)                   -> of predicted failures, how many were real
    recall    = TP / (TP+FN)                   -> of real failures, how many we caught
    f1        = 2 * precision*recall/(precision+recall)  -> harmonic mean, balances both
    """
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    result = dict(
        accuracy=accuracy, precision=precision, recall=recall, f1_score=f1,
        true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn,
    )
    if y_proba is not None and len(np.unique(y_true)) > 1:
        result["roc_auc"] = roc_auc_score(y_true, y_proba)
    return result


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def get_roc_curve(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr, thresholds


# ---------------- Regression metrics ----------------

def regression_report(y_true, y_pred):
    """
    MAE  = mean(|y - y_hat|)             -> average magnitude of error, same units as target
    RMSE = sqrt(mean((y - y_hat)^2))     -> penalizes large errors more than MAE
    R^2  = 1 - SS_res/SS_tot             -> proportion of variance explained by the model
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return dict(mae=mae, rmse=rmse, r2=r2)
