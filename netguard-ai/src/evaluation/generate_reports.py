"""Generate the analytics plots (confusion matrix, ROC curve, feature
importance, model comparison) referenced in the README and usable
standalone outside Streamlit."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
MODELS = BASE / "models"
REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)


def plot_confusion_matrix():
    cm = np.load(MODELS / "confusion_matrix.npy")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Failure", "Failure"])
    ax.set_yticklabels(["No Failure", "Failure"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Regularized Logistic Regression")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def plot_roc_curve():
    data = np.load(MODELS / "roc_curve.npz")
    fpr, tpr = data["fpr"], data["tpr"]
    auc = np.trapezoid(tpr, fpr)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC={auc:.3f})", color="#2563eb")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Failure Classification")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "roc_curve.png", dpi=120)
    plt.close(fig)


def plot_feature_importance():
    imp = pd.read_csv(MODELS / "feature_importance.csv", index_col=0).head(15)
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#dc2626" if v > 0 else "#2563eb" for v in imp["coefficient"]]
    ax.barh(imp.index[::-1], imp["coefficient"][::-1], color=colors[::-1])
    ax.set_xlabel("Logistic Regression Coefficient (standardized)")
    ax.set_title("Top 15 Failure-Risk Indicators")
    fig.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=120)
    plt.close(fig)


def plot_model_comparison():
    with open(REPORTS / "classification_results.json") as f:
        results = json.load(f)
    models = ["logreg_baseline_manual", "logreg_baseline_sklearn",
              "logreg_regularized_manual", "logreg_regularized_sklearn"]
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(metrics):
        vals = [results[mod][m] for mod in models]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(["Baseline\n(manual)", "Baseline\n(sklearn)",
                          "Regularized\n(manual)", "Regularized\n(sklearn)"])
    ax.set_ylim(0, 1)
    ax.set_title("Classification Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS / "model_comparison.png", dpi=120)
    plt.close(fig)

    df = pd.DataFrame({mod: results[mod] for mod in models}).T
    df.to_csv(REPORTS / "model_comparison_table.csv")


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_roc_curve()
    plot_feature_importance()
    plot_model_comparison()
    print(f"Reports saved to {REPORTS}")
