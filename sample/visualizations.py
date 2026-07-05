"""Matplotlib figures for Streamlit evaluation (return fig, do not save)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)


def fig_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Good", "Defect"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def fig_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_score_distribution(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> plt.Figure:
    good = scores[y_true == 0]
    defect = scores[y_true == 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(good, bins=30, alpha=0.65, label=f"Good (n={len(good)})", color="#21c354")
    ax.hist(defect, bins=30, alpha=0.65, label=f"Defect (n={len(defect)})", color="#ff4b4b")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold {threshold:.4f}")
    ax.set_xlabel("Reconstruction error")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_f1_vs_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    optimal_threshold: float | None = None,
) -> plt.Figure:
    candidates = np.linspace(float(scores.min()), float(scores.max()), 100)
    f1s = []
    for t in candidates:
        preds = (scores >= t).astype(int)
        f1s.append(f1_score(y_true, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(candidates, f1s, linewidth=2)
    ax.axvline(threshold, color="green", linestyle=":", linewidth=2, label=f"Current: {threshold:.4f}")
    if optimal_threshold is not None:
        ax.axvline(
            optimal_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Best F1: {optimal_threshold:.4f}",
        )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_per_defect_recall(per_defect_recall: list[dict[str, Any]]) -> plt.Figure:
    names = [row["category"] for row in per_defect_recall]
    recalls = [row["recall"] for row in per_defect_recall]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, recalls, color="#4c8bf5", edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Recall per Defect Type")
    ax.tick_params(axis="x", rotation=35)
    for bar, val in zip(bars, recalls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_models_comparison(
    conv_ae_metrics: dict[str, float],
    unet_metrics: dict[str, float],
) -> plt.Figure:
    """Bar chart comparing Conv AE vs custom U-Net on the same test set."""
    keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    conv_vals = [conv_ae_metrics.get(k, 0.0) for k in keys]
    unet_vals = [unet_metrics.get(k, 0.0) for k in keys]
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, conv_vals, width, label="Conv AE", color="#4c8bf5")
    ax.bar(x + width / 2, unet_vals, width, label="U-Net AE", color="#f5a623")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Conv AE vs U-Net (custom architectures)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_baseline_comparison(
    ae_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> plt.Figure:
    keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    ae_vals = [ae_metrics.get(k, 0.0) for k in keys]
    base_vals = [baseline_metrics.get(k, 0.0) for k in keys]
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, ae_vals, width, label="Autoencoder", color="#4c8bf5")
    ax.bar(x + width / 2, base_vals, width, label="Pixel baseline", color="#aaa")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Autoencoder vs Pixel Baseline")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
