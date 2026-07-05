"""Matplotlib figures for Streamlit (return fig, do not save)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve


def fig_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — CNN")
    fig.tight_layout()
    return fig


def fig_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2, color="#4c8bf5")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (defect = positive)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_prob_distribution(
    y_true: np.ndarray,
    defect_probs: np.ndarray,
    class_names: list[str],
    defect_idx: int = 0,
) -> plt.Figure:
    normal_idx = 1 - defect_idx if len(class_names) == 2 else 0
    normal_mask = y_true == normal_idx
    defect_mask = y_true == defect_idx
    fig, ax = plt.subplots(figsize=(8, 5))
    if normal_mask.any():
        ax.hist(
            defect_probs[normal_mask],
            bins=30,
            alpha=0.65,
            label=f"{class_names[normal_idx]} (n={normal_mask.sum()})",
            color="#21c354",
        )
    if defect_mask.any():
        ax.hist(
            defect_probs[defect_mask],
            bins=30,
            alpha=0.65,
            label=f"{class_names[defect_idx]} (n={defect_mask.sum()})",
            color="#ff4b4b",
        )
    ax.set_xlabel("P(defect)")
    ax.set_ylabel("Count")
    ax.set_title("Defect probability distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_per_class_bars(per_class: list[dict[str, Any]]) -> plt.Figure:
    names = [row["class"] for row in per_class]
    f1s = [row.get("f1-score", 0) for row in per_class]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, f1s, color="#4c8bf5", edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("F1 per class")
    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_training_history(df_hist: pd.DataFrame) -> tuple[plt.Figure, plt.Figure]:
    fig_loss, ax_l = plt.subplots(figsize=(6, 4))
    ax_l.plot(df_hist["epoch"], df_hist["train_loss"], label="train")
    ax_l.plot(df_hist["epoch"], df_hist["val_loss"], label="val")
    ax_l.set_title("Loss")
    ax_l.set_xlabel("Epoch")
    ax_l.legend()
    ax_l.grid(True, alpha=0.3)
    fig_loss.tight_layout()

    fig_acc, ax_a = plt.subplots(figsize=(6, 4))
    ax_a.plot(df_hist["epoch"], df_hist["train_acc"], label="train")
    ax_a.plot(df_hist["epoch"], df_hist["val_acc"], label="val")
    ax_a.set_title("Accuracy")
    ax_a.set_xlabel("Epoch")
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)
    fig_acc.tight_layout()
    return fig_loss, fig_acc


def fig_models_comparison(
    cnn_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    *,
    cnn_label: str = "CNN",
    baseline_label: str = "Baseline",
) -> plt.Figure:
    keys = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    cnn_vals = [cnn_metrics.get(k, 0.0) for k in keys]
    base_vals = [baseline_metrics.get(k, 0.0) for k in keys]
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, cnn_vals, width, label=cnn_label, color="#4c8bf5")
    ax.bar(x + width / 2, base_vals, width, label=baseline_label, color="#f5a623")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{cnn_label} vs {baseline_label}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
