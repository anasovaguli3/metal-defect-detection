"""Test-set evaluation and confusion matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_pipeline import make_dataloader
from src.model import build_cnn
from src.utils import ensure_dir, get_device, load_config, save_json, set_seed


@torch.no_grad()
def evaluate_model(
    cfg: dict,
    checkpoint_path: Path | str | None = None,
    batch_size: int | None = None,
    output_metrics: Path | str | None = None,
    output_plot: Path | str | None = None,
) -> dict:
    set_seed(cfg.get("random_seed", 42))
    device = get_device(cfg["training"]["device"])
    classes = cfg["data"]["classes"]
    ckpt_path = Path(checkpoint_path or cfg["paths"]["cnn_checkpoint"])
    metrics_path = Path(output_metrics or cfg["paths"]["test_metrics"])
    plot_path = Path(output_plot or cfg["paths"]["confusion_matrix"])

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint topilmadi: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_cnn(len(classes), cfg["training"]["dropout"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    loader = make_dataloader("test", cfg, batch_size=batch_size, train=False, shuffle=False)
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
        ),
    }
    save_json(metrics, metrics_path)

    cm = confusion_matrix(y_true, y_pred)
    ensure_dir(plot_path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — CNN")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {"metrics": metrics, "metrics_path": str(metrics_path), "plot_path": str(plot_path)}


def compare_models_csv(rows: list[dict], path: Path | str) -> None:
    import pandas as pd

    ensure_dir(Path(path).parent)
    pd.DataFrame(rows).to_csv(path, index=False)
