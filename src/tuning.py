"""Hyperparameter tuning: batch size 8 vs 16."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.evaluate import compare_models_csv
from src.trainer import train_cnn
from src.utils import ensure_dir, get_device, resolve_path, set_seed


def run_batch_tuning(cfg: dict, skip_trained: bool = False) -> pd.DataFrame:
    set_seed(cfg.get("random_seed", 42))
    tuning_cfg = cfg.get("tuning", {})
    batch_sizes = tuning_cfg.get("batch_sizes", [8, 16])
    epochs = tuning_cfg.get("epochs", 15)
    tuning_dir = resolve_path(cfg["paths"]["tuning_dir"])
    ensure_dir(tuning_dir)

    rows = []
    for bs in batch_sizes:
        ckpt = tuning_dir / f"cnn_bs{bs}_best.pth"
        hist = tuning_dir / f"history_bs{bs}.json"

        if skip_trained and ckpt.exists() and hist.exists():
            print(f"Skip: batch_size={bs} allaqachon mavjud.")
        else:
            print(f"Tuning: batch_size={bs}, epochs={epochs}")
            train_cnn(
                cfg,
                batch_size=bs,
                epochs=epochs,
                checkpoint_path=ckpt,
                history_path=hist,
                resume=False,
            )

        result = _eval_tuning_checkpoint(cfg, ckpt, bs)
        rows.append(result)

    comparison_path = resolve_path(cfg["paths"]["tuning_comparison"])
    df = pd.DataFrame(rows)
    df.to_csv(comparison_path, index=False)
    print(f"Tuning comparison: {comparison_path}")
    return df


@torch.no_grad()
def _eval_tuning_checkpoint(cfg: dict, ckpt: Path, batch_size: int) -> dict:
    from src.data_pipeline import make_dataloader
    from src.model import build_cnn
    from sklearn.metrics import accuracy_score, f1_score

    device = get_device(cfg["training"]["device"])
    classes = cfg["data"]["classes"]
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_cnn(len(classes), cfg["training"]["dropout"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    loader = make_dataloader("val", cfg, batch_size=batch_size, train=False, shuffle=False)
    preds, labels = [], []
    for images, y in loader:
        out = model(images.to(device))
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())

    return {
        "batch_size": batch_size,
        "val_accuracy": round(float(accuracy_score(labels, preds)), 4),
        "val_f1_weighted": round(float(f1_score(labels, preds, average="weighted", zero_division=0)), 4),
        "checkpoint": str(ckpt),
        "best_val_acc_saved": state.get("best_val_acc"),
    }
