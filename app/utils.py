"""Streamlit app helpers: inference and test-set evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import transforms

from src.data_pipeline import load_split_dataframe
from src.model import build_cnn
from src.utils import class_to_idx, get_device, load_config, resolve_path

CLASS_LABELS = {
    "defect": "NUQSONLI",
    "normal": "NUQSONSIZ",
}


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_checkpoint_info(checkpoint_path: Path | str) -> dict[str, Any]:
    path = resolve_path(str(checkpoint_path)) if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
    if not path.exists():
        return {}
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "val_acc": float(ckpt.get("val_acc", 0.0)),
        "val_loss": float(ckpt.get("val_loss", 0.0)),
    }


def load_model_for_inference(cfg: dict | None = None):
    cfg = cfg or load_config()
    device = get_device(cfg["training"]["device"])
    ckpt_path = resolve_path(cfg["paths"]["cnn_checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model topilmadi: {ckpt_path}. Avval train.py ishga tushiring.")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = cfg["data"]["classes"]
    model = build_cnn(len(classes), cfg["training"]["dropout"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, classes, cfg, device, state, ckpt_path


def _preprocess(img: Image.Image, size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(img).unsqueeze(0)


def run_inference(
    image: Image.Image,
    model: torch.nn.Module,
    classes: list[str],
    cfg: dict,
    device: torch.device,
) -> dict[str, Any]:
    tensor = _preprocess(image, cfg["data"]["image_size"]).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    prob_map = dict(zip(classes, probs))
    pred_class = max(prob_map, key=prob_map.get)
    confidence = prob_map[pred_class]
    defect_idx = classes.index("defect") if "defect" in classes else 0
    return {
        "pred_class": pred_class,
        "label": CLASS_LABELS.get(pred_class, pred_class.upper()),
        "confidence": confidence * 100,
        "prob_map": prob_map,
        "is_defect": pred_class == "defect",
        "defect_prob": prob_map[classes[defect_idx]],
    }


@torch.no_grad()
def evaluate_test_set(
    cfg: dict,
    model: torch.nn.Module,
    device: torch.device,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    classes = cfg["data"]["classes"]
    cmap = class_to_idx(classes)
    img_col = cfg["data"]["image_column"]
    lbl_col = cfg["data"]["label_column"]
    raw_dir = resolve_path(cfg["data"]["raw_images_dir"])
    size = cfg["data"]["image_size"]
    defect_idx = cmap.get("defect", 0)

    test_df = load_split_dataframe("test", cfg)
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    y_true_list: list[int] = []
    y_pred_list: list[int] = []
    defect_probs: list[float] = []
    rows: list[dict[str, Any]] = []
    by_class: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "errors": 0, "rows": []}
    )

    total = len(test_df)
    for idx, (_, row) in enumerate(test_df.iterrows()):
        if progress_callback:
            progress_callback((idx + 1) / total, str(row[img_col]))

        path = raw_dir / row[img_col]
        true_name = row[lbl_col]
        true_idx = cmap[true_name]
        image = Image.open(path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_name = classes[pred_idx]
        is_correct = pred_idx == true_idx

        y_true_list.append(true_idx)
        y_pred_list.append(pred_idx)
        defect_probs.append(float(probs[defect_idx]))

        item = {
            "filename": row[img_col],
            "path": str(path),
            "true_class": true_name,
            "pred_class": pred_name,
            "is_correct": is_correct,
            "confidence": float(probs[pred_idx]) * 100,
            "defect_prob": float(probs[defect_idx]),
            "label": true_idx,
            "prediction": pred_idx,
        }
        rows.append(item)

        cat = by_class[true_name]
        cat["total"] += 1
        cat["correct"] += int(is_correct)
        cat["errors"] += int(not is_correct)
        cat["rows"].append(item)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    scores = np.array(defect_probs)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        normal_idx = 1 - defect_idx
        tp = int(cm[defect_idx, defect_idx])
        fn = int(cm[defect_idx, normal_idx])
        fp = int(cm[normal_idx, defect_idx])
        tn = int(cm[normal_idx, normal_idx])
    else:
        tn = fp = fn = tp = 0

    per_class = [
        {"class": cls, **report[cls]}
        for cls in classes
        if cls in report
    ]

    overall_correct = sum(1 for r in rows if r["is_correct"])
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "scores": scores,
        "metrics": metrics,
        "classification_report": report,
        "per_class": per_class,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "rows": rows,
        "by_class": dict(by_class),
        "overall": {
            "total": len(rows),
            "correct": overall_correct,
            "accuracy": overall_correct / max(len(rows), 1),
        },
        "defect_idx": defect_idx,
        "classes": classes,
    }
