"""Streamlit app helpers for DefectVision inference."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.dataset import build_test_samples, create_test_dataloader, get_processed_paths

from src.augmentation import denormalize, get_val_test_transforms
from src.evaluate import _collect_predictions, compute_pixel_baseline_scores, resolve_anomaly_threshold
from src.model import build_model, get_model_architecture, get_model_checkpoint_path

# Both checkpoints kept on disk — switch in Streamlit sidebar.
MODEL_PRESETS: dict[str, dict[str, str]] = {
    "conv_ae": {
        "label": "Conv AE (production, epoch 112)",
        "architecture": "conv_ae",
        "checkpoint": "models/conv_ae_best.pth",
        "name": "DefectVisionAE",
    },
    "unet": {
        "label": "U-Net (custom, ~epoch 100+)",
        "architecture": "unet",
        "checkpoint": "models/unet_best.pth",
        "name": "DefectVisionUNet",
    },
}
from src.utils import (
    compute_classification_metrics,
    compute_per_defect_metrics,
    compute_reconstruction_error,
    find_balanced_threshold,
    find_optimal_threshold,
    generate_heatmap,
    get_device,
    get_score_settings,
    load_checkpoint,
    load_threshold_json,
    load_threshold_metadata,
    overlay_heatmap,
    parse_threshold_setting,
    resolve_path,
    threshold_metadata_matches,
)


def apply_model_preset(config: dict[str, Any], preset_key: str) -> dict[str, Any]:
    """Return config copy wired to a saved checkpoint preset."""
    if preset_key not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {preset_key}")
    preset = MODEL_PRESETS[preset_key]
    cfg = copy.deepcopy(config)
    cfg["model"]["architecture"] = preset["architecture"]
    cfg["model"]["name"] = preset["name"]
    cfg["app"]["model_path"] = preset["checkpoint"]
    cfg["evaluation"]["model_path"] = preset["checkpoint"]
    return cfg


def get_checkpoint_info(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read epoch / val_loss / architecture from a .pth file."""
    import torch

    path = resolve_path(checkpoint_path)
    if not path.exists():
        return {}
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "val_loss": float(ckpt.get("loss", 0.0)),
        "architecture": str(ckpt.get("architecture", "")),
    }


def get_inference_threshold(
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[float, str]:
    """Threshold for Streamlit: validate saved json, else recompute on val/good."""
    setting = parse_threshold_setting(config.get("app", {}).get("threshold", "auto"))
    if setting != "auto":
        return float(setting), "config app.threshold"

    model_path = resolve_path(get_model_checkpoint_path(config))
    meta = load_threshold_metadata()
    if meta is not None and threshold_metadata_matches(meta, config, model_path):
        source = str(meta.get("source", "threshold.json"))
        return float(meta["threshold"]), source

    threshold, source = resolve_anomaly_threshold(config, model, device, section="app")
    return threshold, f"{source} (recomputed — threshold.json mismatch)"


def load_model_for_inference(config: dict[str, Any]) -> tuple[torch.nn.Module, torch.device]:
    """Load trained autoencoder for Streamlit inference."""
    device = get_device(config.get("training", {}).get("device", "auto"))
    model_path = resolve_path(get_model_checkpoint_path(config))

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first."
        )

    model = build_model(config).to(device)
    load_checkpoint(model_path, model)
    model.eval()
    return model, device


def run_inference(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """
    Run autoencoder inference on a PIL image.

    Returns:
        Dict with label, error, confidence, and visualization arrays.
    """
    transform = get_val_test_transforms(config)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed = model(tensor)

    orig_cpu = tensor[0].cpu()
    recon_cpu = reconstructed[0].cpu()
    score_cfg = get_score_settings(config)
    error = compute_reconstruction_error(
        orig_cpu,
        recon_cpu,
        method=score_cfg["method"],
        topk_percent=score_cfg["topk_percent"],
    )
    is_defect = error >= threshold
    label = "DEFECT" if is_defect else "GOOD"
    confidence = min(abs(error - threshold) / (threshold + 1e-8) * 100, 100.0)

    orig_display = denormalize(orig_cpu).permute(1, 2, 0).numpy()
    recon_display = denormalize(recon_cpu).permute(1, 2, 0).numpy()
    heatmap = generate_heatmap(orig_display, recon_display)

    return {
        "label": label,
        "is_defect": is_defect,
        "error": error,
        "confidence": confidence,
        "original": orig_display,
        "reconstructed": recon_display,
        "heatmap": heatmap,
        "overlay": overlay_heatmap(orig_display, heatmap),
    }


# Display order: good first, then defect types alphabetically.
CATEGORY_ORDER = [
    "good",
    "bent_wire",
    "cable_swap",
    "combined",
    "cut_inner_insulation",
    "cut_outer_insulation",
    "missing_cable",
    "missing_wire",
    "poke_insulation",
]

CATEGORY_TITLES: dict[str, str] = {
    "good": "Good (норма)",
    "bent_wire": "Bent wire (согнутые провода)",
    "cable_swap": "Cable swap (перепутаны)",
    "combined": "Combined (несколько дефектов)",
    "cut_inner_insulation": "Cut inner insulation (порез внутри)",
    "cut_outer_insulation": "Cut outer insulation (порез снаружи)",
    "missing_cable": "Missing cable (нет кабеля)",
    "missing_wire": "Missing wire (нет жилы)",
    "poke_insulation": "Poke insulation (прокол изоляции)",
}


def _category_sort_key(name: str) -> tuple[int, str]:
    if name in CATEGORY_ORDER:
        return (CATEGORY_ORDER.index(name), name)
    return (len(CATEGORY_ORDER), name)


def evaluate_test_set_by_category(
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    *,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Run inference on the full MVTec cable test set, grouped by folder/category.

    Returns per-category predicted GOOD/DEFECT counts and per-image rows.
    """
    paths = get_processed_paths(config)
    samples = build_test_samples(paths["test"])
    if not samples:
        raise FileNotFoundError(
            f"No test images in {paths['test']}. Run scripts/prepare_data.py first."
        )

    test_loader = create_test_dataloader(config)
    if progress_callback is not None:
        progress_callback(0.5, "model inference")

    labels_arr, scores_arr, defect_types, _, _ = _collect_predictions(
        model, test_loader, device, config
    )

    if progress_callback is not None:
        progress_callback(0.75, "baseline")

    baseline_scores = compute_pixel_baseline_scores(create_test_dataloader(config))

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = len(samples)

    for idx, (img_path, label, defect_type) in enumerate(samples):
        if progress_callback is not None:
            progress_callback(0.75 + 0.25 * (idx + 1) / total, defect_type)

        error = float(scores_arr[idx])
        is_defect = error >= threshold
        is_correct = (is_defect and label == 1) or (not is_defect and label == 0)
        confidence = min(abs(error - threshold) / (threshold + 1e-8) * 100, 100.0)
        by_category[defect_type].append(
            {
                "filename": img_path.name,
                "path": str(img_path),
                "label": label,
                "ground_truth": "good" if label == 0 else "defect",
                "prediction": "defect" if is_defect else "good",
                "is_defect": is_defect,
                "is_correct": is_correct,
                "error": error,
                "confidence": confidence,
            }
        )

    categories: dict[str, dict[str, Any]] = {}
    overall_correct = 0
    overall_total = 0

    for defect_type in sorted(by_category.keys(), key=_category_sort_key):
        rows = by_category[defect_type]
        n = len(rows)
        pred_good = sum(1 for r in rows if not r["is_defect"])
        pred_defect = n - pred_good
        correct = sum(1 for r in rows if r["is_correct"])
        is_good_bucket = defect_type == "good"

        categories[defect_type] = {
            "title": CATEGORY_TITLES.get(defect_type, defect_type),
            "total": n,
            "pred_good": pred_good,
            "pred_defect": pred_defect,
            "correct": correct,
            "errors": n - correct,
            "ground_truth": "good" if is_good_bucket else "defect",
            "recall_or_specificity": correct / n if n else 0.0,
            "mean_error": sum(r["error"] for r in rows) / n if n else 0.0,
            "rows": rows,
        }
        overall_correct += correct
        overall_total += n

    pred_good_all = sum(c["pred_good"] for c in categories.values())
    pred_defect_all = sum(c["pred_defect"] for c in categories.values())

    all_rows = [row for rows in by_category.values() for row in rows]
    y_true = labels_arr.astype(int)
    y_pred = np.array([int(r["is_defect"]) for r in all_rows], dtype=int)
    scores = scores_arr.astype(float)
    cls_metrics = compute_classification_metrics(y_true, y_pred, scores)
    per_defect = compute_per_defect_metrics(defect_types, y_true, y_pred, scores)

    baseline_thr = find_optimal_threshold(y_true, baseline_scores)
    baseline_preds = (baseline_scores >= baseline_thr).astype(int)
    baseline_metrics = compute_classification_metrics(y_true, baseline_preds, baseline_scores)

    eval_cfg = config.get("evaluation", {})
    alt_target = int(eval_cfg.get("max_total_errors", 12))
    alt_threshold, alt_meta = find_balanced_threshold(y_true, scores, alt_target)
    alt_preds = (scores >= alt_threshold).astype(int)
    alt_metrics = compute_classification_metrics(y_true, alt_preds, scores)
    optimal_threshold = find_optimal_threshold(y_true, scores)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    n_good = int(np.sum(y_true == 0))
    n_defect = int(np.sum(y_true == 1))

    per_defect_recall: list[dict[str, Any]] = []
    for defect_type in sorted(by_category.keys(), key=_category_sort_key):
        if defect_type == "good":
            continue
        cat = categories[defect_type]
        per_defect_recall.append(
            {
                "category": defect_type,
                "title": cat["title"],
                "count": cat["total"],
                "recall": cat["correct"] / cat["total"] if cat["total"] else 0.0,
                "detected": cat["correct"],
                "missed": cat["errors"],
            }
        )

    return _build_eval_payload(
        threshold=threshold,
        paths=paths,
        by_category=by_category,
        categories=categories,
        overall_correct=overall_correct,
        overall_total=overall_total,
        pred_good_all=pred_good_all,
        pred_defect_all=pred_defect_all,
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
        cls_metrics=cls_metrics,
        per_defect=per_defect,
        baseline_metrics=baseline_metrics,
        alt_target=alt_target,
        alt_threshold=alt_threshold,
        alt_meta=alt_meta,
        alt_metrics=alt_metrics,
        optimal_threshold=optimal_threshold,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        n_good=n_good,
        n_defect=n_defect,
        per_defect_recall=per_defect_recall,
    )


def _build_eval_payload(
    *,
    threshold: float,
    paths,
    by_category,
    categories,
    overall_correct,
    overall_total,
    pred_good_all,
    pred_defect_all,
    y_true,
    y_pred,
    scores,
    cls_metrics,
    per_defect,
    baseline_metrics,
    alt_target,
    alt_threshold,
    alt_meta,
    alt_metrics,
    optimal_threshold,
    tn,
    fp,
    fn,
    tp,
    n_good,
    n_defect,
    per_defect_recall,
) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "test_root": str(paths["test"]),
        "overall": {
            "total": overall_total,
            "pred_good": pred_good_all,
            "pred_defect": pred_defect_all,
            "correct": overall_correct,
            "accuracy": overall_correct / overall_total if overall_total else 0.0,
            "n_good": n_good,
            "n_defect": n_defect,
        },
        "metrics": cls_metrics,
        "baseline_metrics": baseline_metrics,
        "alt_metrics": alt_metrics,
        "alt_threshold": alt_threshold,
        "alt_threshold_source": (
            f"balanced (target<={alt_target}, errors={alt_meta['total_errors']})"
        ),
        "optimal_threshold": optimal_threshold,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "per_defect_recall": per_defect_recall,
        "per_defect": per_defect,
        "y_true": y_true,
        "y_pred": y_pred,
        "scores": scores,
        "categories": categories,
    }


def evaluate_model_preset(
    base_config: dict[str, Any],
    preset_key: str,
    *,
    progress_callback=None,
) -> dict[str, Any]:
    """Evaluate one saved checkpoint preset with its own threshold."""
    config = apply_model_preset(base_config, preset_key)
    device = get_device(config.get("training", {}).get("device", "auto"), log=False)
    model = build_model(config).to(device)
    load_checkpoint(resolve_path(get_model_checkpoint_path(config)), model)
    model.eval()
    threshold, _ = get_inference_threshold(config, model, device)
    return evaluate_test_set_by_category(
        config,
        model,
        device,
        threshold,
        progress_callback=progress_callback,
    )


def compare_custom_models(
    base_config: dict[str, Any],
    *,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Run full test evaluation for Conv AE and U-Net using each model's threshold.

    Returns metrics for both custom architectures (no VGG/ResNet).
    """
    results: dict[str, dict[str, Any]] = {}
    for idx, preset in enumerate(("conv_ae", "unet")):
        if progress_callback is not None:
            progress_callback(idx / 2, f"Evaluating {preset}")

        def _cb(fraction: float, label: str, _preset=preset, _idx=idx) -> None:
            if progress_callback is not None:
                progress_callback(_idx / 2 + fraction / 2, f"{_preset}: {label}")

        results[preset] = evaluate_model_preset(
            base_config,
            preset,
            progress_callback=_cb,
        )

    if progress_callback is not None:
        progress_callback(1.0, "done")

    return {
        "conv_ae": results["conv_ae"],
        "unet": results["unet"],
        "conv_ae_metrics": results["conv_ae"]["metrics"],
        "unet_metrics": results["unet"]["metrics"],
    }
