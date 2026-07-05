"""EfficientNet-B0 baseline."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from src.data_pipeline import make_dataloader
from src.trainer import _class_weights
from src.utils import ensure_dir, get_device, load_json, save_json, set_seed


def build_efficientnet(num_classes: int, name: str = "efficientnet_b0") -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * images.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_baseline(cfg: dict, resume: bool = False) -> dict:
    set_seed(cfg.get("random_seed", 42))
    bcfg = cfg.get("baseline", {})
    device = get_device(cfg["training"]["device"])
    classes = cfg["data"]["classes"]
    bs = bcfg.get("batch_size", 16)
    ep = bcfg.get("epochs", 20)
    lr = bcfg.get("learning_rate", 5e-4)
    ckpt = Path(cfg["paths"]["baseline_checkpoint"])
    ensure_dir(ckpt.parent)

    train_loader = make_dataloader("train", cfg, batch_size=bs, train=True)
    val_loader = make_dataloader("val", cfg, batch_size=bs, train=False, shuffle=False)
    test_loader = make_dataloader("test", cfg, batch_size=bs, train=False, shuffle=False)

    model = build_efficientnet(len(classes), bcfg.get("model_name", "efficientnet_b0")).to(device)
    cw = _class_weights(cfg, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = 0.0
    history = []
    start = 0
    if resume and ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model_state"])
        optimizer.load_state_dict(st["optimizer_state"])
        best_val = st.get("best_val_acc", 0.0)
        start = st.get("epoch", 0) + 1

    for epoch in range(start, ep):
        t0 = time.time()
        tr_loss, tr_acc = _epoch(model, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = _epoch(model, val_loader, criterion, optimizer, device, False)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(tr_loss, 5),
                "train_acc": round(tr_acc, 5),
                "val_loss": round(va_loss, 5),
                "val_acc": round(va_acc, 5),
                "seconds": round(time.time() - t0, 2),
            }
        )
        print(f"Baseline {epoch+1}/{ep} val_acc={va_acc:.4f}")
        if va_acc >= best_val:
            best_val = va_acc
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "best_val_acc": best_val, "classes": classes},
                ckpt,
            )

    save_json(history, Path(cfg["paths"]["logs_dir"]) / "baseline_history.json")

    # test eval
    st = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(st["model_state"])
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, y in test_loader:
            out = model(images.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    from sklearn.metrics import accuracy_score, f1_score

    baseline_metrics = {
        "model": "efficientnet_b0",
        "test_accuracy": round(float(accuracy_score(labels, preds)), 4),
        "test_f1": round(float(f1_score(labels, preds, average="weighted", zero_division=0)), 4),
    }

    # compare with CNN test metrics if exists
    import pandas as pd
    from src.utils import load_json as lj, resolve_path

    rows = [baseline_metrics]
    cnn_metrics_path = resolve_path(cfg["paths"]["test_metrics"])
    if cnn_metrics_path.exists():
        cnn_m = lj(cnn_metrics_path)
        rows.insert(
            0,
            {
                "model": "custom_cnn",
                "test_accuracy": round(cnn_m.get("accuracy", 0), 4),
                "test_f1": round(cnn_m.get("f1_weighted", 0), 4),
            },
        )
    comp_path = resolve_path(cfg["paths"]["baseline_comparison"])
    pd.DataFrame(rows).to_csv(comp_path, index=False)

    return {"checkpoint": str(ckpt), "baseline_metrics": baseline_metrics, "comparison": str(comp_path)}
