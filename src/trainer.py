"""Training loop with history logging and checkpointing."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.model import build_cnn
from src.data_pipeline import load_split_dataframe, make_dataloader
from src.utils import ensure_dir, get_device, load_json, save_json, set_seed


def _class_weights(cfg: dict, device: torch.device) -> torch.Tensor | None:
    if not cfg["training"].get("use_class_weights", False):
        return None
    df = load_split_dataframe("train", cfg)
    lbl = cfg["data"]["label_column"]
    classes = cfg["data"]["classes"]
    counts = df[lbl].value_counts()
    total = len(df)
    weights = [total / (len(classes) * counts.get(c, 1)) for c in classes]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False, desc="train" if train else "val"):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_cnn(
    cfg: dict,
    batch_size: int | None = None,
    epochs: int | None = None,
    checkpoint_path: Path | str | None = None,
    history_path: Path | str | None = None,
    resume: bool = False,
) -> dict:
    set_seed(cfg.get("random_seed", 42))
    device = get_device(cfg["training"]["device"])
    classes = cfg["data"]["classes"]
    bs = batch_size or cfg["training"]["batch_size"]
    ep = epochs or cfg["training"]["epochs"]
    ckpt = Path(checkpoint_path or cfg["paths"]["cnn_checkpoint"])
    hist_path = Path(history_path or cfg["paths"]["cnn_history"])
    ensure_dir(ckpt.parent)
    ensure_dir(hist_path.parent)

    train_loader = make_dataloader("train", cfg, batch_size=bs, train=True)
    val_loader = make_dataloader("val", cfg, batch_size=bs, train=False, shuffle=False)

    model = build_cnn(len(classes), cfg["training"]["dropout"]).to(device)
    cw = _class_weights(cfg, device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = None
    if cfg["training"].get("scheduler") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

    start_epoch = 0
    best_val_acc = 0.0
    history: list[dict] = []

    if resume and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = state.get("epoch", 0) + 1
        best_val_acc = state.get("best_val_acc", 0.0)
        if hist_path.exists():
            history = load_json(hist_path)

    for epoch in range(start_epoch, ep):
        t0 = time.time()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, False)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "seconds": round(elapsed, 2),
            "batch_size": bs,
        }
        history.append(record)
        save_json(history, hist_path)
        print(
            f"Epoch {epoch+1}/{ep} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "classes": classes,
                    "config": {"batch_size": bs, "dropout": cfg["training"]["dropout"]},
                },
                ckpt,
            )

    return {"best_val_acc": best_val_acc, "checkpoint": str(ckpt), "history": str(hist_path)}
