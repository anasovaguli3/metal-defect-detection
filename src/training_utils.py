"""Shared training helpers for CNN and baseline."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from src.data_pipeline import load_split_dataframe

BAR_FMT = (
    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
)


def format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def delta_str(current: float, previous: float | None, *, higher_is_better: bool = True) -> str:
    if previous is None:
        return "—"
    diff = current - previous
    improved = diff > 0 if higher_is_better else diff < 0
    sign = "+" if diff >= 0 else ""
    arrow = "↑" if improved else "↓"
    return f"{sign}{diff:.4f} {arrow}"


def print_epoch_summary(
    *,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    elapsed: float,
    eta: float | None,
    best_val_acc: float,
    saved_checkpoint: bool,
    lr: float,
    prev: dict | None = None,
) -> None:
    lines = [
        f"  ┌─ Epoch {epoch}/{total_epochs} metrics ─────────────────────────",
        f"  │  Train   loss {train_loss:.4f}   acc {pct(train_acc)}"
        + (f"   (Δ acc {delta_str(train_acc, prev['train_acc'])})" if prev else ""),
        f"  │  Val     loss {val_loss:.4f}   acc {pct(val_acc)}"
        + (f"   (Δ acc {delta_str(val_acc, prev['val_acc'])})" if prev else ""),
        f"  │  Gap     train-val acc {pct(train_acc - val_acc)}"
        + ("  (overfitting)" if train_acc - val_acc > 0.05 else ""),
        f"  │  Best    val acc {pct(best_val_acc)}"
        + ("   checkpoint saved ✓" if saved_checkpoint else ""),
        f"  │  LR      {lr:.2e}",
        f"  │  Time    {elapsed:.1f}s"
        + (f"   remaining ~{format_eta(eta)}" if eta else "   training complete"),
        "  └──────────────────────────────────────────────────────────────",
    ]
    for line in lines:
        tqdm.write(line)


def class_weights(cfg: dict, device: torch.device) -> torch.Tensor | None:
    if not cfg["training"].get("use_class_weights", False):
        return None
    df = load_split_dataframe("train", cfg)
    lbl = cfg["data"]["label_column"]
    classes = cfg["data"]["classes"]
    counts = df[lbl].value_counts()
    total = len(df)
    weights = [total / (len(classes) * counts.get(c, 1)) for c in classes]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    *,
    epoch: int,
    total_epochs: int,
    phase: str,
):
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    desc = f"Epoch {epoch}/{total_epochs} | {phase}"

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(
            loader,
            desc=desc,
            leave=True,
            unit="batch",
            bar_format=BAR_FMT,
        )
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size
            pbar.set_postfix(
                loss=f"{total_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
                refresh=False,
            )

    return total_loss / max(total, 1), correct / max(total, 1)
