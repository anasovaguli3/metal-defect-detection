"""Feature map and filter visualization for DefectVision Streamlit tab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.augmentation import denormalize, get_val_test_transforms
from src.model import get_model_architecture


def _layer_hook(name: str, store: dict[str, torch.Tensor]):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            store[name] = output[0].detach().cpu()
        else:
            store[name] = output.detach().cpu()

    return hook


def feature_layer_names(architecture: str) -> list[str]:
    """Human-readable layer ids used for activation hooks."""
    if architecture == "unet":
        return ["inc", "down1", "down2", "down3", "bottleneck"]
    return ["enc1", "enc2", "enc3", "enc4"]


def register_feature_hooks(model: nn.Module, architecture: str) -> tuple[list[Any], dict[str, torch.Tensor]]:
    """Attach forward hooks and return (handles, empty activation dict)."""
    store: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    if architecture == "unet":
        targets = {
            "inc": model.inc,
            "down1": model.down1.conv,
            "down2": model.down2.conv,
            "down3": model.down3.conv,
            "bottleneck": model.bottleneck,
        }
    else:
        targets = {
            "enc1": model.enc1,
            "enc2": model.enc2,
            "enc3": model.enc3,
            "enc4": model.enc4,
        }

    for name, module in targets.items():
        handles.append(module.register_forward_hook(_layer_hook(name, store)))
    return handles, store


def capture_feature_maps(
    model: nn.Module,
    image: Image.Image,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run one image through the model and return numpy activations per layer."""
    architecture = get_model_architecture(config)
    transform = get_val_test_transforms(config)
    tensor = transform(image).unsqueeze(0).to(device)

    handles, store = register_feature_hooks(model, architecture)
    model.eval()
    with torch.no_grad():
        model(tensor)
    for handle in handles:
        handle.remove()

    return {name: act[0].numpy() for name, act in store.items()}


def get_first_conv_layer(model: nn.Module, architecture: str) -> nn.Conv2d | None:
    """Return the first learnable conv layer for filter visualization."""
    if architecture == "unet":
        return model.inc.block[0]
    if architecture == "conv_ae":
        return model.enc1[0]
    return None


def extract_filter_weights(model: nn.Module, architecture: str) -> np.ndarray | None:
    """
    First-layer conv weights shaped (out_channels, 3, k, k).
    Returns RGB-rendered filters (out_channels, k, k, 3) for display.
    """
    conv = get_first_conv_layer(model, architecture)
    if conv is None:
        return None

    weights = conv.weight.detach().cpu().numpy()
    rendered = []
    for filt in weights:
        f = filt.transpose(1, 2, 0)
        f = f - f.min()
        denom = f.max() - f.min()
        if denom > 1e-8:
            f = f / denom
        rendered.append(f)
    return np.stack(rendered, axis=0)


def fig_filter_grid(
    filters: np.ndarray,
    *,
    title: str = "First Conv Layer Filters",
    max_filters: int = 32,
    ncol: int = 8,
) -> plt.Figure:
    """Grid of RGB filter kernels from the first convolution."""
    n = min(len(filters), max_filters)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.4, nrow * 1.4))
    axes = np.atleast_2d(axes)
    for idx in range(nrow * ncol):
        r, c = divmod(idx, ncol)
        ax = axes[r, c]
        ax.axis("off")
        if idx < n:
            ax.imshow(np.clip(filters[idx], 0, 1))
            ax.set_title(f"F{idx}", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def fig_feature_map_grid(
    activation: np.ndarray,
    *,
    layer_name: str,
    max_channels: int = 16,
    ncol: int = 4,
) -> plt.Figure:
    """
    Visualize activation maps for one layer.

    Args:
        activation: (C, H, W) tensor.
    """
    channels = activation[:max_channels]
    n = len(channels)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 2.0))
    axes = np.atleast_2d(axes)

    for idx in range(nrow * ncol):
        r, c = divmod(idx, ncol)
        ax = axes[r, c]
        ax.axis("off")
        if idx < n:
            ch = channels[idx]
            ch = ch - ch.min()
            denom = ch.max() - ch.min()
            if denom > 1e-8:
                ch = ch / denom
            ax.imshow(ch, cmap="viridis")
            ax.set_title(f"Ch {idx}", fontsize=8)

    fig.suptitle(f"Feature maps — {layer_name}", fontsize=11)
    fig.tight_layout()
    return fig


def fig_input_vs_deep_maps(
    image: Image.Image,
    activations: dict[str, np.ndarray],
    *,
    shallow_layer: str,
    deep_layer: str,
) -> plt.Figure:
    """Compact figure: original + mean activation of shallow and deep layers."""
    shallow = activations.get(shallow_layer)
    deep = activations.get(deep_layer)
    if shallow is None or deep is None:
        raise ValueError(f"Missing layers {shallow_layer} or {deep_layer}")

    shallow_mean = shallow.mean(axis=0)
    deep_mean = deep.mean(axis=0)

    for arr in (shallow_mean, deep_mean):
        arr -= arr.min()
        denom = arr.max() - arr.min()
        if denom > 1e-8:
            arr /= denom

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(shallow_mean, cmap="magma")
    axes[1].set_title(f"{shallow_layer} (mean)")
    axes[2].imshow(deep_mean, cmap="magma")
    axes[2].set_title(f"{deep_layer} (mean)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def iter_test_images_by_category(
    test_root: Path,
    category_order: list[str],
) -> list[tuple[str, Path]]:
    """Yield (category, path) for all test images sorted by category."""
    rows: list[tuple[str, Path]] = []
    for category in category_order:
        folder = test_root / category
        if not folder.is_dir():
            continue
        paths = sorted(
            p
            for p in folder.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        for path in paths:
            rows.append((category, path))
    # any extra folders not in category_order
    for folder in sorted(test_root.iterdir()):
        if not folder.is_dir() or folder.name in category_order:
            continue
        paths = sorted(
            p
            for p in folder.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        for path in paths:
            rows.append((folder.name, path))
    return rows


def tensor_preview(image: Image.Image, config: dict[str, Any]) -> np.ndarray:
    """RGB numpy preview after val/test transforms."""
    transform = get_val_test_transforms(config)
    tensor = transform(image)
    return denormalize(tensor).permute(1, 2, 0).numpy()
