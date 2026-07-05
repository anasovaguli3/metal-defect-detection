"""Feature map and filter visualization for DefectCNN Streamlit tab."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def feature_layer_names() -> list[str]:
    return ["conv1_32", "conv2_64", "conv3_128", "conv4_256"]


def _feature_targets(model: nn.Module) -> dict[str, nn.Module]:
    return {
        "conv1_32": model.features[0],
        "conv2_64": model.features[4],
        "conv3_128": model.features[8],
        "conv4_256": model.features[12],
    }


def _layer_hook(name: str, store: dict[str, torch.Tensor]):
    def hook(_module, _inputs, output):
        store[name] = output.detach().cpu()

    return hook


def capture_feature_maps(
    model: nn.Module,
    image: Image.Image,
    image_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)
    store: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in _feature_targets(model).items():
        handles.append(module.register_forward_hook(_layer_hook(name, store)))

    model.eval()
    with torch.no_grad():
        model(tensor)
    for handle in handles:
        handle.remove()

    return {name: act[0].numpy() for name, act in store.items()}


def extract_filter_weights(model: nn.Module) -> np.ndarray:
    conv = model.features[0]
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
