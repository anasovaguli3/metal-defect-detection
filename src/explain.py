"""Grad-CAM style feature map visualization for defect and normal samples."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data_pipeline import load_split_dataframe
from src.model import build_cnn
from src.utils import ensure_dir, get_device, resolve_path, set_seed


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def _load_raw_image(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((size, size))


def generate_feature_maps(cfg: dict, checkpoint: Path | str | None = None) -> list[str]:
    set_seed(cfg.get("random_seed", 42))
    device = get_device(cfg["training"]["device"])
    classes = cfg["data"]["classes"]
    ckpt = Path(checkpoint or cfg["paths"]["cnn_checkpoint"])
    out_dir = resolve_path(cfg["paths"]["feature_maps_dir"])
    ensure_dir(out_dir)

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_cnn(len(classes), cfg["training"]["dropout"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    target_layer = None
    for module in reversed(list(model.features.modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    if target_layer is None:
        raise RuntimeError("Grad-CAM uchun Conv2d topilmadi.")
    cam = GradCAM(model, target_layer)

    test_df = load_split_dataframe("test", cfg)
    img_col = cfg["data"]["image_column"]
    lbl_col = cfg["data"]["label_column"]
    raw_dir = resolve_path(cfg["data"]["raw_images_dir"])
    size = cfg["data"]["image_size"]

    saved = []
    for label_name in classes:
        subset = test_df[test_df[lbl_col] == label_name]
        if subset.empty:
            continue
        row = subset.iloc[0]
        fname = row[img_col]
        pil = _load_raw_image(raw_dir / fname, size)

        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        tensor = transform(pil).unsqueeze(0).to(device)
        class_idx = classes.index(label_name)
        heatmap = cam.generate(tensor, class_idx)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pil)
        axes[0].set_title(f"Original ({label_name})")
        axes[0].axis("off")
        axes[1].imshow(pil)
        axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[1].set_title(f"Grad-CAM ({label_name})")
        axes[1].axis("off")
        out_path = out_dir / f"feature_map_{label_name}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(str(out_path))

    return saved
