#!/usr/bin/env python3
"""Single-image prediction CLI."""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import build_cnn
from src.utils import get_device, load_config, resolve_path


def predict(image_path: str, cfg: dict) -> dict:
    device = get_device(cfg["training"]["device"])
    classes = cfg["data"]["classes"]
    ckpt = torch.load(resolve_path(cfg["paths"]["cnn_checkpoint"]), map_location=device, weights_only=False)
    model = build_cnn(len(classes), cfg["training"]["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    size = cfg["data"]["image_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {"class": classes[pred_idx], "probabilities": dict(zip(classes, probs))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Rasm yo'li")
    args = parser.parse_args()
    cfg = load_config()
    out = predict(args.image, cfg)
    print(out)


if __name__ == "__main__":
    main()
