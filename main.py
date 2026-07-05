"""FastAPI inference service for the metal defect classifier."""

from __future__ import annotations

import io

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.model import build_cnn
from src.utils import get_device, load_config, resolve_path

cfg = load_config()
device = get_device(cfg["training"]["device"])
checkpoint = torch.load(
    resolve_path(cfg["paths"]["cnn_checkpoint"]), map_location=device, weights_only=False
)
classes = checkpoint.get("classes", cfg["data"]["classes"])
model = build_cnn(len(classes), cfg["training"]["dropout"]).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

size = cfg["data"]["image_size"]
transform = transforms.Compose(
    [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

app = FastAPI(
    title="Metal Defect Detection API",
    description="Ikki klassli tasnif: metall yuzasida `defect` yoki `normal`.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"service": "metal-defect-detection", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "classes": classes}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Fayl rasm emas yoki buzilgan.")

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()

    pred_idx = max(range(len(probs)), key=probs.__getitem__)
    return {
        "class": classes[pred_idx],
        "confidence": round(probs[pred_idx], 4),
        "probabilities": {c: round(p, 4) for c, p in zip(classes, probs)},
    }
