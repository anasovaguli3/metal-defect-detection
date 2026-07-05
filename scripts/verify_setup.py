#!/usr/bin/env python3
"""Ustoz yoki boshqa kompyuterda loyiha tayyorligini tekshirish."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "XATO"
    msg = f"[{mark}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main() -> int:
    print("=== Loyiha tekshiruvi ===\n")
    all_ok = True

    py_ok = sys.version_info >= (3, 10)
    all_ok &= check("Python 3.10+", py_ok, sys.version.split()[0])

    try:
        import torch

        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "CPU"
        all_ok &= check("PyTorch", True, f"{torch.__version__}, device={device}")
    except ImportError:
        all_ok &= check("PyTorch", False, "pip install -r requirements.txt")

    for pkg in ("pandas", "yaml", "sklearn", "streamlit", "PIL"):
        try:
            __import__(pkg if pkg != "yaml" else "yaml")
            check(f"Modul: {pkg}", True)
        except ImportError:
            all_ok &= check(f"Modul: {pkg}", False, "requirements.txt o'rnating")

    cfg = ROOT / "config" / "settings.yaml"
    all_ok &= check("config/settings.yaml", cfg.exists())

    zip_path = ROOT / "data" / "data.zip"
    if not zip_path.exists():
        zip_path = ROOT / "data.zip"
    all_ok &= check("data.zip", zip_path.exists(), str(zip_path))

    label = ROOT / "data" / "label.csv"
    images = ROOT / "data" / "raw_images"
    processed = ROOT / "data" / "processed" / "train.csv"
    check("data/label.csv", label.exists(), "unpack_data.py kerak bo'lsa")
    check("data/raw_images/", images.exists(), "unpack_data.py kerak bo'lsa")
    check("data/processed/train.csv", processed.exists(), "prepare_data.py kerak bo'lsa")

    ckpt = ROOT / "results" / "models" / "cnn_best.pth"
    check("CNN checkpoint", ckpt.exists(), "train.py ishga tushiring" if not ckpt.exists() else "")

    print("\n=== Ishlatish tartibi (birinchi marta) ===")
    print("  python -m venv .venv")
    print("  .venv\\Scripts\\activate")
    print("  pip install -r requirements.txt")
    print("  python scripts/unpack_data.py")
    print("  python scripts/prepare_data.py")
    print("  python scripts/train.py")
    print("  python scripts/evaluate.py")
    print("  streamlit run app/app.py")

    print("\nNatija:", "TAYYOR" if all_ok else "BA'ZI QADAMLAR QOLGAN (yuqoridagi XATO larni bajaring)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
