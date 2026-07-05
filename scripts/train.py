#!/usr/bin/env python3
"""Train custom CNN."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.trainer import train_cnn
from src.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Checkpointdan davom etish")
    args = parser.parse_args()

    cfg = load_config()
    if args.resume or cfg["training"].get("resume"):
        resume = True
    else:
        resume = args.resume
    set_seed(cfg.get("random_seed", 42))
    result = train_cnn(cfg, batch_size=args.batch_size, epochs=args.epochs, resume=resume)
    print("O'qitish tugadi:", result)


if __name__ == "__main__":
    main()
