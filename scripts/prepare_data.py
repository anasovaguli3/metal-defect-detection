#!/usr/bin/env python3
"""Clean labels and create train/val/test splits."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_pipeline import prepare_dataset
from src.utils import load_config, set_seed


def main():
    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))
    info = prepare_dataset(cfg)
    print("Ma'lumot tayyorlandi:")
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
