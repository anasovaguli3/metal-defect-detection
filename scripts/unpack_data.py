#!/usr/bin/env python3
"""Extract data.zip into data/raw_images and data/label.csv."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_pipeline import unpack_zip
from src.utils import load_config, set_seed


def main():
    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))
    result = unpack_zip(cfg)
    print("Unpack tugadi:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
