#!/usr/bin/env python3
"""Train EfficientNet-B0 baseline."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baseline import train_baseline
from src.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))
    result = train_baseline(cfg, resume=args.resume)
    print("Baseline tugadi:", result)


if __name__ == "__main__":
    main()
