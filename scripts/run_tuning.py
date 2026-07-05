#!/usr/bin/env python3
"""Batch size tuning: 8 vs 16."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tuning import run_batch_tuning
from src.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-trained", action="store_true", help="Mavjud checkpointlarni qayta o'qitmaslik")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))
    df = run_batch_tuning(cfg, skip_trained=args.skip_trained)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
