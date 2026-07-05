#!/usr/bin/env python3
"""Evaluate CNN on test set."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_model
from src.explain import generate_feature_maps
from src.utils import load_config, set_seed


def main():
    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))
    result = evaluate_model(cfg)
    print("Test natijalari:", result["metrics_path"])
    print(f"  accuracy: {result['metrics']['accuracy']:.4f}")
    maps = generate_feature_maps(cfg)
    print("Feature maps:", maps)


if __name__ == "__main__":
    main()
