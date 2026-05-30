#!/usr/bin/env python3
"""Download preset YOLO weights (first run can take several minutes)."""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.model_loader import HF_SOURCES, resolve_yolo_weights
from app.models_registry import PRESETS, PRESET_DESCRIPTIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Download plant YOLO weights")
    parser.add_argument(
        "--preset",
        default="plant_disease",
        choices=list(PRESETS.keys()),
    )
    parser.add_argument(
        "--copy-to",
        default="weights/plant_disease.pt",
        help="Copy HF weights to this path for offline / plant_local preset",
    )
    args = parser.parse_args()

    preset_cfg = PRESETS[args.preset]
    dest = ROOT / preset_cfg["yolo_model"]
    print(PRESET_DESCRIPTIONS.get(args.preset, args.preset))

    weights_path = resolve_yolo_weights(str(dest), args.preset)
    print(f"Weights at: {weights_path}")

    from ultralytics import YOLO

    model = YOLO(weights_path)
    print(f"OK — {len(model.names)} classes, sample: {list(model.names.values())[:5]}")

    import numpy as np

    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    print("Warmup done.")


if __name__ == "__main__":
    main()
