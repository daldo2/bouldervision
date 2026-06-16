#!/usr/bin/env python
"""
train_holds.py — fine-tune a YOLOv8 climbing-hold detector (Phase 1).

This is the script form of notebooks/02_train_holds.ipynb, for any machine that
has a GPU (your own box, a cloud VM). On Colab, prefer the notebook.

It does NOT download the dataset for you — point --data at a YOLOv8 `data.yaml`
(e.g. the one Roboflow produces). Color is handled separately by our HSV
classifier, so the model only needs to learn the "hold"/"volume" class(es).

Examples:
    # Train from a Roboflow-exported dataset
    python scripts/train_holds.py --data /path/to/dataset/data.yaml --epochs 100

    # Quick smoke run
    python scripts/train_holds.py --data ./data.yaml --epochs 5 --model yolov8n.pt
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLOv8 hold detector.")
    parser.add_argument("--data", required=True, help="Path to YOLOv8 data.yaml.")
    parser.add_argument("--model", default="yolov8s.pt", help="Base weights to start from.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience.")
    parser.add_argument("--name", default="holds_yolov8s", help="Run name (output folder).")
    args = parser.parse_args()

    # Imported here (not at top) so the rest of the repo doesn't need torch.
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print(
            "WARNING: no GPU detected — training on CPU will be extremely slow.\n"
            "Use Google Colab (notebooks/02_train_holds.ipynb) or a GPU machine."
        )

    model = YOLO(args.model)  # transfer-learn from pretrained weights
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project="bouldervision",
        name=args.name,
    )

    metrics = model.val()
    print(f"\nmAP50:    {float(metrics.box.map50):.3f}")
    print(f"mAP50-95: {float(metrics.box.map):.3f}")
    print(f"\nBest weights: bouldervision/{args.name}/weights/best.pt")
    print("Move it to models/holds.pt and set hold_detector in config/settings.yaml.")


if __name__ == "__main__":
    main()
