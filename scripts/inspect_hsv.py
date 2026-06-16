#!/usr/bin/env python
"""
inspect_hsv.py — sample the HSV value of a spot in an image to calibrate colors.

Use this when a hold's color is being misclassified: point it at the hold, read
the HSV values, and widen/adjust the matching range in config/settings.yaml.

Examples:
    # Sample a 20x20 patch centered at pixel (x=300, y=450)
    python scripts/inspect_hsv.py data/input/wall.jpg 300 450

    # Bigger patch
    python scripts/inspect_hsv.py data/input/photo.jpeg 512 700 --size 40

Tip: open the image in any viewer to read off the (x, y) of the hold first.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# Make src/ importable so we can reuse the real classifier.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import utils  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the HSV of an image patch.")
    parser.add_argument("image", help="Path to the image.")
    parser.add_argument("x", type=int, help="Pixel x (column) of the spot.")
    parser.add_argument("y", type=int, help="Pixel y (row) of the spot.")
    parser.add_argument("--size", type=int, default=20, help="Patch size in px (default 20).")
    args = parser.parse_args()

    image = utils.read_image(args.image)
    h, w = image.shape[:2]
    print(f"Image: {w}x{h} px")

    # Clamp a square patch around the requested point to the image bounds.
    half = args.size // 2
    x1, x2 = max(0, args.x - half), min(w, args.x + half)
    y1, y2 = max(0, args.y - half), min(h, args.y + half)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        sys.exit(f"Point ({args.x}, {args.y}) is outside the image.")

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    median = np.median(hsv.reshape(-1, 3), axis=0).astype(int)
    print(f"Patch: {x2 - x1}x{y2 - y1} px at ({args.x}, {args.y})")
    print(f"Median HSV (OpenCV scale H 0-179, S/V 0-255): {tuple(median)}")

    # Show what the current config would call this color.
    config = utils.load_config()
    color, coverage = utils.classify_color(patch, config["colors"])
    print(f"Classified as: {color}  (coverage {coverage:.0%})")


if __name__ == "__main__":
    main()
