"""
utils.py — shared helpers for BoulderVision.

Three groups of helpers live here so the rest of the codebase stays focused:

  1. Config loading          (read settings.yaml once, reuse everywhere)
  2. Color classification     (turn an image crop into a named color)
  3. Drawing                  (put labeled boxes on an image)

Everything is plain functions — no classes — to keep it approachable.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Project root = the folder one level above this file's `src/` directory.
# We compute it so paths work no matter what directory you run scripts from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "settings.yaml")


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------
def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load settings.yaml into a plain dict.

    Why a function instead of just importing a global: it makes the config
    location explicit and testable, and lets callers pass a custom path.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> str:
    """Turn a project-relative path (from config) into an absolute path.

    Config stores paths like "data/samples/output.jpg" relative to the project
    root; this makes them absolute so they resolve regardless of the current
    working directory.
    """
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)


# ---------------------------------------------------------------------------
# 2. Color classification (HSV)
# ---------------------------------------------------------------------------
def classify_color(
    crop_bgr: np.ndarray,
    color_ranges: Dict[str, List[dict]],
) -> Tuple[str, float]:
    """Decide the dominant named color of an image crop.

    HOW IT WORKS:
      - We convert the crop from BGR (OpenCV's default) to HSV. HSV separates
        *hue* (which color) from *brightness*, which makes color matching far
        more robust to lighting than raw RGB/BGR.
      - For each candidate color we build a binary mask of pixels that fall
        inside that color's HSV range(s), then count how many pixels matched.
      - The color with the most matching pixels wins. We also return the
        fraction of the crop it covered, as a rough confidence.

    Args:
        crop_bgr: an (H, W, 3) BGR image region (one detected hold).
        color_ranges: the `colors:` block from settings.yaml.

    Returns:
        (color_name, coverage_fraction). If nothing matches well, returns
        ("unknown", 0.0).
    """
    # Guard against empty crops (can happen with degenerate boxes).
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0

    # Convert once; every color check reuses this HSV version of the crop.
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    best_color = "unknown"
    best_count = 0

    # Check each color. A color can have several ranges (e.g. red wraps the
    # hue circle at 0/179), so we OR their masks together.
    for color_name, ranges in color_ranges.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            lower = np.array(r["lower"], dtype=np.uint8)
            upper = np.array(r["upper"], dtype=np.uint8)
            # inRange marks pixels inside [lower, upper] as 255, else 0.
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = cv2.bitwise_or(mask_total, mask)

        count = int(cv2.countNonZero(mask_total))
        if count > best_count:
            best_count = count
            best_color = color_name

    # Coverage = what fraction of the crop the winning color filled.
    coverage = best_count / total_pixels if total_pixels else 0.0

    # If even the best color barely showed up, we don't trust it.
    if coverage < 0.10:
        return "unknown", coverage

    return best_color, coverage


# ---------------------------------------------------------------------------
# 3. Drawing
# ---------------------------------------------------------------------------
def draw_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    bgr_color: Tuple[int, int, int],
) -> None:
    """Draw one labeled bounding box onto `image` (modifies it in place).

    `box` is (x1, y1, x2, y2) in pixel coordinates.
    """
    x1, y1, x2, y2 = box

    # The rectangle around the hold.
    cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color, thickness=2)

    # Draw a filled label background so text is readable over any wall color.
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        image,
        (x1, y1 - text_h - baseline - 4),
        (x1 + text_w + 4, y1),
        bgr_color,
        thickness=-1,  # -1 = filled
    )
    # The label text, drawn in white on top of the colored background.
    cv2.putText(
        image,
        label,
        (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def draw_color_for(color_name: str, draw_colors: Dict[str, list]) -> Tuple[int, int, int]:
    """Look up the BGR drawing color for a color name, falling back to gray."""
    bgr = draw_colors.get(color_name, draw_colors.get("unknown", [200, 200, 200]))
    return tuple(int(c) for c in bgr)


def read_image(path: str) -> np.ndarray:
    """Load an image from disk, raising a clear error if it's missing/unreadable."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image at '{path}'. Check the path and file format."
        )
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """Write an image to disk, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, image)
