"""Throwaway prototype: draw a per-hold CONTOUR inside each detected box instead
of the axis-aligned rectangle, so the boxes of adjacent holds stop visually
overlapping. Classical CV only (GrabCut per box) — NO model/annotation change.

Goal is to eyeball whether contours come out clean enough to be worth adopting
as the draw style (and later for cleaner color sampling / contact tests).

    python scripts/prototype_contours.py data/input/new_photos/<img>.jpg
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import hold_detector, utils  # noqa: E402


def load_upright_bgr(path):
    """Load with EXIF orientation applied (phone portrait shots) as BGR uint8."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def box_contour(img, box, inset=0.06, iters=4):
    """GrabCut within a single box -> its largest external contour (image coords).

    Returns an (N,2) int array of polygon points, or None if segmentation was
    degenerate (we then fall back to the box).
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    mask = np.zeros((ch, cw), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    mx, my = int(cw * inset) + 1, int(ch * inset) + 1
    rect = (mx, my, max(1, cw - 2 * mx), max(1, ch - 2 * my))
    try:
        cv2.grabCut(crop, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.08 * cw * ch:  # GrabCut basically failed
        return None
    eps = 0.01 * cv2.arcLength(c, True)
    c = cv2.approxPolyDP(c, eps, True)
    return c.reshape(-1, 2) + np.array([x1, y1])


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/input/new_photos/20260618_092148.jpg"
    config = utils.load_config()
    model = hold_detector.load_detector(config["models"]["hold_detector"])
    image = load_upright_bgr(img_path)
    boxes = hold_detector.detect_holds(
        model, image,
        confidence=config["detection"]["confidence"],
        iou=config["detection"]["iou"],
        max_detections=config["detection"]["max_detections"],
    )
    print(f"{len(boxes)} detections on {os.path.basename(img_path)}")

    boxed = image.copy()
    contoured = image.copy()
    n_contour = 0
    for (x1, y1, x2, y2, _conf) in boxes:
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 200, 0), 2, cv2.LINE_AA)
        poly = box_contour(image, (x1, y1, x2, y2))
        if poly is not None and len(poly) >= 3:
            cv2.polylines(contoured, [poly.astype(np.int32)], True, (0, 200, 0), 2, cv2.LINE_AA)
            n_contour += 1
        else:  # fell back to the box
            cv2.rectangle(contoured, (x1, y1), (x2, y2), (0, 0, 220), 1, cv2.LINE_AA)
    print(f"clean contour for {n_contour}/{len(boxes)} (rest drawn as red box fallback)")

    out_dir = utils.resolve_path(config["paths"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    sbs = np.hstack([boxed, contoured])
    out = os.path.join(out_dir, "prototype_contours.jpg")
    cv2.imwrite(out, sbs)
    print(f"wrote {out}  (LEFT: boxes  |  RIGHT: contours)")

    # Zoom into a busy region so the box-vs-contour difference is legible.
    h, w = image.shape[:2]
    rx1, ry1, rx2, ry2 = int(0.20 * w), int(0.25 * h), int(0.60 * w), int(0.70 * h)
    bz = boxed[ry1:ry2, rx1:rx2]
    cz = contoured[ry1:ry2, rx1:rx2]
    zoom = np.vstack([bz, cz])
    zoom = cv2.resize(zoom, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_LINEAR)
    zout = os.path.join(out_dir, "prototype_contours_zoom.jpg")
    cv2.imwrite(zout, zoom)
    print(f"wrote {zout}  (TOP: boxes  |  BOTTOM: contours)")


if __name__ == "__main__":
    main()
