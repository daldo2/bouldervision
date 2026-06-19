"""Bootstrap annotation labels for a richer-class retrain.

The current detector (best.pt) only knows hold(0)/volume(1), and it mislabels
big volumes as holds. To fix that we retrain with more classes — but labeling
from scratch is slow. This script pre-labels every photo with our best current
guess (detection + the hold/volume/marker/tape filter), in YOLO format, so you
only have to CORRECT in Roboflow instead of drawing everything:

  - fix volumes the model called holds (the main win),
  - add the `downclimb` class by hand (the down-arrow holds — not auto-detectable yet),
  - fix any stray marker/tape.

Output: data/annotation/{images,labels}/ + data.yaml, ready to drag into Roboflow
(Upload -> YOLO v8). Re-export anytime; it overwrites.

Run:  python scripts/export_for_annotation.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import detection_filter as dfilt  # noqa: E402
import hold_detector as hd  # noqa: E402
import utils  # noqa: E402

# Target class set for the retrain. `downclimb` has no auto-detector yet — its id
# is reserved so you can add those boxes in Roboflow without renumbering later.
CLASS_NAMES = ["hold", "volume", "downclimb", "marker", "tape"]
KIND_TO_CLASS = {
    dfilt.HOLD: 0,
    dfilt.VOLUME: 1,
    dfilt.MARKER: 3,
    dfilt.TAPE: 4,
}

# High-recall settings for the bootstrap (NOT the production config). The point
# is to miss as few holds as possible — extra/false boxes are quick to DELETE in
# Roboflow, but holds the model never proposes you'd have to draw from scratch.
EXPORT_CONFIDENCE = 0.20   # below production 0.25 → catches borderline holds
EXPORT_MAX_DET = 600       # headroom once conf is lowered + TTA adds boxes
EXPORT_AUGMENT = True      # test-time augmentation (multi-scale + flips): more recall
MAX_FRAMES_PER_BURST = 10  # thin a phone burst / panorama sweep (e.g. *_001.._053)


def thin_bursts(paths, max_per_group):
    """Drop near-duplicate frames from oversized burst/panorama groups.

    Files that share a base name before a trailing `_NNN` (e.g.
    `20260618_103756_001..._053`) are one rapid sweep with heavy overlap; keep
    only `max_per_group` evenly-spaced frames so the wall is covered without
    annotating the same hold many times. Standalone photos are always kept.
    Returns (kept, dropped).
    """
    groups: dict[str, list] = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        base = re.sub(r"_\d{3,4}$", "", stem)  # strip only a 3-4 digit burst suffix
        groups.setdefault(base, []).append(p)
    kept, dropped = [], []
    for members in groups.values():
        members.sort()
        if len(members) <= max_per_group:
            kept += members
            continue
        step = len(members) / max_per_group
        keep_idx = {int(i * step) for i in range(max_per_group)}
        for i, m in enumerate(members):
            (kept if i in keep_idx else dropped).append(m)
    return sorted(kept), sorted(dropped)


def load_upright_bgr(path: str):
    """Load an image with EXIF orientation APPLIED, as BGR uint8 (or None).

    Phones tag portrait shots "rotate 90" in EXIF; cv2.imread ignores that and
    would hand the model a sideways wall (worse recall) and labels that don't
    line up with how Roboflow renders the photo. We bake the rotation into the
    pixels here and re-save, so detection, labels, and display all agree.
    """
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:
        return None
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _area(box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection(a, b) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    return ix * iy


def merge_nested(dets, contain_thresh: float = 0.75):
    """Drop a smaller same-kind box that sits mostly INSIDE a larger one.

    Catches the "two boxes on one hold" duplicates that plain NMS misses: when a
    small box is nested in a big one their IoU is low (large union), so NMS keeps
    both. We instead measure containment = intersection / smaller-box-area, and
    if a box is >=`contain_thresh` covered by a bigger same-kind box, the bigger
    one (which better bounds the hold) wins. Different kinds never merge, so a
    hold sitting in front of a volume is left alone. Returns (kept, n_removed).
    """
    order = sorted(range(len(dets)), key=lambda i: _area(dets[i].box), reverse=True)
    removed = set()
    for pos, i in enumerate(order):
        if i in removed:
            continue
        bi = dets[i]
        for j in order[pos + 1:]:  # only smaller-or-equal boxes
            if j in removed or dets[j].kind != bi.kind:
                continue
            aj = _area(dets[j].box)
            if aj > 0 and _intersection(bi.box, dets[j].box) / aj >= contain_thresh:
                removed.add(j)
    return [d for k, d in enumerate(dets) if k not in removed], len(removed)


def to_yolo_line(cls_id: int, box, w: int, h: int) -> str:
    """One YOLO label line: class xc yc bw bh, all normalized to [0, 1]."""
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    config = utils.load_config()
    det = config["detection"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir", nargs="?", default=config["paths"]["input_dir"],
                    help="folder of photos to pre-label (default: config input_dir)")
    args = ap.parse_args()

    in_dir = utils.resolve_path(args.input_dir)
    out_dir = utils.resolve_path("data/annotation")
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    # Wipe any previous bundle so a re-run can't leave stale images/labels behind.
    for d in (img_out, lbl_out):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    model = hd.load_detector(config["models"]["hold_detector"])

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"):
        paths += glob.glob(os.path.join(in_dir, ext))
    paths = sorted(p for p in paths if os.path.basename(p) != "wall.jpg")
    paths, dropped = thin_bursts(paths, MAX_FRAMES_PER_BURST)
    print(f"{len(paths)} photos to label (dropped {len(dropped)} near-duplicate "
          f"burst frames) from {in_dir}")
    print(f"recall mode: conf={EXPORT_CONFIDENCE} augment={EXPORT_AUGMENT} "
          f"max_det={EXPORT_MAX_DET}")

    totals = {name: 0 for name in CLASS_NAMES}
    merged_total = 0
    for p in paths:
        image = load_upright_bgr(p)
        if image is None:
            continue
        h, w = image.shape[:2]
        raw = hd.detect_objects(model, image, EXPORT_CONFIDENCE, det["iou"],
                                EXPORT_MAX_DET, augment=EXPORT_AUGMENT)
        dets = dfilt.classify_detections(raw, image, config["filter"])
        dets, n_merged = merge_nested(dets)  # collapse "two boxes on one hold"
        merged_total += n_merged

        lines = []
        for d in dets:
            cls_id = KIND_TO_CLASS.get(d.kind)
            if cls_id is None:
                continue
            lines.append(to_yolo_line(cls_id, d.box, w, h))
            totals[CLASS_NAMES[cls_id]] += 1

        stem = os.path.splitext(os.path.basename(p))[0]
        # Save the UPRIGHT image (orientation baked in) so its pixels match the
        # labels and Roboflow's display — never copy the raw EXIF-rotated file.
        cv2.imwrite(os.path.join(img_out, stem + ".jpg"), image)
        with open(os.path.join(lbl_out, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"  {os.path.basename(p):45} {len(lines):4d} boxes")

    with open(os.path.join(out_dir, "data.yaml"), "w") as fh:
        fh.write("# Bootstrap labels for richer-class hold detector retrain.\n")
        fh.write("train: images\nval: images\n")
        fh.write(f"nc: {len(CLASS_NAMES)}\n")
        fh.write(f"names: {CLASS_NAMES}\n")

    print("-" * 50)
    print("pre-labeled by class:", ", ".join(f"{k}={v}" for k, v in totals.items()))
    print(f"merged {merged_total} nested duplicate boxes")
    print(f"bundle: {out_dir}")
    print("Next: upload images/ + labels/ to Roboflow (YOLOv8), then CORRECT —")
    print("  reclassify big volumes the model called 'hold', add 'downclimb' arrows.")


if __name__ == "__main__":
    main()
