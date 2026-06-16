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

import glob
import os
import shutil
import sys

import cv2

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
    in_dir = utils.resolve_path(config["paths"]["input_dir"])
    out_dir = utils.resolve_path("data/annotation")
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    model = hd.load_detector(config["models"]["hold_detector"])

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG"):
        paths += glob.glob(os.path.join(in_dir, ext))
    paths = sorted(p for p in paths if os.path.basename(p) != "wall.jpg")

    totals = {name: 0 for name in CLASS_NAMES}
    for p in paths:
        image = cv2.imread(p)
        if image is None:
            continue
        h, w = image.shape[:2]
        raw = hd.detect_objects(model, image, det["confidence"], det["iou"], det["max_detections"])
        dets = dfilt.classify_detections(raw, image, config["filter"])

        lines = []
        for d in dets:
            cls_id = KIND_TO_CLASS.get(d.kind)
            if cls_id is None:
                continue
            lines.append(to_yolo_line(cls_id, d.box, w, h))
            totals[CLASS_NAMES[cls_id]] += 1

        stem = os.path.splitext(os.path.basename(p))[0]
        shutil.copy2(p, os.path.join(img_out, os.path.basename(p)))
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
    print(f"bundle: {out_dir}")
    print("Next: upload images/ + labels/ to Roboflow (YOLOv8), then CORRECT —")
    print("  reclassify big volumes the model called 'hold', add 'downclimb' arrows.")


if __name__ == "__main__":
    main()
