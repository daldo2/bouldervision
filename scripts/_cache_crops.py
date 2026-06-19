"""Throwaway: detect holds once, cache each hold's BGR crop + GrabCut mask, and
build a numbered montage so the TRUE color of each hold can be read off by eye.

This is the ground-truth foundation for tuning color extraction: the slow part
(detection + GrabCut) runs once here; extraction algorithms are then swept over
the cached (crop, mask) pairs in scripts/_color_eval.py.

    python scripts/_cache_crops.py data/input/new_photos/<img>.jpg
"""
import os
import pickle
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import detection_filter as dfilt  # noqa: E402
from src import hold_detector as hd  # noqa: E402
from src import utils  # noqa: E402


def load_upright_bgr(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def main():
    img_path = sys.argv[1]
    stem = os.path.splitext(os.path.basename(img_path))[0]
    cfg = utils.load_config()
    model = hd.load_detector(cfg["models"]["hold_detector"])
    det = cfg["detection"]
    image = utils.white_balance(load_upright_bgr(img_path))  # balance once, like build_holds
    raw = hd.detect_objects(model, image, det["confidence"], det["iou"], det["max_detections"])
    dets = dfilt.classify_detections(raw, image, cfg["filter"])
    boxes = dfilt.holds_only(dets)

    crops = []
    for (x1, y1, x2, y2, conf) in boxes:
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
        crop = image[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            continue
        mask = utils.hold_foreground_mask(crop)
        crops.append({"crop": crop, "mask": mask, "box": (x1, y1, x2, y2)})

    out_pkl = f"data/output/_crops_{stem}.pkl"
    with open(out_pkl, "wb") as fh:
        pickle.dump(crops, fh)

    # numbered montage
    n = len(crops)
    cell, cols = 96, 14
    rows = (n + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 30, np.uint8)
    for i, c in enumerate(crops):
        r, col = divmod(i, cols)
        thumb = cv2.resize(c["crop"], (cell - 6, cell - 18))
        y0, x0 = r * cell + 16, col * cell + 3
        sheet[y0:y0 + thumb.shape[0], x0:x0 + thumb.shape[1]] = thumb
        cv2.putText(sheet, str(i), (col * cell + 3, r * cell + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    out_png = f"data/output/_montage_{stem}.jpg"
    cv2.imwrite(out_png, sheet)
    print(f"{n} holds cached -> {out_pkl}")
    print(f"montage ({rows}x{cols}) -> {out_png}")


if __name__ == "__main__":
    main()
