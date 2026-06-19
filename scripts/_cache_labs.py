"""Throwaway tuning helper: detect holds once and cache each hold's Lab vector,
so color-naming palettes can be swept in-memory without re-running the slow
detection + GrabCut mask per experiment.

    python scripts/_cache_labs.py data/input/new_photos/<img>.jpg [...]

Writes data/output/_labs_<stem>.npy (an (N,3) float64 array of Lab).
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import detection_filter as dfilt  # noqa: E402
from src import hold_detector as hd  # noqa: E402
from src import route_extractor as rex  # noqa: E402
from src import utils  # noqa: E402


def load_upright_bgr(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def main():
    cfg = utils.load_config()
    model = hd.load_detector(cfg["models"]["hold_detector"])
    det = cfg["detection"]
    use_mask = cfg.get("color_naming", {}).get("use_mask", False)
    for img_path in sys.argv[1:]:
        image = load_upright_bgr(img_path)
        raw = hd.detect_objects(model, image, det["confidence"], det["iou"], det["max_detections"])
        dets = dfilt.classify_detections(raw, image, cfg["filter"])
        holds = rex.build_holds(image, dfilt.holds_only(dets), use_mask=use_mask)
        labs = np.array([h.lab for h in holds], dtype=np.float64)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out = f"data/output/_labs_{stem}.npy"
        np.save(out, labs)
        print(f"{img_path}: {len(labs)} holds -> {out}")


if __name__ == "__main__":
    main()
