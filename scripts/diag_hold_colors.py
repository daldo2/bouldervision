"""Diagnostic: draw every detected hold with a THICK outline in its detected
color + the color NAME printed at its center. No route grouping — just the raw
per-hold color decision, so we can see where color naming goes wrong.

    python scripts/diag_hold_colors.py data/input/new_photos/<img>.jpg
"""
import os
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import detection_filter as dfilt  # noqa: E402
from src import hold_detector as hd  # noqa: E402
from src import route_extractor as rex  # noqa: E402
from src import utils  # noqa: E402


def load_upright_bgr(path):
    """Load with EXIF orientation applied (phone portrait shots) as BGR uint8."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/input/new_photos/20260618_092148.jpg"
    cfg = utils.load_config()
    image = load_upright_bgr(img_path)
    model = hd.load_detector(cfg["models"]["hold_detector"])
    det = cfg["detection"]
    raw = hd.detect_objects(model, image, det["confidence"], det["iou"], det["max_detections"])
    dets = dfilt.classify_detections(raw, image, cfg["filter"])
    use_mask = cfg.get("color_naming", {}).get("use_mask", False)
    holds = rex.build_holds(image, dfilt.holds_only(dets), use_mask=use_mask)

    refs = utils.reference_labs(cfg["draw_colors"],
                                cfg.get("color_naming", {}).get("hue_anchors"))
    cmin = cfg.get("color_naming", {}).get("chroma_min", 12.0)
    draw_colors = cfg["draw_colors"]

    out = image.copy()
    counts = Counter()
    for h in holds:
        name = utils.nearest_color_name(h.lab, refs, cmin)
        counts[name] += 1
        col = utils.draw_color_for(name, draw_colors)
        x1, y1, x2, y2 = h.box
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
        cx, cy = (x1 + x2) // 2 - 22, (y1 + y2) // 2
        cv2.putText(out, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs("data/output", exist_ok=True)
    outp = "data/output/hold_colors_named.jpg"
    cv2.imwrite(outp, out)
    print(f"{len(holds)} holds by color:", dict(counts.most_common()))
    print("wrote", outp)


if __name__ == "__main__":
    main()
