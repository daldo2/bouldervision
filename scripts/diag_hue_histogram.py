"""Diagnostic: measure the real hue distribution of detected holds.

For a set of photos, run detection -> furniture filter -> per-hold color, then
dump each chromatic hold's hue angle (deg on the Lab a*/b* wheel), chroma and
lightness. Prints a hue histogram and, per current color *name*, the median hue
/ chroma / count. This is the measurement that calibrates `hue_anchors`: it
shows where real hold clusters actually sit, and whether warm holds (red /
orange / pink) are being merged.

    python scripts/diag_hue_histogram.py data/input/new_photos/*.jpg
    python scripts/diag_hue_histogram.py            # defaults to new_photos/*.jpg
"""
import glob
import math
import os
import sys
from collections import Counter, defaultdict

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
    paths = sys.argv[1:] or sorted(glob.glob("data/input/new_photos/*.jpg"))
    cfg = utils.load_config()
    cmin = cfg.get("color_naming", {}).get("chroma_min", 12.0)
    use_mask = cfg.get("color_naming", {}).get("use_mask", False)
    refs = utils.reference_labs(cfg["draw_colors"],
                                cfg.get("color_naming", {}).get("hue_anchors"))
    det = cfg["detection"]
    model = hd.load_detector(cfg["models"]["hold_detector"])

    rows = []  # (hue_deg, chroma, L, name)
    for i, p in enumerate(paths):
        image = load_upright_bgr(p)
        raw = hd.detect_objects(model, image, det["confidence"], det["iou"], det["max_detections"])
        dets = dfilt.classify_detections(raw, image, cfg["filter"])
        holds = rex.build_holds(image, dfilt.holds_only(dets), use_mask=use_mask)
        for h in holds:
            a, b = float(h.lab[1]) - 128.0, float(h.lab[2]) - 128.0
            chroma = math.hypot(a, b)
            hue = math.degrees(math.atan2(b, a))
            name = utils.nearest_color_name(h.lab, refs, cmin)
            rows.append((hue, chroma, float(h.lab[0]), name))
        print(f"[{i+1}/{len(paths)}] {os.path.basename(p)}: {len(holds)} holds", flush=True)

    chromatic = [r for r in rows if r[1] >= cmin]
    neutral = [r for r in rows if r[1] < cmin]
    print(f"\n{len(rows)} holds total | {len(chromatic)} chromatic (chroma>={cmin}) | {len(neutral)} neutral\n")

    # Hue histogram in 20-deg bins over [-180,180).
    print("HUE HISTOGRAM (chromatic holds), 20deg bins:")
    bins = Counter()
    for hue, *_ in chromatic:
        bins[int(math.floor(hue / 20.0)) * 20] += 1
    for lo in range(-180, 180, 20):
        n = bins.get(lo, 0)
        bar = "#" * n
        print(f"  [{lo:4d},{lo+20:4d})  {n:4d}  {bar}")

    # Per current name: median hue / chroma / L, count.
    print("\nPER-NAME stats (current naming):")
    by_name = defaultdict(list)
    for hue, chroma, L, name in rows:
        by_name[name].append((hue, chroma, L))
    for name, vals in sorted(by_name.items(), key=lambda kv: -len(kv[1])):
        hues = np.array([v[0] for v in vals])
        chr_ = np.array([v[1] for v in vals])
        Ls = np.array([v[2] for v in vals])
        # circular-ish median: only meaningful for non-wrapping clusters, fine for a diag
        print(f"  {name:8s} n={len(vals):4d}  hue med={np.median(hues):7.1f}  "
              f"iqr=[{np.percentile(hues,25):6.1f},{np.percentile(hues,75):6.1f}]  "
              f"chroma med={np.median(chr_):5.1f}  L med={np.median(Ls):5.1f}")

    print(f"\ncurrent hue_anchors: {cfg.get('color_naming', {}).get('hue_anchors')}")


if __name__ == "__main__":
    main()
