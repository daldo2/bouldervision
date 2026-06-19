"""Color-naming evaluation harness.

Measures how well per-hold color naming recovers the colors that are actually
on each wall, against hand-labeled ground truth in eval/labels.yaml. This turns
"does it look right?" into numbers, so palette/threshold changes can be judged
objectively instead of by eye.

For each labeled image we run detection -> furniture filter -> per-hold color
naming, keep the colors that label at least `min_holds` holds, and compare that
set to `colors_present`:

  recall    = |found & present| / |present|   (are real colors detected? e.g. green)
  precision = |found & present| / |found|      (do we avoid inventing colors? e.g. purple)

Not offline (loads the detector). Needs models/best.pt and the local photos.

Run:  python scripts/eval_colors.py
      python scripts/eval_colors.py --chroma-min 14   # try an override
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import cv2
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import detection_filter as dfilt  # noqa: E402
import hold_detector as hd  # noqa: E402
import route_extractor as rex  # noqa: E402
import utils  # noqa: E402


def found_colors(image, model, config, chroma_min):
    """Set of colors -> hold counts the pipeline names for one image."""
    det = config["detection"]
    raw = hd.detect_objects(model, image, det["confidence"], det["iou"], det["max_detections"])
    dets = dfilt.classify_detections(raw, image, config["filter"])
    holds = rex.build_holds(image, dfilt.holds_only(dets))
    refs = utils.reference_labs(config["draw_colors"],
                                config.get("color_naming", {}).get("hue_anchors"))
    return Counter(utils.nearest_color_name(h.lab, refs, chroma_min) for h in holds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate color naming vs eval/labels.yaml.")
    parser.add_argument("--labels", default=os.path.join(PROJECT_ROOT, "eval", "labels.yaml"))
    parser.add_argument("--chroma-min", type=float, default=None,
                        help="Override color_naming.chroma_min for a sweep.")
    args = parser.parse_args()

    with open(args.labels) as fh:
        spec = yaml.safe_load(fh)
    config = utils.load_config()
    chroma_min = args.chroma_min if args.chroma_min is not None else \
        config.get("color_naming", {}).get("chroma_min", 16.0)
    min_holds = int(spec.get("min_holds", 2))
    input_dir = utils.resolve_path(config["paths"]["input_dir"])
    model = hd.load_detector(config["models"]["hold_detector"])

    print(f"chroma_min={chroma_min}  min_holds={min_holds}\n")
    header = f"{'image':10} {'recall':>6} {'prec':>6}  {'missed':22} {'spurious'}"
    print(header)
    print("-" * len(header))

    tot_recall = tot_prec = 0.0
    n = 0
    missed_tally: Counter = Counter()
    spurious_tally: Counter = Counter()

    for item in spec["images"]:
        path = os.path.join(input_dir, item["file"])
        image = cv2.imread(path)
        if image is None:
            print(f"{item['file'][:10]:10}  (missing on disk — skipped)")
            continue
        present = set(item["colors_present"])
        counts = found_colors(image, model, config, chroma_min)
        found = {c for c, k in counts.items() if k >= min_holds and c not in ("unknown",)}

        hit = present & found
        recall = len(hit) / len(present) if present else 1.0
        precision = len(hit) / len(found) if found else 1.0
        missed = present - found
        spurious = found - present
        missed_tally.update(missed)
        spurious_tally.update(spurious)
        tot_recall += recall
        tot_prec += precision
        n += 1
        print(f"{item['file'][:10]:10} {recall:6.2f} {precision:6.2f}  "
              f"{' '.join(sorted(missed)) or '-':22} {' '.join(sorted(spurious)) or '-'}")

    if n:
        print("-" * len(header))
        print(f"{'MACRO':10} {tot_recall/n:6.2f} {tot_prec/n:6.2f}")
        if missed_tally:
            print("most missed  :", ", ".join(f"{c}×{k}" for c, k in missed_tally.most_common()))
        if spurious_tally:
            print("most spurious:", ", ".join(f"{c}×{k}" for c, k in spurious_tally.most_common()))


if __name__ == "__main__":
    main()
