"""Throwaway: sweep per-hold color-EXTRACTION algorithms over cached crops and
score them against hand-read ground truth (my eyes, from the montage).

Ground truth: data/output/_gt_<stem>.json  ->  {"0":"red","1":"grey",...}
Cached crops: data/output/_crops_<stem>.pkl (from scripts/_cache_crops.py)

    python scripts/_color_eval.py 20260618_092148 [--method topsat]
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import utils  # noqa: E402

CFG = utils.load_config()
REFS = utils.reference_labs(CFG["draw_colors"], CFG.get("color_naming", {}).get("hue_anchors"))
CMIN = CFG.get("color_naming", {}).get("chroma_min", 10)


def _region(crop, mask):
    if mask is not None:
        return mask.reshape(-1)
    c = utils.center_inset(crop, 0.6)
    return np.ones(c.shape[:2], dtype=bool).reshape(-1), c  # not used; see below


def lab_current(crop, mask, sat_min=45, val_min=40, val_max=235, min_fraction=0.05):
    """Reproduces src.utils.dominant_color_lab (median of chromatic pixels)."""
    if mask is None:
        crop = utils.center_inset(crop, 0.6)
        reg = np.ones(crop.shape[:2], bool).reshape(-1)
    else:
        reg = mask.reshape(-1)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat, val = hsv[:, 1], hsv[:, 2]
    chrom = reg & (sat >= sat_min) & (val >= val_min) & (val <= val_max)
    if int(chrom.sum()) >= max(1, int(min_fraction * int(reg.sum()))):
        sel = lab[chrom]
    else:
        valid = reg & (val >= val_min) & (val <= val_max)
        sel = lab[valid] if int(valid.sum()) else (lab[reg] if int(reg.sum()) else lab)
    return np.median(sel, axis=0).astype(np.float64)


def lab_topsat(crop, mask, val_min=40, val_max=235, pct=70, sat_floor=30):
    """Sample the hold's COLOR from its most-saturated coherent pixels.

    Drop shadow + blown/chalk pixels (val out of range), then keep the top
    (100-pct)% most-saturated of what's left (but only those above sat_floor).
    Median their Lab. Recovers muted/chalky colors that a plain median washes
    out; if even the saturated end is below sat_floor the hold reads neutral.
    """
    if mask is None:
        crop = utils.center_inset(crop, 0.6)
        reg = np.ones(crop.shape[:2], bool).reshape(-1)
    else:
        reg = mask.reshape(-1)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat, val = hsv[:, 1], hsv[:, 2]
    valid = reg & (val >= val_min) & (val <= val_max)
    if int(valid.sum()) == 0:
        valid = reg if int(reg.sum()) else np.ones_like(reg)
    svals = sat[valid]
    thresh = max(sat_floor, np.percentile(svals, pct))
    colored = valid & (sat >= thresh)
    if int(colored.sum()) == 0:
        colored = valid
    return np.median(lab[colored], axis=0).astype(np.float64)


METHODS = {
    "current": lab_current,
    "topsat": lab_topsat,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--method", default=None, help="run only this method")
    ap.add_argument("--pct", type=int, default=70)
    ap.add_argument("--floor", type=int, default=30)
    ap.add_argument("--show-errors", action="store_true")
    args = ap.parse_args()

    crops = pickle.load(open(f"data/output/_crops_{args.stem}.pkl", "rb"))
    gt_path = f"data/output/_gt_{args.stem}.json"
    gt = json.load(open(gt_path)) if os.path.exists(gt_path) else {}
    labeled = [(i, c) for i, c in enumerate(crops) if str(i) in gt]
    print(f"{len(crops)} crops, {len(labeled)} labeled in GT\n")

    methods = {args.method: METHODS[args.method]} if args.method else METHODS
    for mname, fn in methods.items():
        kw = {}
        if mname == "topsat":
            kw = {"pct": args.pct, "sat_floor": args.floor}
        correct, conf, errors = 0, Counter(), []
        for i, c in labeled:
            lab = fn(c["crop"], c["mask"], **kw)
            pred = utils.nearest_color_name(lab, REFS, CMIN)
            truth = gt[str(i)]
            if pred == truth:
                correct += 1
            else:
                conf[(truth, pred)] += 1
                errors.append((i, truth, pred))
        acc = correct / len(labeled) if labeled else 0
        print(f"== {mname} (pct={kw.get('pct','-')} floor={kw.get('sat_floor','-')}): "
              f"accuracy {correct}/{len(labeled)} = {acc:.2f}")
        for (t, p), n in conf.most_common(12):
            print(f"     {t:7} -> {p:7}  x{n}")
        if args.show_errors:
            print("   wrong idx:", ", ".join(f"{i}({t}->{p})" for i, t, p in errors))
        print()


if __name__ == "__main__":
    main()
