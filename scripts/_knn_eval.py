"""Throwaway: cross-photo k-NN color classifier vs the anchor/threshold baseline.

Trains a k-NN on per-hold color features from one or more TRAIN photos (their
hand-read GT) and scores it on a held-out TEST photo — the honest test of whether
a learned naming generalizes across lighting, not just fits one wall.

Feature = the production dominant-Lab, computed from the cached crop + cached mask
(no GrabCut re-run). L is down-weighted so chroma/hue drive the match.

    python scripts/_knn_eval.py --train 20260618_092148 --test 20260618_100456
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


def feat(crop, mask, sat_min=45, val_min=40, val_max=235, min_fraction=0.05):
    """Production dominant-Lab using the cached mask (mirrors dominant_color_lab)."""
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


def load(stem):
    crops = pickle.load(open(f"data/output/_crops_{stem}.pkl", "rb"))
    gt = json.load(open(f"data/output/_gt_{stem}.json"))
    X, y = [], []
    for i, c in enumerate(crops):
        if str(i) in gt:
            X.append(feat(c["crop"], c["mask"]))
            y.append(gt[str(i)])
    return np.array(X, float), np.array(y)


def knn_predict(Xtr, ytr, Xte, k=3, wl=0.5):
    F, G = Xtr.copy(), Xte.copy()
    F[:, 0] *= wl
    G[:, 0] *= wl
    out = []
    for g in G:
        d = np.linalg.norm(F - g, axis=1)
        out.append(Counter(ytr[np.argsort(d)[:k]]).most_common(1)[0][0])
    return np.array(out)


def report(name, pred, truth):
    ok = int((pred == truth).sum())
    print(f"\n== {name}: {ok}/{len(truth)} = {ok/len(truth):.2f}")
    conf = Counter((t, p) for t, p in zip(truth, pred) if t != p)
    for (t, p), c in conf.most_common(10):
        print(f"   {t:7}->{p:7} x{c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--wl", type=float, default=0.5)
    args = ap.parse_args()

    Xtr = np.empty((0, 3)); ytr = np.array([], dtype=object)
    for s in args.train:
        X, y = load(s)
        Xtr = np.vstack([Xtr, X]); ytr = np.concatenate([ytr, y])
    Xte, yte = load(args.test)
    print(f"train {len(ytr)} holds ({'+'.join(args.train)})  ->  test {len(yte)} ({args.test})")

    # anchor/threshold baseline on the test photo
    base = np.array([utils.nearest_color_name(x, REFS, CMIN) for x in Xte])
    report("anchors/threshold baseline", base, yte)
    pred = knn_predict(Xtr, ytr, Xte, args.k, args.wl)
    report(f"kNN (k={args.k}, L-weight={args.wl})", pred, yte)


if __name__ == "__main__":
    main()
