"""Throwaway: sweep neutral-anchor palettes over cached hold Labs (fast).

Uses the PRODUCTION naming path (utils.reference_labs + nearest_color_name); the
only thing it varies is the draw_colors palette (e.g. injecting a grey anchor).

    python scripts/_sweep_neutral.py 20260618_092148 20260618_100456 --grey 128
    python scripts/_sweep_neutral.py 20260618_092148 --grey 100 170   # two greys
"""
import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+", help="image stems (loads data/output/_labs_<stem>.npy)")
    ap.add_argument("--grey", type=int, nargs="*", default=[],
                    help="one or more grey BGR levels to add as neutral anchors (e.g. 128)")
    args = ap.parse_args()

    cfg = utils.load_config()
    draw = dict(cfg["draw_colors"])
    for i, lvl in enumerate(args.grey):
        draw[f"grey{i}" if len(args.grey) > 1 else "grey"] = [lvl, lvl, lvl]
    refs = utils.reference_labs(draw)
    cmin = cfg.get("color_naming", {}).get("chroma_min", 12.0)

    print(f"grey anchors: {args.grey or 'none'}   chroma_min={cmin}")
    grand = Counter()
    for stem in args.stems:
        labs = np.load(f"data/output/_labs_{stem}.npy")
        counts = Counter()
        black_Ls = []
        for lab in labs:
            name = utils.nearest_color_name(lab, refs, cmin)
            # collapse grey0/grey1 -> grey for reporting
            rep = "grey" if name.startswith("grey") else name
            counts[rep] += 1
            grand[rep] += 1
            if rep == "black":
                black_Ls.append(float(lab[0]))
        print(f"\n{stem}: {len(labs)} holds")
        print("  ", dict(counts.most_common()))
        if black_Ls:
            bl = np.array(black_Ls)
            print(f"   black L: n={len(bl)} min={bl.min():.0f} med={np.median(bl):.0f} max={bl.max():.0f}")
    if len(args.stems) > 1:
        print("\nGRAND:", dict(grand.most_common()))


if __name__ == "__main__":
    main()
