"""eval_pose.py — measure pose accuracy against hand-corrected ground truth.

Compares the pose model's predictions (data/pose_annotation/predictions_coco.json,
written by export_pose_frames.py) against your corrected keypoints exported from
CVAT (COCO Keypoints). This is the baseline number we need BEFORE deciding whether
a fine-tune is worth it, and it shows WHICH joints fail — wrists/ankles are the
ones that drive contact detection.

Eval set: only frames you actually CORRECTED. Untouched frames still carry the raw
model output, so GT == prediction there and would falsely score 100%. We detect
edited frames by diffing GT vs prediction and evaluate only those — i.e. the hard
cases. On an edited frame every joint is trusted (you reviewed the whole skeleton),
so joints you left in place count as confirmed-correct (error 0).

Metrics, per joint and overall:
  - median / mean Euclidean error in pixels;
  - PCK@alpha: fraction of joints within alpha * torso-size of GT (scale-free, so
    a big climber and a small one are judged fairly). Torso size = distance from
    mid-shoulder to mid-hip; falls back to GT bbox diagonal.

Run:
  python scripts/eval_pose.py
  python scripts/eval_pose.py --gt data/pose_annotation/gt/annotations/person_keypoints_default.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_DIR = os.path.join(PROJECT_ROOT, "data", "pose_annotation")

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
CONTACT = {"left_wrist", "right_wrist", "left_ankle", "right_ankle"}
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


def load_keypoints(path):
    """{file_name: (17,3) array} from a COCO-keypoints file."""
    d = json.load(open(path))
    id2name = {im["id"]: im["file_name"] for im in d["images"]}
    out = {}
    for a in d["annotations"]:
        out[id2name[a["image_id"]]] = np.array(a["keypoints"], dtype=float).reshape(-1, 3)
    return out


def torso_size(kp):
    """Scale for normalization: mid-shoulder to mid-hip distance, else bbox diag."""
    pts = kp[[L_SH, R_SH, L_HIP, R_HIP]]
    if (pts[:, 2] > 0).all():
        mid_sh = pts[:2, :2].mean(0)
        mid_hip = pts[2:, :2].mean(0)
        d = np.hypot(*(mid_sh - mid_hip))
        if d > 1:
            return d
    seen = kp[kp[:, 2] > 0]
    if len(seen) < 2:
        return None
    w = seen[:, 0].max() - seen[:, 0].min()
    h = seen[:, 1].max() - seen[:, 1].min()
    return float(np.hypot(w, h)) or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default=os.path.join(ANN_DIR, "predictions_coco.json"))
    ap.add_argument("--gt", default=None, help="corrected COCO export (auto-found under gt/ if omitted)")
    ap.add_argument("--alpha", type=float, default=0.2, help="PCK threshold as fraction of torso (default 0.2)")
    args = ap.parse_args()

    gt_path = args.gt
    if gt_path is None:
        hits = glob.glob(os.path.join(ANN_DIR, "gt", "**", "*.json"), recursive=True)
        if not hits:
            print("No ground-truth json found under data/pose_annotation/gt/. "
                  "Unzip your CVAT export there, or pass --gt.")
            return 1
        gt_path = hits[0]

    pred = load_keypoints(args.pred)
    gt = load_keypoints(gt_path)

    # Edited frames = GT keypoints differ from the model's prediction.
    eval_frames = []
    for name, gk in gt.items():
        pk = pred.get(name)
        if pk is None:
            continue
        if np.abs(gk[:, :2] - pk[:, :2]).sum() > 1.0:
            eval_frames.append(name)

    if not eval_frames:
        print("No corrected frames detected (GT == predictions everywhere).")
        return 1

    # Per-joint error accumulators (over evaluated joints with GT visibility > 0).
    err = {n: [] for n in KEYPOINT_NAMES}     # pixel error
    norm_err = {n: [] for n in KEYPOINT_NAMES}  # error / torso size
    skipped_scale = 0
    for name in eval_frames:
        gk, pk = gt[name], pred[name]
        scale = torso_size(gk)
        for i, jn in enumerate(KEYPOINT_NAMES):
            if gk[i, 2] <= 0:
                continue  # joint not labeled / marked invisible in GT
            d = float(np.hypot(*(gk[i, :2] - pk[i, :2])))
            err[jn].append(d)
            if scale:
                norm_err[jn].append(d / scale)
        if scale is None:
            skipped_scale += 1

    def pck(vals):
        return 100.0 * np.mean([v <= args.alpha for v in vals]) if vals else float("nan")

    clips = sorted({n.split("__")[0][:12] for n in eval_frames})
    print(f"Ground truth: {gt_path}")
    print(f"Evaluated {len(eval_frames)} corrected frames across {len(clips)} clip(s).\n")

    header = f"{'joint':16s} {'n':>4s} {'median_px':>10s} {'mean_px':>9s} {'PCK@'+str(args.alpha):>9s}"
    print(header)
    print("-" * len(header))

    def row(jn):
        e, ne = err[jn], norm_err[jn]
        if not e:
            return f"{jn:16s} {'0':>4s} {'-':>10s} {'-':>9s} {'-':>9s}"
        return (f"{jn:16s} {len(e):>4d} {np.median(e):>10.1f} {np.mean(e):>9.1f} "
                f"{pck(ne):>8.1f}%")

    all_e, all_ne = [], []
    for jn in KEYPOINT_NAMES:
        marker = " <- contact" if jn in CONTACT else ""
        print(row(jn) + marker)
        all_e += err[jn]
        all_ne += norm_err[jn]

    print("-" * len(header))
    print(f"{'ALL':16s} {len(all_e):>4d} {np.median(all_e):>10.1f} {np.mean(all_e):>9.1f} {pck(all_ne):>8.1f}%")

    contact_ne = [v for jn in CONTACT for v in norm_err[jn]]
    contact_e = [v for jn in CONTACT for v in err[jn]]
    print(f"{'CONTACT (h/f)':16s} {len(contact_e):>4d} {np.median(contact_e):>10.1f} "
          f"{np.mean(contact_e):>9.1f} {pck(contact_ne):>8.1f}%")
    if skipped_scale:
        print(f"\n({skipped_scale} frame(s) had no torso scale; counted in px, not PCK.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
