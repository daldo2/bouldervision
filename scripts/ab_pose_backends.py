"""ab_pose_backends.py — measured A/B of pose backends (RTMPose vs yolov8-pose).

The original "rtmpose beats yolo" call was eyeballed. Now we have hand-corrected
ground truth (round 1), so we can settle it with numbers: run BOTH backends on the
exact frames you corrected and compare PCK / pixel error per joint against your GT.

Why only the CORRECTED frames: untouched frames carry the raw RTMPose pre-labels,
so their GT == RTMPose output. Scoring there would hand RTMPose a free win. The
frames you edited are the only trustworthy ground truth, so we evaluate on those.
We find them by re-running RTMPose and taking frames whose GT differs from it (=
the joints you moved). NOTE: on an edited frame, joints you DIDN'T move still equal
RTMPose's position, so RTMPose has a small home-field edge on those — the fair,
meaningful comparison is the CONTACT joints (wrists/ankles), which you corrected.

Run (after any heavy render finishes — pose on CPU is slow):
  python scripts/ab_pose_backends.py
  python scripts/ab_pose_backends.py --clips 210297d7 28a2d8fb 78315c47 f64fecbd
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import zipfile

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
import utils  # noqa: E402

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
CONTACT = {"left_wrist", "right_wrist", "left_ankle", "right_ankle"}
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
# Clips the user actually corrected in round 1 (others contribute no GT).
DEFAULT_CLIPS = ["210297d7", "28a2d8fb", "78315c47", "f64fecbd"]


def load_gt(path):
    """{file_name: (17,3)} from a COCO-keypoints file (inside a .zip or plain)."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".json"))
            d = json.loads(z.read(name))
    else:
        d = json.load(open(path))
    id2name = {im["id"]: im["file_name"] for im in d["images"]}
    return {id2name[a["image_id"]]: np.array(a["keypoints"], dtype=float).reshape(-1, 3)
            for a in d["annotations"]}


def parse_name(fn):
    """'<clip-stem>__f<idx>.jpg' -> (stem, frame_index)."""
    base = fn[:-4] if fn.lower().endswith(".jpg") else fn
    stem, fidx = base.rsplit("__f", 1)
    return stem, int(fidx)


def extract_frames(filenames, input_dir):
    """Decode the exact frames named in `filenames` from their source videos.

    Returns {file_name: BGR image}; files whose video is missing are skipped.
    """
    by_clip = {}
    for fn in filenames:
        stem, fidx = parse_name(fn)
        by_clip.setdefault(stem, []).append((fidx, fn))
    images = {}
    for stem, items in by_clip.items():
        hits = glob.glob(os.path.join(input_dir, stem + "*"))
        vids = [h for h in hits if h.lower().endswith((".mp4", ".mov", ".avi"))]
        if not vids:
            print(f"  ! no video for clip {stem} — skipping {len(items)} frame(s)")
            continue
        cap = cv2.VideoCapture(vids[0])
        for fidx, fn in items:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if ok:
                images[fn] = frame
        cap.release()
    return images


def pick_climber(poses):
    """Largest confident skeleton = the climber (ignores belayer/bystander)."""
    best, best_score = None, -1.0
    for p in poses:
        kp = p.keypoints
        seen = kp[:, 2] > 0.1
        if seen.sum() < 4:
            continue
        xs, ys = kp[seen, 0], kp[seen, 1]
        score = ((xs.max() - xs.min()) + (ys.max() - ys.min())) * float(kp[seen, 2].mean())
        if score > best_score:
            best, best_score = kp, score
    return best


def run_rtmpose(images, mode, conf):
    import pose_rtmpose as pr
    model = pr.load_pose_model(mode)
    out = {}
    for fn, img in images.items():
        kp = pick_climber(pr.estimate_pose(model, img, conf))
        if kp is not None:
            out[fn] = kp
    return out


def run_yolov8(images, weights, conf):
    import pose_estimator as pe
    model = pe.load_pose_model(weights)
    out = {}
    for fn, img in images.items():
        kp = pick_climber(pe.estimate_pose(model, img, conf))
        if kp is not None:
            out[fn] = kp
    return out


def torso_size(kp):
    pts = kp[[L_SH, R_SH, L_HIP, R_HIP]]
    if (pts[:, 2] > 0).all():
        d = math.hypot(*(pts[:2, :2].mean(0) - pts[2:, :2].mean(0)))
        if d > 1:
            return d
    seen = kp[kp[:, 2] > 0]
    if len(seen) < 2:
        return None
    return math.hypot(np.ptp(seen[:, 0]), np.ptp(seen[:, 1])) or None


def evaluate(pred, gt, frames, alpha=0.2):
    """Per-joint pixel error + torso-normalized errors over `frames`."""
    err = {n: [] for n in KEYPOINT_NAMES}
    nerr = {n: [] for n in KEYPOINT_NAMES}
    for f in frames:
        gk, pk = gt[f], pred.get(f)
        if pk is None:
            continue
        scale = torso_size(gk)
        for i, n in enumerate(KEYPOINT_NAMES):
            if gk[i, 2] <= 0:
                continue
            d = math.hypot(*(gk[i, :2] - pk[i, :2]))
            err[n].append(d)
            if scale:
                nerr[n].append(d / scale)
    return err, nerr


def pck(vals, alpha):
    return 100.0 * np.mean([v <= alpha for v in vals]) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default=os.path.join(PROJECT_ROOT, "data/pose_annotation/round1/round1_corrected.zip"))
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS,
                    help="clip-stem prefixes to evaluate (default: the round-1 corrected clips)")
    ap.add_argument("--alpha", type=float, default=0.2)
    args = ap.parse_args()

    config = utils.load_config()
    conf = config["pose"]["confidence"]
    input_dir = utils.resolve_path(config["paths"]["input_dir"])

    gt = load_gt(args.gt)
    wanted = [fn for fn in gt if any(parse_name(fn)[0].startswith(c) for c in args.clips)]
    print(f"GT frames in selected clips: {len(wanted)}  (extracting from videos...)")
    images = extract_frames(wanted, input_dir)
    print(f"Decoded {len(images)} frames.\n")

    print(f"Running RTMPose ({config['models'].get('rtmpose_mode', 'balanced')}) on all frames...")
    rt = run_rtmpose(images, config["models"].get("rtmpose_mode", "balanced"), conf)

    # Edited frames = GT differs from RTMPose's own output (the joints you moved).
    edited = [f for f in images
              if f in rt and np.abs(gt[f][:, :2] - rt[f][:, :2]).sum() > 1.0]
    print(f"Corrected (ground-truth) frames detected: {len(edited)}\n")
    if not edited:
        print("No corrected frames found — nothing to compare.")
        return 1

    print("Running yolov8-pose on the corrected frames...")
    edited_imgs = {f: images[f] for f in edited}
    yo = run_yolov8(edited_imgs, config["models"]["pose_estimator"], conf)
    print(f"  yolov8 detected a climber on {len(yo)}/{len(edited)} frames "
          f"(RTMPose: {sum(1 for f in edited if f in rt)}/{len(edited)})\n")

    rt_err, rt_n = evaluate(rt, gt, edited, args.alpha)
    yo_err, yo_n = evaluate(yo, gt, edited, args.alpha)

    hdr = f"{'joint':16s} | {'RTMPose mean_px  PCK':>22s} | {'yolov8 mean_px  PCK':>22s}"
    print(hdr); print("-" * len(hdr))

    def cell(err, nerr, jn):
        if not err[jn]:
            return f"{'—':>11s}{'—':>11s}"
        return f"{np.mean(err[jn]):>9.1f}{pck(nerr[jn], args.alpha):>10.1f}%"

    for jn in KEYPOINT_NAMES:
        mark = " *" if jn in CONTACT else "  "
        print(f"{jn:16s}{mark}| {cell(rt_err, rt_n, jn)} | {cell(yo_err, yo_n, jn)}")

    def agg(errd, nerrd, names):
        e = [v for n in names for v in errd[n]]
        ne = [v for n in names for v in nerrd[n]]
        return (np.mean(e) if e else float("nan")), pck(ne, args.alpha)

    print("-" * len(hdr))
    for label, names in [("ALL joints", KEYPOINT_NAMES), ("CONTACT (h/f) *", sorted(CONTACT))]:
        rm, rp = agg(rt_err, rt_n, names)
        ym, yp = agg(yo_err, yo_n, names)
        print(f"{label:16s}  | {rm:>9.1f}{rp:>10.1f}% | {ym:>9.1f}{yp:>10.1f}%")
    print("\n* contact joints = the fair comparison (you corrected these). Lower px / "
          "higher PCK is better. RTMPose has a slight edge on non-contact joints you "
          "left in place (GT==RTMPose there).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
