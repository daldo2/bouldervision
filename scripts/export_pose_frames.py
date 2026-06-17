"""Extract climbing frames + bootstrap keypoint pre-labels for a pose fine-tune.

Off-the-shelf 2D pose (even RTMPose) mislocates wrists/ankles in extreme climbing
poses — wide splits, heel hooks, far reaches. The durable fix is to fine-tune a
pose model on climbing frames with hand-corrected keypoints. But annotating 17
keypoints per frame from scratch is slow, so this script does what
`export_for_annotation.py` does for boxes: it samples frames from the clips, runs
the CURRENT pose backend, and writes its predictions as PRE-LABELS. You then only
CORRECT them in an annotation tool instead of clicking every joint.

Two products, both reused later:
  - an EVAL set: corrected keypoints become ground truth for scripts/eval_pose.py
    (PCK), so we can measure the current model before changing anything;
  - a TRAIN set: the same corrected keypoints fine-tune the model.

Output (gitignored — frames come from private clips):
  data/pose_annotation/
    images/                     one .jpg per sampled frame
    predictions_coco.json       COCO-keypoints: pre-labels to import & correct
    manifest.csv                clip,frame_index,file_name,num_kp,mean_score

Frames are named "<clip-stem>__f<frame_index>.jpg" so the eval/train split can be
done BY CLIP (never by frame — adjacent frames are near-identical and would leak).

Run (one video at a time is fine; pose on CPU is slow):
  source .venv/bin/activate
  python scripts/export_pose_frames.py                       # all clips in data/input
  python scripts/export_pose_frames.py data/input/foo.mp4    # one clip
  python scripts/export_pose_frames.py --every-seconds 0.4 --max-per-clip 60
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import cv2  # cv2 is light; the heavy pose import stays lazy (offline-safe)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import utils  # noqa: E402

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

# COCO-17 keypoint order (matches FramePose / RTMPose output) + the standard
# skeleton, so the JSON imports cleanly into CVAT / Roboflow / FiftyOne.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# 1-indexed pairs, COCO convention.
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]


def load_pose_backend(config):
    """Return (estimate_pose_fn, model) for the configured backend, lazily."""
    backend = config["models"].get("pose_backend", "rtmpose")
    if backend == "rtmpose":
        import pose_rtmpose as pose_src
        model = pose_src.load_pose_model(config["models"].get("rtmpose_mode", "balanced"))
    elif backend == "mediapipe":
        import pose_mediapipe as pose_src
        model = pose_src.load_pose_model(config["models"].get("pose_landmarker_task", "pose_landmarker.task"))
    else:
        import pose_estimator as pose_src
        model = pose_src.load_pose_model(config["models"]["pose_estimator"])
    return backend, pose_src, model


def pick_climber(poses):
    """From the people detected in a frame, return the climber's keypoints.

    Top-down pose can pick up a belayer/bystander; the climber is almost always
    the largest confident skeleton, so we score each person by its keypoint
    spread weighted by confidence and keep the top one. Returns an (17, 3) array
    or None if nothing was detected.
    """
    best, best_score = None, -1.0
    for p in poses:
        kp = p.keypoints
        conf = kp[:, 2]
        seen = conf > 0.1
        if seen.sum() < 4:  # too few joints to be a real person
            continue
        xs, ys = kp[seen, 0], kp[seen, 1]
        spread = (xs.max() - xs.min()) + (ys.max() - ys.min())
        score = spread * float(conf[seen].mean())
        if score > best_score:
            best, best_score = kp, score
    return best


def to_coco_keypoints(kp, conf_thresh):
    """Flatten (17, 3) keypoints to COCO [x, y, v]*17, mapping score -> visibility.

    v=2 confident (visible), v=1 detected-but-unsure, v=0 not labeled. Returns
    (flat_list, num_labeled).
    """
    flat, num = [], 0
    for x, y, c in kp:
        if c <= 0:
            flat += [0, 0, 0]
        else:
            v = 2 if c >= conf_thresh else 1
            flat += [round(float(x), 1), round(float(y), 1), v]
            num += 1
    return flat, num


def keypoint_bbox(kp, pad_frac=0.1):
    """COCO [x, y, w, h] around the labeled keypoints, padded a little."""
    seen = kp[:, 2] > 0
    xs, ys = kp[seen, 0], kp[seen, 1]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    x1, y1, x2, y2 = x1 - pw, y1 - ph, x2 + pw, y2 + ph
    return [round(float(x1), 1), round(float(y1), 1),
            round(float(x2 - x1), 1), round(float(y2 - y1), 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*",
                    help="video files (default: all videos in data/input)")
    ap.add_argument("--every-seconds", type=float, default=0.5,
                    help="sample one frame every N seconds of clip time (default 0.5)")
    ap.add_argument("--max-per-clip", type=int, default=80,
                    help="cap sampled frames per clip (default 80)")
    ap.add_argument("--config", default=None, help="settings.yaml override")
    args = ap.parse_args()

    config = utils.load_config(args.config) if args.config else utils.load_config()
    conf_thresh = config["pose"]["confidence"]

    input_dir = utils.resolve_path(config["paths"]["input_dir"])
    if args.videos:
        videos = [os.path.abspath(v) for v in args.videos]
    else:
        videos = sorted(
            p for p in glob.glob(os.path.join(input_dir, "*"))
            if p.lower().endswith(VIDEO_EXTS)
        )
    if not videos:
        print("No videos found. Pass a file or drop clips in data/input/.")
        return 1

    out_dir = os.path.join(PROJECT_ROOT, "data", "pose_annotation")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"Loading pose backend ({config['models'].get('pose_backend')}) ...")
    backend, pose_src, model = load_pose_backend(config)

    images, annotations, manifest = [], [], []
    img_id = ann_id = 0

    for vpath in videos:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  ! could not open {vpath}, skipping")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, round(fps * args.every_seconds))
        targets = list(range(0, total, step))[: args.max_per_clip] if total else []
        print(f"  {stem}: {total} frames @ {fps:.0f}fps -> sampling {len(targets)} "
              f"(every {step} frames)")

        kept = 0
        for fidx in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                continue
            poses = pose_src.estimate_pose(model, frame, conf_thresh)
            kp = pick_climber(poses)
            if kp is None:
                continue  # no climber detected; not worth annotating

            h, w = frame.shape[:2]
            fname = f"{stem}__f{fidx}.jpg"
            cv2.imwrite(os.path.join(img_dir, fname), frame)

            flat, num = to_coco_keypoints(kp, conf_thresh)
            images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "keypoints": flat, "num_keypoints": num,
                "bbox": keypoint_bbox(kp), "iscrowd": 0,
            })
            mean_score = float(kp[kp[:, 2] > 0, 2].mean()) if (kp[:, 2] > 0).any() else 0.0
            manifest.append([stem, fidx, fname, num, round(mean_score, 3)])
            img_id += 1
            ann_id += 1
            kept += 1
        cap.release()
        print(f"    kept {kept} frames with a climber")

    coco = {
        "info": {"description": f"BoulderVision pose pre-labels ({backend}); CORRECT these",
                 "categories_note": "category 1 = climber, COCO-17 keypoints"},
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1, "name": "climber", "supercategory": "person",
            "keypoints": KEYPOINT_NAMES, "skeleton": SKELETON,
        }],
    }
    with open(os.path.join(out_dir, "predictions_coco.json"), "w") as f:
        json.dump(coco, f, indent=2)
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        csv.writer(f).writerows(
            [["clip", "frame_index", "file_name", "num_kp", "mean_score"], *manifest])

    print(f"\nWrote {len(images)} frames -> {img_dir}")
    print(f"Pre-labels  -> {os.path.join(out_dir, 'predictions_coco.json')}")
    print("Next: import images + predictions_coco.json into CVAT/Roboflow (COCO "
          "Keypoints), CORRECT the joints, then split BY CLIP for eval vs train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
