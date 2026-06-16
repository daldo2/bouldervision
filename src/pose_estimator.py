"""
pose_estimator.py — Phase 3 (scaffold).

Goal: track the climber's body through a video using YOLOv8-pose, which returns
17 COCO keypoints per detected person:

    0 nose        5 left_shoulder   11 left_hip      15 left_ankle
    1 left_eye    6 right_shoulder  12 right_hip     16 right_ankle
    2 right_eye   7 left_elbow      13 left_knee
    3 left_ear    8 right_elbow     14 right_knee
    4 right_ear   9 left_wrist
                 10 right_wrist

The "limbs that grab holds" are the wrists (hands) and ankles (feet). Phase 3
ties those keypoints to the nearest hold box per frame to decide what the
climber is touching.

This is a scaffold: model loading and single-frame keypoint extraction are
implemented; per-frame video iteration and hold-contact logic are TODOs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# COCO keypoint indices we care about for climbing contact.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
CONTACT_LIMBS = {
    "left_hand": 9,    # left_wrist
    "right_hand": 10,  # right_wrist
    "left_foot": 15,   # left_ankle
    "right_foot": 16,  # right_ankle
}


@dataclass
class FramePose:
    """Keypoints for one person in one frame."""
    # keypoints[i] = (x, y, confidence) for COCO joint i.
    keypoints: np.ndarray  # shape (17, 3)

    def limb_point(self, limb: str) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, conf) for a named contact limb, or None if unknown."""
        idx = CONTACT_LIMBS.get(limb)
        if idx is None:
            return None
        x, y, c = self.keypoints[idx]
        return float(x), float(y), float(c)


def load_pose_model(model_path: str):
    """Load (and auto-download) the YOLOv8 pose model."""
    from ultralytics import YOLO  # imported lazily; torch is heavy
    return YOLO(model_path)


def estimate_pose(model, image: np.ndarray, confidence: float) -> List[FramePose]:
    """Run pose estimation on a single image/frame.

    Returns one FramePose per detected person (usually just the climber).
    """
    results = model(image, conf=confidence, verbose=False)
    poses: List[FramePose] = []
    result = results[0]

    # `result.keypoints` is None if no person was detected.
    if result.keypoints is None or result.keypoints.data is None:
        return poses

    # `.data` is shape (num_people, 17, 3): (x, y, confidence) per joint.
    for person in result.keypoints.data:
        poses.append(FramePose(keypoints=person.cpu().numpy()))
    return poses


def point_to_box_distance(x: float, y: float, box: Tuple[int, int, int, int]) -> float:
    """Shortest distance from point (x, y) to a box's rectangle.

    0 if the point is inside the box; otherwise the distance to the nearest
    edge/corner. We use edge distance (not center distance) so a big volume and a
    tiny crimp are judged fairly — a hand resting anywhere on a large hold counts.
    """
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def touched_holds(
    pose: FramePose,
    hold_boxes: List[Tuple[int, int, int, int]],
    touch_distance_px: float,
    min_confidence: float,
) -> Dict[str, Optional[int]]:
    """Map each contact limb to the index of the hold it's touching (or None).

    For each limb keypoint confident enough (`min_confidence`), we find the
    nearest hold by edge distance; if that distance is within `touch_distance_px`
    the limb is considered to be touching that hold.

    Returned dict: {"left_hand": hold_index_or_None, ...}.
    """
    result: Dict[str, Optional[int]] = {}
    for limb, idx in CONTACT_LIMBS.items():
        x, y, conf = pose.keypoints[idx]
        if conf < min_confidence:
            result[limb] = None  # joint not reliably seen this frame
            continue

        nearest_idx, nearest_dist = None, float("inf")
        for i, box in enumerate(hold_boxes):
            dist = point_to_box_distance(float(x), float(y), box)
            if dist < nearest_dist:
                nearest_dist, nearest_idx = dist, i

        result[limb] = nearest_idx if nearest_dist <= touch_distance_px else None
    return result


def smooth_keypoint_sequence(poses: List[FramePose], window: int = 5) -> List[FramePose]:
    """Smooth jittery keypoints across time with a confidence-weighted moving
    average (centered window). Reduces frame-to-frame noise so contact decisions
    don't flicker. `window` should be odd; 1 returns the input unchanged.

    Each output position is the average of the surrounding `window` frames,
    weighting each by its keypoint confidence so unreliable joints contribute
    little. Confidence itself is averaged (unweighted).
    """
    if window <= 1 or len(poses) <= 1:
        return list(poses)

    data = np.stack([p.keypoints for p in poses]).astype(np.float64)  # (T, 17, 3)
    n_frames = data.shape[0]
    half = window // 2
    out = data.copy()

    for t in range(n_frames):
        lo, hi = max(0, t - half), min(n_frames, t + half + 1)
        seg = data[lo:hi]                       # (w, 17, 3)
        weights = np.clip(seg[:, :, 2:3], 1e-6, None)  # (w, 17, 1)
        out[t, :, :2] = (seg[:, :, :2] * weights).sum(axis=0) / weights.sum(axis=0)
        out[t, :, 2] = seg[:, :, 2].mean(axis=0)

    return [FramePose(keypoints=out[t]) for t in range(n_frames)]
