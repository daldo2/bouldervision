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

# (base joint, tip joint) per limb. The real contact point (hand/foot) is BEYOND
# the tip joint, continuing the base->tip direction — we extrapolate to it so a
# wrist keypoint isn't judged against a hold the fingers are actually on.
LIMB_JOINTS = {
    "left_hand": (7, 9),    # left_elbow  -> left_wrist
    "right_hand": (8, 10),  # right_elbow -> right_wrist
    "left_foot": (13, 15),  # left_knee   -> left_ankle
    "right_foot": (14, 16), # right_knee  -> right_ankle
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


def extremity_point(
    keypoints: np.ndarray, limb: str, reach_frac: float = 0.33, min_joint_conf: float = 0.3
) -> Tuple[float, float, float]:
    """Estimate a limb's actual contact point (hand/foot), not just the joint.

    Continues the base->tip direction (elbow->wrist, knee->ankle) a fraction
    `reach_frac` past the tip, so the point sits closer to where the hand/foot
    really meets the wall. If the base joint isn't confidently seen we can't get
    a direction, so we fall back to the tip joint itself. Returns (x, y, conf)
    where conf is the tip joint's confidence.
    """
    base_i, tip_i = LIMB_JOINTS[limb]
    bx, by, bc = keypoints[base_i]
    tx, ty, tc = keypoints[tip_i]
    if bc < min_joint_conf:
        return float(tx), float(ty), float(tc)
    return float(tx + reach_frac * (tx - bx)), float(ty + reach_frac * (ty - by)), float(tc)


def limb_points(
    pose: FramePose, reach_frac: float = 0.33, min_joint_conf: float = 0.3
) -> Dict[str, Tuple[float, float, float]]:
    """Extremity contact points for all four contact limbs of one pose."""
    return {
        limb: extremity_point(pose.keypoints, limb, reach_frac, min_joint_conf)
        for limb in CONTACT_LIMBS
    }


def nearest_hold_index(
    x: float, y: float, hold_boxes: List[Tuple[int, int, int, int]], touch_distance_px: float
) -> Optional[int]:
    """Index of the nearest hold within `touch_distance_px` of (x, y), else None."""
    nearest_idx, nearest = None, float("inf")
    for i, box in enumerate(hold_boxes):
        d = point_to_box_distance(x, y, box)
        if d < nearest:
            nearest, nearest_idx = d, i
    return nearest_idx if nearest <= touch_distance_px else None


def frame_contacts(
    points: Dict[str, Tuple[float, float, float]],
    prev_points: Optional[Dict[str, Tuple[float, float, float]]],
    hold_boxes: List[Tuple[int, int, int, int]],
    touch_distance_px: float,
    max_speed_px: float,
    min_confidence: float,
) -> Dict[str, Optional[int]]:
    """Decide each limb's held hold this frame, gating out reaching/hovering.

    A limb counts as gripping only if it is (a) confidently seen, (b) *anchored*
    — moving slower than `max_speed_px` from the previous frame, since a planted
    hand barely moves while a reaching/hovering one travels — and (c) within
    `touch_distance_px` of a hold. This removes most false contacts where a limb
    merely passes near a hold in the 2D projection.
    """
    result: Dict[str, Optional[int]] = {}
    for limb, (x, y, c) in points.items():
        if c < min_confidence:
            result[limb] = None
            continue
        if prev_points is not None and prev_points.get(limb) is not None:
            px, py, _ = prev_points[limb]
            if math.hypot(x - px, y - py) > max_speed_px:
                result[limb] = None  # moving too fast to be gripping
                continue
        result[limb] = nearest_hold_index(x, y, hold_boxes, touch_distance_px)
    return result


def smooth_contact_sequence(
    seq: List[Optional[int]], max_gap: int = 8, min_run: int = 2
) -> List[Optional[int]]:
    """Clean one limb's frame-by-frame hold sequence (hysteresis).

    Two passes that make contacts match reality during dynamic moves:
      1. Bridge short dropouts: a None gap of <= `max_gap` frames flanked by the
         *same* hold on both sides is filled — the climber didn't actually let go
         for a few frames (a speed blip or a momentary low-confidence joint).
      2. Drop flicker: a contact run shorter than `min_run` frames is removed —
         a one- or two-frame touch is almost always a 2D near-pass, not a grip.

    Returns a new list the same length as `seq`.
    """
    out = list(seq)
    n = len(out)

    i = 0
    while i < n:                       # 1. gap fill
        if out[i] is None:
            j = i
            while j < n and out[j] is None:
                j += 1
            left = out[i - 1] if i - 1 >= 0 else None
            right = out[j] if j < n else None
            if left is not None and left == right and (j - i) <= max_gap:
                for k in range(i, j):
                    out[k] = left
            i = j
        else:
            i += 1

    i = 0
    while i < n:                       # 2. drop short runs
        if out[i] is not None:
            j = i
            while j < n and out[j] == out[i]:
                j += 1
            if (j - i) < min_run:
                for k in range(i, j):
                    out[k] = None
            i = j
        else:
            i += 1

    return out


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
