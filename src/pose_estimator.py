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

# COCO-17 skeleton edges (0-indexed joint pairs) for drawing the stick figure.
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),                  # nose-eyes-ears
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),         # shoulders + arms
    (5, 11), (6, 12), (11, 12),                      # torso
    (11, 13), (13, 15), (12, 14), (14, 16),          # legs
]


@dataclass
class FramePose:
    """Keypoints for one person in one frame."""
    # keypoints[i] = (x, y, confidence) for COCO joint i.
    keypoints: np.ndarray  # shape (17, 3)
    # Optional precise contact points {limb: (x, y, conf)} from a whole-body model
    # (real big-toe / fingertips). When present and confident, limb_points uses
    # these instead of extrapolating from the wrist/ankle. None => extrapolate.
    contact_pts: Optional[Dict[str, Tuple[float, float, float]]] = None

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


def body_scale(keypoints: np.ndarray, min_conf: float = 0.2) -> Optional[float]:
    """A body-size reference in pixels, for scale-invariant contact thresholds.

    Returns the mid-shoulder→mid-hip (torso) distance when shoulders+hips are
    confidently seen — a stable proxy for how big the climber appears, independent
    of video resolution or distance. Falls back to ~1.5× shoulder width, else None.
    Expressing touch/release distances as a FRACTION of this means a threshold
    catches the same way whether the climber is 74px or 740px tall on screen.
    """
    sh = keypoints[[5, 6]]
    hp = keypoints[[11, 12]]
    if (sh[:, 2] > min_conf).all() and (hp[:, 2] > min_conf).all():
        d = math.hypot(*(sh[:, :2].mean(0) - hp[:, :2].mean(0)))
        if d > 1:
            return float(d)
    if (sh[:, 2] > min_conf).all():
        w = math.hypot(*(sh[0, :2] - sh[1, :2]))
        if w > 1:
            return float(w) * 1.5
    return None


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
    pose: FramePose, reach_frac: float = 0.33,
    reach_frac_foot: Optional[float] = None, min_joint_conf: float = 0.3
) -> Dict[str, Tuple[float, float, float]]:
    """Extremity contact points for all four contact limbs of one pose.

    Hands and feet use different extrapolation: a hand reaches well past the wrist
    (fingers on the hold), so `reach_frac` ~0.33 is right. But a foot's contact is
    essentially at the ankle — extrapolating along the shin (knee->ankle) overshoots
    *below* where the shoe meets the wall — so feet use the smaller `reach_frac_foot`
    (defaults to `reach_frac` for backward compatibility when not given).
    """
    foot_reach = reach_frac if reach_frac_foot is None else reach_frac_foot
    out: Dict[str, Tuple[float, float, float]] = {}
    for limb in CONTACT_LIMBS:
        # Prefer a real whole-body contact point (big toe / fingertips) when it's
        # confident; otherwise fall back to extrapolating from wrist/ankle.
        cp = pose.contact_pts.get(limb) if pose.contact_pts else None
        if cp is not None and cp[2] >= min_joint_conf:
            out[limb] = cp
            continue
        r = foot_reach if limb.endswith("_foot") else reach_frac
        out[limb] = extremity_point(pose.keypoints, limb, r, min_joint_conf)
    return out


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


def flag_implausible_keypoints(
    poses: List[FramePose], collapse_px: float = 40.0, jump_px: float = 120.0
) -> List[FramePose]:
    """Mark mislocalized/occluded contact joints as unreliable (confidence -> 0).

    The model emits a confident position even for a hidden limb, often by
    collapsing one joint onto its partner (both ankles on one foot during a wide
    split) or teleporting it across the frame. We detect these *geometrically*,
    not by trusting confidence:

      - COLLAPSE: the left and right wrist (or ankle) sit within `collapse_px` of
        each other — impossible for real hands/feet, so the one that moved more
        from the previous frame is the mislocated one and gets zeroed.
      - JUMP: a joint that leapt more than `jump_px` between frames (faster than a
        limb can move) is zeroed for that frame.

    Zeroing the confidence makes downstream treat the limb as occluded: smoothing
    interpolates it from neighbours, contact resolution holds its last hold
    instead of grabbing a wrong one, and the overlay shows nothing rather than a
    false marker. Pure: returns new FramePoses.
    """
    if not poses:
        return list(poses)
    seq = [p.keypoints.copy() for p in poses]
    pairs = [(9, 10), (15, 16)]  # (left, right) wrist, then ankle
    for t, cur in enumerate(seq):
        prev = seq[t - 1] if t > 0 else None
        for li, ri in pairs:
            lx, ly, lc = cur[li]
            rx, ry, rc = cur[ri]
            if lc <= 0 or rc <= 0:
                continue
            if math.hypot(lx - rx, ly - ry) < collapse_px and prev is not None:
                lmove = math.hypot(lx - prev[li][0], ly - prev[li][1]) if prev[li][2] > 0 else float("inf")
                rmove = math.hypot(rx - prev[ri][0], ry - prev[ri][1]) if prev[ri][2] > 0 else float("inf")
                cur[li if lmove > rmove else ri][2] = 0.0  # the jumper is the bad one
        if prev is not None:
            for idx in (9, 10, 15, 16):
                if cur[idx][2] > 0 and prev[idx][2] > 0 and \
                        math.hypot(cur[idx][0] - prev[idx][0], cur[idx][1] - prev[idx][1]) > jump_px:
                    cur[idx][2] = 0.0
    return [FramePose(keypoints=k, contact_pts=p.contact_pts) for k, p in zip(seq, poses)]


def enforce_lr_consistency(poses: List[FramePose], min_conf: float = 0.3) -> List[FramePose]:
    """Fix left/right limb-label swaps across a pose sequence.

    yolov8-pose can flip which ankle/wrist is "left" vs "right" from frame to
    frame (common in a wide split), which makes a planted foot's hold oscillate
    and fabricates moves. Walking the sequence, we keep each limb pair assigned so
    motion stays continuous: if swapping this frame's left/right tip brings both
    closer to the previous frame's positions, we swap the whole limb (joint+tip).

    Pure: returns new FramePoses, originals untouched. Low-confidence frames are
    left as-is (no reliable basis to decide).
    """
    if len(poses) <= 1:
        return list(poses)
    seq = [p.keypoints.copy() for p in poses]
    # ((left_base, left_tip), (right_base, right_tip)) for arms then legs
    groups = [((7, 9), (8, 10)), ((13, 15), (14, 16))]
    for t in range(1, len(seq)):
        prev, cur = seq[t - 1], seq[t]
        for (lb, lt), (rb, rt) in groups:
            lx, ly, lc = cur[lt]; rx, ry, rc = cur[rt]
            plx, ply, plc = prev[lt]; prx, pry, prc = prev[rt]
            if min(lc, rc, plc, prc) < min_conf:
                continue
            same = math.hypot(lx - plx, ly - ply) + math.hypot(rx - prx, ry - pry)
            swapped = math.hypot(lx - prx, ly - pry) + math.hypot(rx - plx, ry - ply)
            if swapped < same:
                cur[[lt, rt]] = cur[[rt, lt]]
                cur[[lb, rb]] = cur[[rb, lb]]
    return [FramePose(keypoints=k, contact_pts=p.contact_pts) for k, p in zip(seq, poses)]


def resolve_contact_sequence(
    points_per_frame: List[Optional[Dict[str, Tuple[float, float, float]]]],
    boxes_per_frame: List[List[Tuple[int, int, int, int]]],
    touch_distance_px: float,
    release_distance_px: float,
    max_speed_px: float,
    min_confidence: float,
    engage_min_frames: int = 1,
) -> List[Dict[str, Optional[int]]]:
    """Stateful "sticky" contact resolution over a whole clip.

    Models how a limb actually uses a hold:
      - ENGAGE: a limb grabs the nearest hold only when it settles there — within
        `touch_distance_px`, moving slower than `max_speed_px`, AND it has stayed
        that way on the SAME hold for `engage_min_frames` consecutive frames. The
        dwell requirement stops an eager grip from latching the instant a hand
        passes near a hold; it must actually come to rest there. (1 = engage
        immediately, the original behavior.)
      - STAY (sticky): once gripping a hold it keeps gripping while it stays
        within `release_distance_px` of THAT hold, regardless of speed — so a
        foot adjusting on a hold (a brief fast wiggle) isn't dropped. It releases
        only when it clearly leaves the hold's neighborhood.

    This fixes the "leg disappears mid-split" case: a planted foot that briefly
    speeds up to reposition stays gripped instead of flickering off. State is
    kept across confidence dropouts / missing poses so short gaps don't release.

    `points_per_frame[t]` is {limb: (x, y, conf)} or None (no climber that frame);
    `boxes_per_frame[t]` are that frame's hold boxes (already stabilized if moving).
    Returns a per-frame {limb: hold_index or None}.
    """
    state: Dict[str, Optional[int]] = {limb: None for limb in CONTACT_LIMBS}
    # (candidate hold, consecutive qualifying frames) per limb, for the engage dwell.
    pending: Dict[str, Tuple[Optional[int], int]] = {limb: (None, 0) for limb in CONTACT_LIMBS}
    prev: Optional[Dict[str, Tuple[float, float, float]]] = None
    out: List[Dict[str, Optional[int]]] = []

    for pts, boxes in zip(points_per_frame, boxes_per_frame):
        frame: Dict[str, Optional[int]] = {limb: None for limb in CONTACT_LIMBS}
        if pts is None:
            out.append(frame)   # keep `state` so a brief gap doesn't release
            pending = {limb: (None, 0) for limb in CONTACT_LIMBS}  # must re-settle after a gap
            prev = None
            continue
        for limb in CONTACT_LIMBS:
            x, y, c = pts[limb]
            cur = state[limb]
            if c < min_confidence:
                frame[limb] = cur  # can't see it this frame; assume unchanged
                pending[limb] = (None, 0)
                continue
            if cur is not None and cur < len(boxes) and \
                    point_to_box_distance(x, y, boxes[cur]) <= release_distance_px:
                frame[limb] = cur  # sticky: still on the gripped hold
                pending[limb] = (None, 0)
                continue
            idx = nearest_hold_index(x, y, boxes, touch_distance_px)
            speed = math.hypot(x - prev[limb][0], y - prev[limb][1]) if prev and prev.get(limb) else 0.0
            if idx is not None and speed <= max_speed_px:
                # Near and settled — count consecutive frames on this same hold.
                run = pending[limb][1] + 1 if pending[limb][0] == idx else 1
                pending[limb] = (idx, run)
                if run >= engage_min_frames:
                    state[limb] = idx
                    frame[limb] = idx
                else:
                    state[limb] = None
                    frame[limb] = None  # candidate, not yet committed
            else:
                pending[limb] = (None, 0)
                state[limb] = None
                frame[limb] = None
        out.append(frame)
        prev = pts
    return out


def mode_smooth_contacts(seq: List[Optional[int]], window: int) -> List[Optional[int]]:
    """Replace each frame's hold with the most common value in a window around it.

    Kills oscillation between two adjacent holds (a foot jittering #23<->#29):
    whichever hold the limb is actually on dominates the local window and wins,
    so the assignment stays put instead of flickering. `window` should span ~the
    minimum real dwell (e.g. 0.5 s) so genuine moves still come through.
    """
    if window <= 1 or len(seq) <= 1:
        return list(seq)
    from collections import Counter
    n, half = len(seq), window // 2
    out = list(seq)
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        out[t] = Counter(seq[lo:hi]).most_common(1)[0][0]
    return out


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

    # Carry the precise contact points through unchanged (they live outside the
    # 17-joint array, so the moving average doesn't touch them).
    return [FramePose(keypoints=out[t], contact_pts=poses[t].contact_pts) for t in range(n_frames)]
