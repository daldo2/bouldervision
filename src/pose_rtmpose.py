"""pose_rtmpose.py — RTMPose backend (top-down, via rtmlib / ONNX).

Same interface as the yolov8 path in `pose_estimator` (load_pose_model +
estimate_pose -> List[FramePose]), selectable via models.pose_backend = rtmpose.

Why it wins for climbing: RTMPose is **top-down** — a person detector finds the
climber and crops to them, then pose runs on that large, centred crop. That
fixes the failure mode of yolov8-pose and MediaPipe, which mislocate or miss a
small, oddly-posed climber in a wide-angle frame. It beat both in an A/B (visual
and PCK).

Two model families:
  - **Body** (COCO-17): body joints only; foot/hand contact points are then
    EXTRAPOLATED from the ankle/wrist downstream.
  - **Wholebody** (COCO-WholeBody, 133 kpts): adds real foot (toes/heel) and hand
    (finger) keypoints. We keep the 17 body joints for the skeleton AND attach
    precise contact points (big toe per foot, fingertip mean per hand) so a foot
    standing on its toes / a hand gripping is localized where it actually touches
    — no extrapolation guesswork. We ignore the 68 face keypoints entirely.

Runs on CPU via onnxruntime — no torch / mmcv. rtmlib auto-downloads the ONNX
weights on first use (network once, then cached). Heavy import is lazy.
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_estimator  # noqa: E402  (FramePose; no torch)

# COCO-WholeBody indices (0-16 body, then:)
#   foot:  17 L-big-toe 18 L-small-toe 19 L-heel  20 R-big-toe 21 R-small-toe 22 R-heel
#   hand:  91 = left-hand root, 112 = right-hand root; 21 joints each, fingertips at
#          root + {4 thumb, 8 index, 12 middle, 16 ring, 20 pinky}.
L_BIG_TOE, R_BIG_TOE = 17, 20
L_HEEL, R_HEEL = 19, 22
LEFT_FINGERTIPS = [91 + 8, 91 + 12, 91 + 16, 91 + 20]    # index, middle, ring, pinky tips
RIGHT_FINGERTIPS = [112 + 8, 112 + 12, 112 + 16, 112 + 20]
# Points drawn (display only) to show the contact AREA: 3 fingertips per hand,
# heel + big toe per foot. Contact LOGIC still uses the single point per limb.
LEFT_HAND_DISP = [91 + 8, 91 + 12, 91 + 16]              # index, middle, ring tips
RIGHT_HAND_DISP = [112 + 8, 112 + 12, 112 + 16]


def load_pose_model(mode: str = "balanced", wholebody: bool = False):
    """Create an rtmlib estimator (detector + pose). `mode` is one of
    'lightweight' / 'balanced' / 'performance'. `wholebody=True` uses the 133-kpt
    Wholebody model (real foot/hand points); otherwise the COCO-17 Body model."""
    if wholebody:
        from rtmlib import Wholebody  # lazy: pulls onnxruntime
        return Wholebody(mode=mode, backend="onnxruntime", device="cpu")
    from rtmlib import Body
    return Body(mode=mode, backend="onnxruntime", device="cpu")


def _contact_points(kp: np.ndarray, sc: np.ndarray) -> dict:
    """Build {limb: (x, y, conf)} contact points from whole-body keypoints.

    Foot -> big toe; hand -> mean of the four fingertips (where a grip lands).
    Confidence carries through so a low-confidence point lets limb_points fall
    back to extrapolation for that limb.
    """
    def tip_mean(idxs):
        pts = kp[idxs]
        confs = sc[idxs]
        return float(pts[:, 0].mean()), float(pts[:, 1].mean()), float(confs.mean())

    def pt(i):
        return float(kp[i, 0]), float(kp[i, 1]), float(sc[i])

    return {
        # Single point per limb used by the contact LOGIC (toe / mean of fingertips).
        "left_foot": pt(L_BIG_TOE),
        "right_foot": pt(R_BIG_TOE),
        "left_hand": tip_mean(LEFT_FINGERTIPS),
        "right_hand": tip_mean(RIGHT_FINGERTIPS),
        # Extra points for DISPLAY ONLY (the contact area): 3 fingertips per hand,
        # heel + big toe per foot. Ignored by limb_points (not a CONTACT_LIMBS key).
        "_display": {
            "left_hand": [pt(i) for i in LEFT_HAND_DISP],
            "right_hand": [pt(i) for i in RIGHT_HAND_DISP],
            "left_foot": [pt(L_HEEL), pt(L_BIG_TOE)],
            "right_foot": [pt(R_HEEL), pt(R_BIG_TOE)],
        },
    }


def estimate_pose(model, image: np.ndarray, confidence: float) -> List["pose_estimator.FramePose"]:
    """Run RTMPose on one frame -> FramePoses with 17 COCO body keypoints.

    For the Wholebody model we also attach precise contact_pts (real toe/finger).
    `confidence` is accepted for interface parity (detection is gated in rtmlib).
    """
    keypoints, scores = model(image)  # (N, K, 2), (N, K); K is 17 (Body) or 133 (Wholebody)
    poses: List[pose_estimator.FramePose] = []
    for person_kp, person_sc in zip(keypoints, scores):
        body = np.zeros((17, 3), dtype=float)
        body[:, :2] = person_kp[:17]
        body[:, 2] = person_sc[:17]
        contact = _contact_points(person_kp, person_sc) if person_kp.shape[0] >= 23 else None
        poses.append(pose_estimator.FramePose(keypoints=body, contact_pts=contact))
    return poses
