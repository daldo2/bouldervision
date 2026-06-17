"""pose_rtmpose.py — RTMPose backend (top-down, via rtmlib / ONNX).

Same interface as the yolov8 path in `pose_estimator` (load_pose_model +
estimate_pose -> List[FramePose]), selectable via models.pose_backend = rtmpose.

Why it wins for climbing: RTMPose is **top-down** — a person detector finds the
climber and crops to them, then pose runs on that large, centred crop. That
fixes the failure mode of yolov8-pose and MediaPipe, which mislocate or miss a
small, oddly-posed climber in a wide-angle frame. On the szpagat test segment it
detected the climber on 100% of frames and kept each foot pinned to the correct
hold (no oscillation), where yolov8n's ankle was a coin-flip.

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


def load_pose_model(mode: str = "balanced"):
    """Create an rtmlib Body estimator (detector + RTMPose). `mode` is one of
    'lightweight' / 'balanced' / 'performance' (accuracy vs speed)."""
    from rtmlib import Body  # lazy: pulls onnxruntime
    return Body(mode=mode, backend="onnxruntime", device="cpu")


def estimate_pose(model, image: np.ndarray, confidence: float) -> List["pose_estimator.FramePose"]:
    """Run RTMPose on one frame -> FramePoses with 17 COCO keypoints.

    rtmlib returns keypoints already in COCO-17 order plus per-keypoint scores,
    which we carry through as confidence. `confidence` is accepted for interface
    parity (detection is gated inside rtmlib).
    """
    keypoints, scores = model(image)  # (N, 17, 2), (N, 17)
    poses: List[pose_estimator.FramePose] = []
    for person_kp, person_sc in zip(keypoints, scores):
        kp = np.zeros((17, 3), dtype=float)
        kp[:, :2] = person_kp
        kp[:, 2] = person_sc
        poses.append(pose_estimator.FramePose(keypoints=kp))
    return poses
