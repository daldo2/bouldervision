"""pose_mediapipe.py — MediaPipe Pose backend (alternative to yolov8-pose).

Same interface as the pose functions in `pose_estimator` (load_pose_model +
estimate_pose -> List[FramePose]) so the video pipeline can A/B the two backends
behind a config switch. MediaPipe BlazePose gives two things yolov8-pose doesn't:

  - a per-landmark **visibility** score (calibrated occlusion signal), which we
    feed through as the keypoint confidence — so a hidden limb reads as low conf;
  - **3D world landmarks** (depth, from one camera) — not used yet, but the hook
    is here for resolving on-hold vs. hovering later.

We run the IMAGE (per-frame) running mode: its VIDEO tracking mode loses our
small, oddly-posed climbers, while per-frame detection finds them reliably.
MediaPipe returns 33 landmarks; we remap to the 17 COCO joints the rest of the
pipeline expects. Heavy imports are lazy, matching the offline-safe convention.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_estimator  # noqa: E402  (for FramePose; no torch)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# COCO joint i  <-  MediaPipe BlazePose landmark index
_COCO_FROM_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def load_pose_model(model_path: str = "pose_landmarker.task", det_confidence: float = 0.3):
    """Create a MediaPipe PoseLandmarker (IMAGE mode). Resolves the .task model
    inside models/ if a bare name is given."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    path = model_path if os.path.exists(model_path) else os.path.join(MODELS_DIR, os.path.basename(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=path),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=det_confidence,
        min_pose_presence_confidence=det_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def estimate_pose(model, image: np.ndarray, confidence: float) -> List["pose_estimator.FramePose"]:
    """Run MediaPipe on one frame -> FramePoses with 17 COCO keypoints.

    `confidence` is accepted for interface parity but pose detection is gated at
    model-creation time; the per-keypoint visibility flows through as the joint
    confidence, which downstream gating then uses.
    """
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = model.detect(mp_image)
    h, w = image.shape[:2]
    poses: List[pose_estimator.FramePose] = []
    for person in result.pose_landmarks:
        kp = np.zeros((17, 3), dtype=float)
        for coco_i, mp_i in enumerate(_COCO_FROM_MP):
            lm = person[mp_i]
            kp[coco_i] = (lm.x * w, lm.y * h, lm.visibility)
        poses.append(pose_estimator.FramePose(keypoints=kp))
    return poses
