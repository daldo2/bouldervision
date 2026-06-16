"""stabilize.py — keep static hold boxes aligned under a moving (handheld) camera.

Holds are detected once on a reference frame, but a handheld camera drifts, so
those fixed boxes slide off the real holds (a green "contact" box ends up on bare
wall). This module estimates, per frame, the rigid-ish transform from the
reference frame to the current frame from matched features, and warps the hold
boxes by it. RANSAC rejects the climber's moving features as outliers, so the
estimate locks onto the static wall. Hold *indices* are preserved (we warp the
same reference boxes), so the contact summary stays meaningful.

Best for translation/rotation/zoom drift on a roughly planar wall. Big viewpoint
swings need a homography (not done here). Pure OpenCV, no model.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def warp_box(box: Box, M: np.ndarray) -> Box:
    """Apply a 2x3 affine M to a box; return the axis-aligned bounds of the result."""
    x1, y1, x2, y2 = box
    pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    w = cv2.transform(pts, M).reshape(-1, 2)
    return (int(w[:, 0].min()), int(w[:, 1].min()), int(w[:, 0].max()), int(w[:, 1].max()))


class CameraStabilizer:
    """Maps reference-frame coordinates to a later frame via feature matching."""

    def __init__(self, ref_bgr: np.ndarray, max_features: int = 1500, min_matches: int = 12):
        self.orb = cv2.ORB_create(max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = min_matches
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(ref_gray, None)

    def transform_to(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """2x3 affine mapping reference coords -> this frame, or None if unsure.

        Uses RANSAC so the climber's (outlier) motion doesn't pull the estimate
        off the static wall.
        """
        if self.ref_des is None:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < self.min_matches:
            return None
        matches = self.matcher.match(self.ref_des, des)
        if len(matches) < self.min_matches:
            return None
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in matches])
        dst = np.float32([kp[m.trainIdx].pt for m in matches])
        M, _inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0
        )
        return M

    def warp_boxes(self, boxes: List[Box], frame_bgr: np.ndarray) -> List[Box]:
        """Warp reference hold boxes into the given frame. Identity if unsure."""
        M = self.transform_to(frame_bgr)
        if M is None:
            return list(boxes)
        return [warp_box(b, M) for b in boxes]
