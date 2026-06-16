"""perspective.py — rectify an angled wall to a frontal view for route grouping.

On a photo taken down the length of a wall, holds far from the camera project
small and bunched together, so spatial clustering in image pixels cannot tell
routes apart — they all fall into one compact band. The fix is to undo the
projection: given the wall's four corners in the image, we build the homography
that maps that quadrilateral to a rectangle (a head-on view) and apply it to
each hold's position/size BEFORE the spatial step. Colour reading is unaffected
(it already happened on the original crop).

We deliberately warp only the holds' *coordinates*, not the whole image — that
is all the clustering needs, and it is cheap. A full-image warp is offered
separately (`warp_image`) only as a visual sanity check of the chosen corners.

Pure + offline: OpenCV geometry only, no model.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def order_corners(pts: Sequence[Point]) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    Uses the classic sum/difference trick: TL has the smallest x+y, BR the
    largest; TR has the smallest (y-x), BL the largest. Robust for the gently
    rotated quadrilaterals a wall produces.
    """
    p = np.asarray(pts, dtype=np.float64)
    if p.shape != (4, 2):
        raise ValueError("exactly 4 (x, y) corners are required")
    s = p.sum(axis=1)
    d = p[:, 1] - p[:, 0]
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]],
                    dtype=np.float32)


def compute_homography(corners: Sequence[Point]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Homography mapping the wall quad to a frontal rectangle.

    The output rectangle's size is taken from the quad's own edge lengths (max
    of opposite sides) so the rectified plane keeps a roughly isotropic, real-
    world scale — important because the spatial clustering measures distances
    there. Returns (H, (out_w, out_h)).
    """
    tl, tr, br, bl = order_corners(corners)
    out_w = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    out_h = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    out_w, out_h = max(out_w, 1), max(out_h, 1)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst), (out_w, out_h)


def warp_points(H: np.ndarray, pts: Sequence[Point]) -> np.ndarray:
    """Apply homography H to a list of (x, y) points -> (N, 2) array."""
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, H).reshape(-1, 2)


def rectify_holds(holds: list, H: np.ndarray) -> None:
    """Set each hold's `rect_box` to its box warped into the rectified plane.

    Warps all four box corners and takes their axis-aligned bounds, so both the
    rectified centre and the rectified size (used by the adaptive spatial split)
    reflect the head-on view. Mutates the holds in place.
    """
    for h in holds:
        x1, y1, x2, y2 = h.box
        w = warp_points(H, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        xs, ys = w[:, 0], w[:, 1]
        h.rect_box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def warp_image(image: np.ndarray, H: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Warp a full image to the rectified plane — for eyeballing corner choices."""
    return cv2.warpPerspective(image, H, out_size)
