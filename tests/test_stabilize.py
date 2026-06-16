"""
Tests for camera stabilization geometry (handheld hold tracking).

Offline, no model: synthetic textured frame shifted by a known amount; we check
the estimated transform recovers the shift and warps boxes accordingly.

Run with:  pytest tests/
"""

import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import stabilize as stab  # noqa: E402


def test_warp_box_translation():
    M = np.float32([[1, 0, 10], [0, 1, -5]])  # shift +10x, -5y
    assert stab.warp_box((100, 100, 140, 140), M) == (110, 95, 150, 135)


def _textured_frame(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((400, 400, 3), 30, np.uint8)
    # scatter bright blobs so ORB has corners to latch onto
    for _ in range(120):
        x, y = rng.integers(20, 380), rng.integers(20, 380)
        cv2.rectangle(img, (x, y), (x + 6, y + 6), (220, 220, 220), -1)
    return img


def test_stabilizer_recovers_translation_and_tracks_box():
    ref = _textured_frame()
    shift_x, shift_y = 18, -12
    M_true = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    moved = cv2.warpAffine(ref, M_true, (400, 400))

    s = stab.CameraStabilizer(ref)
    boxes = s.warp_boxes([(150, 150, 190, 190)], moved)
    x1, y1, x2, y2 = boxes[0]
    # the box should have followed the camera shift (within a couple px)
    assert abs(x1 - (150 + shift_x)) <= 3
    assert abs(y1 - (150 + shift_y)) <= 3
    assert abs((x2 - x1) - 40) <= 3  # size roughly preserved
