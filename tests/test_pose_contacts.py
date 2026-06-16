"""
Tests for the improved contact logic: extremity extrapolation + velocity gate.

Offline and model-free — synthetic keypoints and points, no YOLO.

Run with:  pytest tests/
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import pose_estimator as pe  # noqa: E402


def make_keypoints(joints):
    """17x3 keypoints array; `joints` maps joint_index -> (x, y, conf)."""
    kp = np.zeros((17, 3), dtype=float)
    for idx, val in joints.items():
        kp[idx] = val
    return kp


def test_extremity_extrapolates_past_wrist():
    # left elbow (7) at (100,100), left wrist (9) at (120,100): hand is further right.
    kp = make_keypoints({7: (100, 100, 0.9), 9: (120, 100, 0.9)})
    x, y, c = pe.extremity_point(kp, "left_hand", reach_frac=0.5)
    assert x == 130 and y == 100   # 120 + 0.5*(120-100)
    assert c == 0.9


def test_extremity_falls_back_to_tip_when_base_unseen():
    kp = make_keypoints({7: (0, 0, 0.0), 9: (120, 100, 0.8)})  # elbow not seen
    x, y, c = pe.extremity_point(kp, "left_hand", reach_frac=0.5)
    assert (x, y, c) == (120, 100, 0.8)  # no direction -> use the wrist itself


HOLDS = [(200, 200, 240, 240)]  # one hold


def test_contact_when_anchored_and_near():
    pts = {"left_hand": (220, 220, 0.9)}        # inside the hold
    prev = {"left_hand": (219, 220, 0.9)}        # barely moved -> anchored
    out = pe.frame_contacts(pts, prev, HOLDS, touch_distance_px=40, max_speed_px=12, min_confidence=0.5)
    assert out["left_hand"] == 0


def test_no_contact_when_moving_fast():
    pts = {"left_hand": (220, 220, 0.9)}         # near the hold...
    prev = {"left_hand": (180, 200, 0.9)}        # ...but jumped ~44px -> reaching
    out = pe.frame_contacts(pts, prev, HOLDS, touch_distance_px=40, max_speed_px=12, min_confidence=0.5)
    assert out["left_hand"] is None


def test_no_contact_when_far():
    pts = {"left_hand": (500, 500, 0.9)}
    prev = {"left_hand": (500, 500, 0.9)}
    out = pe.frame_contacts(pts, prev, HOLDS, touch_distance_px=40, max_speed_px=12, min_confidence=0.5)
    assert out["left_hand"] is None


def test_no_contact_when_low_confidence():
    pts = {"left_hand": (220, 220, 0.2)}
    out = pe.frame_contacts(pts, None, HOLDS, touch_distance_px=40, max_speed_px=12, min_confidence=0.5)
    assert out["left_hand"] is None


def test_first_frame_not_gated_on_missing_prev():
    # prev_points None (first frame / after a gap) -> velocity gate is skipped.
    pts = {"left_hand": (220, 220, 0.9)}
    out = pe.frame_contacts(pts, None, HOLDS, touch_distance_px=40, max_speed_px=12, min_confidence=0.5)
    assert out["left_hand"] == 0
