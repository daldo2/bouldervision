"""
Tests for Phase 3 pose helpers: hold-contact geometry and keypoint smoothing.

Offline and model-free — no YOLO pose model is loaded. We hand-craft keypoint
arrays and hold boxes and check the pure logic.

Run with:  pytest tests/
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import pose_estimator as pe  # noqa: E402


def make_pose(points):
    """Build a FramePose from {keypoint_index: (x, y, conf)}; rest default low-conf."""
    kp = np.zeros((17, 3), dtype=np.float64)
    for idx, (x, y, c) in points.items():
        kp[idx] = (x, y, c)
    return pe.FramePose(keypoints=kp)


# --- point/box distance ----------------------------------------------------
def test_point_inside_box_is_zero():
    assert pe.point_to_box_distance(50, 50, (0, 0, 100, 100)) == 0.0


def test_point_outside_box_edge_distance():
    # 10 px to the right of the box edge.
    assert pe.point_to_box_distance(110, 50, (0, 0, 100, 100)) == 10.0


def test_point_outside_box_corner_distance():
    d = pe.point_to_box_distance(103, 104, (0, 0, 100, 100))  # 3,4 -> 5
    assert abs(d - 5.0) < 1e-9


# --- touched_holds ---------------------------------------------------------
def test_hand_touches_nearby_hold():
    # left_wrist (idx 9) sits right on hold 1.
    pose = make_pose({9: (200, 200, 0.9)})
    holds = [(0, 0, 50, 50), (190, 190, 230, 230)]
    touched = pe.touched_holds(pose, holds, touch_distance_px=40, min_confidence=0.5)
    assert touched["left_hand"] == 1


def test_hand_too_far_is_no_contact():
    pose = make_pose({9: (500, 500, 0.9)})
    holds = [(0, 0, 50, 50)]
    touched = pe.touched_holds(pose, holds, touch_distance_px=40, min_confidence=0.5)
    assert touched["left_hand"] is None


def test_low_confidence_limb_is_ignored():
    # Right on the hold, but the joint confidence is below threshold.
    pose = make_pose({9: (10, 10, 0.2)})
    holds = [(0, 0, 50, 50)]
    touched = pe.touched_holds(pose, holds, touch_distance_px=40, min_confidence=0.5)
    assert touched["left_hand"] is None


def test_all_four_limbs_reported():
    pose = make_pose({9: (0, 0, 0.9), 10: (0, 0, 0.9), 15: (0, 0, 0.9), 16: (0, 0, 0.9)})
    touched = pe.touched_holds(pose, [(0, 0, 10, 10)], 40, 0.5)
    assert set(touched.keys()) == {"left_hand", "right_hand", "left_foot", "right_foot"}


# --- smoothing -------------------------------------------------------------
def test_smoothing_window_one_is_identity():
    poses = [make_pose({9: (100, 100, 0.9)}), make_pose({9: (200, 200, 0.9)})]
    out = pe.smooth_keypoint_sequence(poses, window=1)
    assert np.allclose(out[1].keypoints[9, :2], (200, 200))


def test_smoothing_reduces_jitter():
    # A spike at the middle frame should be pulled toward its neighbors.
    poses = [
        make_pose({9: (100, 100, 0.9)}),
        make_pose({9: (180, 180, 0.9)}),  # jittery spike
        make_pose({9: (100, 100, 0.9)}),
    ]
    out = pe.smooth_keypoint_sequence(poses, window=3)
    smoothed_x = out[1].keypoints[9, 0]
    assert smoothed_x < 180  # spike was pulled down toward 100
