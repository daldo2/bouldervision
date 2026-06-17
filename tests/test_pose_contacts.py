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


# --- contact hysteresis -----------------------------------------------------
def test_smooth_bridges_short_dropout():
    # On hold 3, a 2-frame blip to None, back on 3 -> the gap is filled.
    seq = [3, 3, None, None, 3, 3]
    assert pe.smooth_contact_sequence(seq, max_gap=8, min_run=2) == [3, 3, 3, 3, 3, 3]


def test_smooth_does_not_bridge_different_holds():
    # 3 then None then 5 -> a real move between holds, not a dropout.
    seq = [3, 3, None, 5, 5]
    assert pe.smooth_contact_sequence(seq, max_gap=8, min_run=1) == [3, 3, None, 5, 5]


def test_smooth_does_not_bridge_too_long_a_gap():
    seq = [3, None, None, None, 3]
    assert pe.smooth_contact_sequence(seq, max_gap=2, min_run=1) == [3, None, None, None, 3]


def test_smooth_drops_flicker_runs():
    # A single-frame touch of hold 9 is dropped as flicker.
    seq = [None, None, 9, None, None]
    assert pe.smooth_contact_sequence(seq, max_gap=8, min_run=2) == [None, None, None, None, None]


def test_mode_smooth_locks_to_dominant_hold():
    # Foot mostly on 29 with jitter to 23 -> majority vote pins it to 29.
    seq = [29, 29, 23, 29, 23, 29, 29, 23, 29]
    out = pe.mode_smooth_contacts(seq, window=5)
    assert out == [29] * 9


def test_mode_smooth_keeps_a_real_switch():
    # A genuine, sustained switch from 1 to 2 survives the mode filter.
    seq = [1] * 6 + [2] * 6
    out = pe.mode_smooth_contacts(seq, window=5)
    assert out[0] == 1 and out[-1] == 2 and set(out) == {1, 2}


# --- sticky (stateful) contact resolution -----------------------------------
def _foot(x, y, c=0.9):
    return {"left_foot": (x, y, c), "left_hand": (0, 0, 0), "right_hand": (0, 0, 0),
            "right_foot": (0, 0, 0)}


def test_sticky_keeps_grip_through_fast_adjustment():
    # Foot on hold 0; one frame it jumps 30px (fast) but stays near hold 0 -> stays.
    box = [(200, 200, 260, 260)]
    seq = [_foot(230, 230), _foot(231, 230), _foot(258, 235), _foot(232, 230)]
    boxes = [box] * len(seq)
    out = pe.resolve_contact_sequence(seq, boxes, touch_distance_px=30,
                                      release_distance_px=70, max_speed_px=12, min_confidence=0.5)
    assert [f["left_foot"] for f in out] == [0, 0, 0, 0]


def test_sticky_releases_then_engages_new_hold_when_settled():
    box0 = (100, 100, 140, 140)
    box1 = (400, 100, 440, 140)
    boxes = [[box0, box1]] * 5
    # Start on hold 0, then a big leap far from both (releases), then settle on hold 1.
    seq = [_foot(120, 120), _foot(120, 120), _foot(260, 110), _foot(420, 120), _foot(421, 120)]
    out = pe.resolve_contact_sequence(seq, boxes, touch_distance_px=30,
                                      release_distance_px=70, max_speed_px=12, min_confidence=0.5)
    got = [f["left_foot"] for f in out]
    assert got[0] == 0 and got[2] is None and got[-1] == 1


def test_lr_consistency_unswaps_flipped_labels():
    # Left foot near x=100, right foot near x=300, steady across frames — but
    # frame 1 has the labels swapped. Consistency should restore them.
    def pose(lank, rank):
        kp = np.zeros((17, 3), dtype=float)
        kp[13] = (lank[0], lank[1] - 50, 0.9); kp[15] = (*lank, 0.9)  # left knee/ankle
        kp[14] = (rank[0], rank[1] - 50, 0.9); kp[16] = (*rank, 0.9)  # right knee/ankle
        return pe.FramePose(keypoints=kp)
    seq = [pose((100, 400), (300, 400)),
           pose((300, 400), (100, 400)),   # swapped!
           pose((100, 400), (300, 400))]
    fixed = pe.enforce_lr_consistency(seq)
    # after fixing, the left ankle stays ~x=100 every frame
    assert all(abs(p.keypoints[15][0] - 100) < 1 for p in fixed)
    assert all(abs(p.keypoints[16][0] - 300) < 1 for p in fixed)


def test_sticky_no_engage_while_reaching_fast_past_hold():
    box = [(200, 200, 260, 260)]
    # Not gripping; arrives near the hold but moving fast (reaching past) -> no grip.
    seq = [_foot(120, 200), _foot(232, 230)]  # ~112px jump
    boxes = [box] * len(seq)
    out = pe.resolve_contact_sequence(seq, boxes, touch_distance_px=30,
                                      release_distance_px=70, max_speed_px=12, min_confidence=0.5)
    assert out[1]["left_foot"] is None
