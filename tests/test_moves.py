"""
Tests for move-sequence ("beta") extraction from a contact timeline.

Offline, model-free.

Run with:  pytest tests/
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import video_pipeline as vp  # noqa: E402


def tl(seq):
    """Build a timeline from a list of contact dicts."""
    return [vp.FrameAnalysis(frame_index=i, contacts=c) for i, c in enumerate(seq)]


def hold(**kw):
    base = {"left_hand": None, "right_hand": None, "left_foot": None, "right_foot": None}
    base.update(kw)
    return base


def tlp(rows):
    """Timeline from (contacts, points) rows."""
    return [vp.FrameAnalysis(frame_index=i, contacts=c, points=p) for i, (c, p) in enumerate(rows)]


def test_first_grip_is_a_start_move():
    timeline = tl([hold(left_hand=5)] * 6)
    moves = vp.extract_moves(timeline, fps=30, min_hold_frames=3)
    assert len(moves) == 1
    assert moves[0].limb == "left_hand" and moves[0].hold == 5 and moves[0].start


def test_moving_to_new_hold_is_a_move():
    timeline = tl([hold(left_hand=5)] * 5 + [hold(left_hand=3)] * 5)
    moves = vp.extract_moves(timeline, fps=30, min_hold_frames=3)
    holds = [(m.limb, m.hold, m.start) for m in moves]
    assert holds == [("left_hand", 5, True), ("left_hand", 3, False)]


def test_release_then_regrip_same_hold_is_not_a_move():
    # On 5, let go for a few frames, back on 5 -> still just the one start move.
    timeline = tl([hold(left_hand=5)] * 4 + [hold()] * 3 + [hold(left_hand=5)] * 4)
    moves = vp.extract_moves(timeline, fps=30, min_hold_frames=3)
    assert [(m.limb, m.hold) for m in moves] == [("left_hand", 5)]


def test_brief_touch_below_threshold_is_ignored():
    # Touches hold 8 for only 2 frames -> not a move at min_hold_frames=3.
    timeline = tl([hold(left_hand=5)] * 5 + [hold(left_hand=8)] * 2 + [hold(left_hand=5)] * 5)
    moves = vp.extract_moves(timeline, fps=30, min_hold_frames=3)
    assert [(m.limb, m.hold) for m in moves] == [("left_hand", 5)]


def test_stationary_hold_flip_is_not_a_move():
    # Right hand sits at ~(200,200), nearest-hold oscillates 0<->2 but it never
    # travels -> no phantom moves beyond the initial settle.
    pt = {"right_hand": (200, 200, 0.9)}
    rows = ([(hold(right_hand=0), pt)] * 5 + [(hold(right_hand=2), pt)] * 5
            + [(hold(right_hand=0), pt)] * 5)
    moves = vp.extract_moves(tlp(rows), fps=30, min_hold_frames=3, min_move_px=45)
    assert [(m.hold, m.start) for m in moves] == [(0, True)]  # only the start


def test_real_travel_counts_as_a_move():
    # Hand actually moves ~150px to a new hold -> a move.
    rows = ([(hold(right_hand=0), {"right_hand": (200, 200, 0.9)})] * 5
            + [(hold(right_hand=2), {"right_hand": (350, 200, 0.9)})] * 5)
    moves = vp.extract_moves(tlp(rows), fps=30, min_hold_frames=3, min_move_px=45)
    assert [(m.hold, m.start) for m in moves] == [(0, True), (2, False)]


def test_moves_are_time_ordered_across_limbs():
    timeline = tl(
        [hold(left_hand=1, right_hand=2)] * 5    # both establish at t0
        + [hold(left_hand=1, right_hand=4)] * 5  # RH moves to 4
        + [hold(left_hand=7, right_hand=4)] * 5  # LH moves to 7
    )
    moves = vp.extract_moves(timeline, fps=30, min_hold_frames=3)
    non_start = [(m.limb, m.hold) for m in moves if not m.start]
    assert non_start == [("right_hand", 4), ("left_hand", 7)]
