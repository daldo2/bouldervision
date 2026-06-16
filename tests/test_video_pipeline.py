"""
Tests for Phase 3 video timeline aggregation (model-free, offline).

We build a synthetic per-frame timeline of limb->hold contacts and check the
summary helpers. The video loop itself needs models + footage and is not tested
here.

Run with:  pytest tests/
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import video_pipeline as vp  # noqa: E402


def frame(i, **contacts):
    return vp.FrameAnalysis(frame_index=i, contacts=contacts)


def test_summarize_counts_frames_per_hold():
    timeline = [
        frame(0, left_hand=2, right_hand=5),
        frame(1, left_hand=2, right_hand=5),
        frame(2, left_hand=3, right_hand=5),
    ]
    summary = vp.summarize_contacts(timeline)
    assert summary["left_hand"][2] == 2
    assert summary["left_hand"][3] == 1
    assert summary["right_hand"][5] == 3


def test_summarize_ignores_no_contact():
    timeline = [frame(0, left_hand=None), frame(1, left_hand=7)]
    summary = vp.summarize_contacts(timeline)
    assert summary["left_hand"][7] == 1
    assert None not in summary["left_hand"]


def test_summarize_empty_timeline():
    assert vp.summarize_contacts([]) == {}


def test_holds_used_collects_all_limbs():
    timeline = [
        frame(0, left_hand=1, right_foot=9),
        frame(1, right_hand=4),
    ]
    assert vp.holds_used(timeline) == {1, 9, 4}


def test_holds_used_min_frames_filters_brush_pasts():
    timeline = [
        frame(0, left_hand=1),
        frame(1, left_hand=1),
        frame(2, left_hand=8),   # touched only once — a fleeting brush
    ]
    assert vp.holds_used(timeline, min_frames=2) == {1}
