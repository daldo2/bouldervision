"""
Tests for detection post-processing (holds vs. volumes / markers / tape).

Offline and model-free: synthetic boxes + a small painted image, no YOLO.

Run with:  pytest tests/
"""

import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import detection_filter as df  # noqa: E402
import utils  # noqa: E402

FCFG = utils.load_config()["filter"]
# Heuristic-only config: drop the class_kinds map so the geometry rules run.
# (With class_kinds present the model's class id is trusted instead — see the
# dedicated test below.) The retrained holds.pt uses the trust path; the old
# 2-class best.pt and these shape heuristics use HEUR.
HEUR = {k: v for k, v in FCFG.items() if k != "class_kinds"}


def test_tape_detected_by_aspect_ratio():
    # 200x20 strip on a 1000x1000 image -> aspect 10, thin -> tape.
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    assert df.classify_detection((100, 100, 300, 120), 0, img, HEUR) == df.TAPE


def test_elongated_but_thick_shape_is_not_tape():
    # A tall, partly cut-off shape (158x653): elongated aspect but far too thick
    # to be tape. Big enough to read as a volume instead, but never tape.
    img = np.full((1500, 1500, 3), 180, dtype=np.uint8)
    assert df.classify_detection((0, 74, 158, 727), 0, img, HEUR) != df.TAPE


def test_volume_detected_by_class():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    # Small box, but the model says class == volume_class.
    assert df.classify_detection((10, 10, 60, 60), HEUR["volume_class"], img, HEUR) == df.VOLUME


def test_volume_detected_by_size():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    # 300x300 = 9% of the image, above volume_area_frac, even as a "hold" class.
    assert df.classify_detection((0, 0, 300, 300), 0, img, HEUR) == df.VOLUME


def test_marker_detected_when_small_dark_round():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    cv2.circle(img, (500, 500), 12, (10, 10, 10), -1)  # small black dot
    assert df.classify_detection((488, 488, 512, 512), 0, img, HEUR) == df.MARKER


def test_colored_hold_is_not_a_marker():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    cv2.circle(img, (500, 500), 12, (255, 0, 0), -1)  # small BLUE dot, same size
    # Same geometry as the marker, but it is colored -> stays a hold.
    assert df.classify_detection((488, 488, 512, 512), 0, img, HEUR) == df.HOLD


def test_small_dark_but_irregular_is_not_a_marker():
    # Small, dark, near-square box, but the dark blob is a triangle, not round.
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    cv2.fillPoly(img, [np.array([[500, 488], [512, 512], [488, 512]])], (10, 10, 10))
    # Fails the circularity gate -> stays a hold, not a marker.
    assert df.classify_detection((488, 488, 512, 512), 0, img, HEUR) == df.HOLD


def test_normal_hold_passes_through():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    cv2.circle(img, (500, 500), 30, (0, 0, 255), -1)  # medium red hold
    assert df.classify_detection((460, 460, 540, 540), 0, img, HEUR) == df.HOLD


def test_classify_and_holds_only_split():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    cv2.circle(img, (500, 500), 30, (0, 0, 255), -1)
    boxes = [
        (460, 460, 540, 540, 0.9, 0),     # red hold
        (0, 0, 300, 300, 0.8, 0),         # big -> volume
        (100, 100, 300, 120, 0.7, 0),     # strip -> tape
    ]
    dets = df.classify_detections(boxes, img, HEUR)
    kinds = sorted(d.kind for d in dets)
    assert kinds == sorted([df.HOLD, df.VOLUME, df.TAPE])
    assert len(df.holds_only(dets)) == 1
    assert df.holds_only(dets)[0] == (460, 460, 540, 540, 0.9)


def test_class_kinds_trusts_model_over_geometry():
    """With a class_kinds map (retrained holds.pt), the model's class id wins —
    geometry heuristics are skipped. The real config maps 0->downclimb, 1->hold,
    2->marker, 3->tape, 4->volume."""
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    assert "class_kinds" in FCFG  # the shipped config uses the trust path
    # A big box the size heuristic would call a volume, but the model says hold.
    assert df.classify_detection((0, 0, 300, 300), 1, img, FCFG) == df.HOLD
    # A small box the model labels downclimb (no geometry would ever say that).
    assert df.classify_detection((10, 10, 60, 60), 0, img, FCFG) == df.DOWNCLIMB
    assert df.classify_detection((10, 10, 60, 60), 4, img, FCFG) == df.VOLUME
    # downclimb is not a route hold -> set aside.
    dets = df.classify_detections([(0, 0, 50, 50, 0.9, 0)], img, FCFG)
    assert df.holds_only(dets) == []
