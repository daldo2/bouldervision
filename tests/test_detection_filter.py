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


def test_tape_detected_by_aspect_ratio():
    # 200x20 strip on a 1000x1000 image -> aspect 10, well over the threshold.
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    assert df.classify_detection((100, 100, 300, 120), 0, img, FCFG) == df.TAPE


def test_volume_detected_by_class():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    # Small box, but the model says class == volume_class.
    assert df.classify_detection((10, 10, 60, 60), FCFG["volume_class"], img, FCFG) == df.VOLUME


def test_volume_detected_by_size():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    # 300x300 = 9% of the image, above volume_area_frac, even as a "hold" class.
    assert df.classify_detection((0, 0, 300, 300), 0, img, FCFG) == df.VOLUME


def test_marker_detected_when_small_dark_round():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    cv2.circle(img, (500, 500), 12, (10, 10, 10), -1)  # small black dot
    assert df.classify_detection((488, 488, 512, 512), 0, img, FCFG) == df.MARKER


def test_colored_hold_is_not_a_marker():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    cv2.circle(img, (500, 500), 12, (255, 0, 0), -1)  # small BLUE dot, same size
    # Same geometry as the marker, but it is colored -> stays a hold.
    assert df.classify_detection((488, 488, 512, 512), 0, img, FCFG) == df.HOLD


def test_normal_hold_passes_through():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    cv2.circle(img, (500, 500), 30, (0, 0, 255), -1)  # medium red hold
    assert df.classify_detection((460, 460, 540, 540), 0, img, FCFG) == df.HOLD


def test_classify_and_holds_only_split():
    img = np.full((1000, 1000, 3), 180, dtype=np.uint8)
    cv2.circle(img, (500, 500), 30, (0, 0, 255), -1)
    boxes = [
        (460, 460, 540, 540, 0.9, 0),     # red hold
        (0, 0, 300, 300, 0.8, 0),         # big -> volume
        (100, 100, 300, 120, 0.7, 0),     # strip -> tape
    ]
    dets = df.classify_detections(boxes, img, FCFG)
    kinds = sorted(d.kind for d in dets)
    assert kinds == sorted([df.HOLD, df.VOLUME, df.TAPE])
    assert len(df.holds_only(dets)) == 1
    assert df.holds_only(dets)[0] == (460, 460, 540, 540, 0.9)
