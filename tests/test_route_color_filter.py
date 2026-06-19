"""
Test the video pipeline's single-route color filter (model-free).

Run with:  pytest tests/
"""

import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import utils  # noqa: E402
import video_pipeline as vp  # noqa: E402


def test_filter_by_color_keeps_only_matching_route():
    config = utils.load_config()
    img = np.full((300, 600, 3), 180, dtype=np.uint8)
    cv2.circle(img, (100, 150), 30, (0, 0, 255), -1)    # red
    # Real blue holds read at Lab hue ~-75 (the naming anchors are calibrated to
    # that); pure-blue [255,0,0] sits at -54, which is in violet territory for the
    # calibrated palette. Use a realistic blue so the test reflects real holds.
    cv2.circle(img, (300, 150), 30, (200, 90, 20), -1)  # blue
    cv2.circle(img, (500, 150), 30, (0, 255, 255), -1)  # yellow
    boxes = [(70, 120, 130, 180), (270, 120, 330, 180), (470, 120, 530, 180)]

    kept = vp._filter_by_color(img, boxes, "red", config)
    assert kept == [(70, 120, 130, 180)]               # only the red one survives
    assert vp._filter_by_color(img, boxes, "blue", config) == [(270, 120, 330, 180)]
    assert vp._filter_by_color(img, boxes, "green", config) == []  # no green route here
