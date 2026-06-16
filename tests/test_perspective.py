"""
Tests for perspective rectification (wall homography).

Offline and model-free: pure OpenCV geometry on synthetic points.

Run with:  pytest tests/
"""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import perspective as persp  # noqa: E402
import route_extractor as re  # noqa: E402


def test_order_corners_canonicalizes_any_order():
    # Given out of order, should come back TL, TR, BR, BL.
    pts = [(100, 900), (100, 100), (1100, 900), (1100, 100)]  # BL, TL, BR, TR
    ordered = persp.order_corners(pts)
    assert np.allclose(ordered[0], (100, 100))    # TL
    assert np.allclose(ordered[1], (1100, 100))   # TR
    assert np.allclose(ordered[2], (1100, 900))   # BR
    assert np.allclose(ordered[3], (100, 900))    # BL


def test_homography_maps_corners_to_rectangle():
    corners = [(120, 80), (1400, 140), (1500, 1900), (60, 1850)]
    H, (w, h) = persp.compute_homography(corners)
    out = persp.warp_points(H, persp.order_corners(corners))
    expected = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    assert np.allclose(out, expected, atol=1e-3)


def test_rectify_holds_undoes_foreshortening_spacing():
    # A trapezoid wall: the top edge (far) is compressed, the bottom (near) wide.
    corners = [(400, 100), (1100, 100), (1500, 1900), (0, 1900)]
    H, _ = persp.compute_homography(corners)

    def box_at(cx, cy, s=20):
        return re.Hold(box=(cx - s, cy - s, cx + s, cy + s), confidence=1.0,
                       lab=np.array([128.0, 128.0, 128.0]))

    # Two holds near the compressed top vs two near the wide bottom, each pair
    # spanning the SAME fraction (0.3..0.7) of the wall's width at its depth.
    # Coordinates derived from the trapezoid edges at y=200 and y=1800.
    top = [box_at(601, 200), box_at(899, 200)]
    bottom = [box_at(459, 1800), box_at(1041, 1800)]
    persp.rectify_holds(top + bottom, H)

    def gap(a, b):
        return np.hypot(*(np.subtract(a.spatial_center, b.spatial_center)))

    # In the original image the top pair is much closer than the bottom pair;
    # after rectification their spacings should be nearly equal.
    raw_ratio = np.hypot(*(np.subtract(top[0].center, top[1].center))) / \
        np.hypot(*(np.subtract(bottom[0].center, bottom[1].center)))
    rect_ratio = gap(top[0], top[1]) / gap(bottom[0], bottom[1])
    assert raw_ratio < 0.6           # strongly compressed at the top originally
    assert rect_ratio > 0.9          # nearly equalized after un-warping


def test_homography_requires_four_points():
    with pytest.raises(ValueError):
        persp.compute_homography([(0, 0), (10, 0), (10, 10)])
