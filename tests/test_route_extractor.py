"""
Tests for Phase 2 route extraction (color clustering + spatial split).

Offline and model-free: we build synthetic Holds with known Lab colors and
positions, plus a small synthetic image for the detector->holds bridge.

Run with:  pytest tests/
"""

import os
import sys

import cv2
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import route_extractor as re  # noqa: E402
import utils  # noqa: E402


def lab_of(bgr):
    """Lab vector for a solid BGR color (OpenCV 8-bit scale)."""
    px = np.uint8([[list(bgr)]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)


def hold(bgr, center, conf=0.9):
    """Make a Hold of the given color centered at (x, y)."""
    cx, cy = center
    box = (cx - 10, cy - 10, cx + 10, cy + 10)
    return re.Hold(box=box, confidence=conf, lab=lab_of(bgr))


RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


# --- color clustering ------------------------------------------------------
def test_cluster_by_color_separates_distinct_colors():
    holds = [hold(RED, (50, 50)), hold(BLUE, (60, 60)), hold(GREEN, (70, 70))]
    groups = re.cluster_by_color(holds, eps=28)
    assert len(groups) == 3  # red, blue, green are all far apart in Lab


def test_cluster_by_color_merges_same_color():
    holds = [hold(RED, (10, 10)), hold(RED, (20, 20)), hold(RED, (30, 30))]
    groups = re.cluster_by_color(holds, eps=28)
    assert len(groups) == 1
    assert len(groups[0]) == 3


# --- spatial split ---------------------------------------------------------
def test_split_by_position_separates_far_clusters():
    # Same color, two groups far apart on the wall.
    left = [hold(RED, (100, 100)), hold(RED, (120, 110))]
    right = [hold(RED, (900, 100)), hold(RED, (920, 110))]
    clusters = re.split_by_position(left + right, eps_px=200)
    assert len(clusters) == 2


def test_split_by_position_keeps_near_holds_together():
    holds = [hold(RED, (100, 100)), hold(RED, (150, 130)), hold(RED, (130, 160))]
    clusters = re.split_by_position(holds, eps_px=200)
    assert len(clusters) == 1


def test_split_by_position_drops_outlier_when_min_holds_raised():
    holds = [hold(RED, (100, 100)), hold(RED, (130, 120)), hold(RED, (2000, 2000))]
    clusters = re.split_by_position(holds, eps_px=150, min_holds=2)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2  # the lone far hold is dropped as noise


# --- end to end ------------------------------------------------------------
def test_extract_routes_two_red_one_blue():
    # Two separate red problems + one blue problem.
    holds = [
        hold(RED, (100, 100)), hold(RED, (130, 130)),     # red route A (left)
        hold(RED, (900, 100)), hold(RED, (930, 130)),     # red route B (right)
        hold(BLUE, (500, 500)), hold(BLUE, (530, 520)),   # blue route
    ]
    routes = re.extract_routes(holds, color_eps=28, spatial_eps_px=200)
    assert len(routes) == 3
    # All routes have 2 holds; sorted output is stable.
    assert all(r.hold_count == 2 for r in routes)


def test_extract_routes_names_from_references():
    config = utils.load_config()
    refs = utils.reference_labs(config["draw_colors"])
    holds = [hold(BLUE, (100, 100)), hold(BLUE, (140, 120))]
    routes = re.extract_routes(holds, references=refs)
    assert len(routes) == 1
    assert routes[0].color_name == "blue"
    assert routes[0].holds[0].name == "blue"


def test_extract_routes_empty():
    assert re.extract_routes([]) == []


# --- detector -> holds bridge ---------------------------------------------
def test_build_holds_reads_colors_from_image():
    # Gray wall with a red blob on the left and a blue blob on the right.
    img = np.full((300, 600, 3), 180, dtype=np.uint8)
    cv2.circle(img, (100, 150), 30, RED, -1)
    cv2.circle(img, (500, 150), 30, BLUE, -1)
    boxes = [(70, 120, 130, 180, 0.9), (470, 120, 530, 180, 0.8)]

    holds = re.build_holds(img, boxes)
    assert len(holds) == 2

    config = utils.load_config()
    refs = utils.reference_labs(config["draw_colors"])
    names = sorted(utils.nearest_color_name(h.lab, refs) for h in holds)
    assert names == ["blue", "red"]


def test_build_holds_skips_empty_crop():
    img = np.full((100, 100, 3), 180, dtype=np.uint8)
    # Degenerate box (zero area) should be skipped, not crash.
    boxes = [(50, 50, 50, 50, 0.5)]
    assert re.build_holds(img, boxes) == []


# --- ordering & drawing ----------------------------------------------------
def test_ordered_holds_bottom_to_top():
    top = hold(RED, (100, 50))      # small y = high on the wall
    mid = hold(RED, (100, 250))
    bottom = hold(RED, (100, 450))  # large y = low on the wall = start
    route = re.Route(color_name="red", lab=lab_of(RED), holds=[top, bottom, mid])
    ys = [h.center[1] for h in route.ordered_holds]
    assert ys == [450, 250, 50]  # start at the bottom, climb upward


def test_draw_routes_does_not_crash():
    img = np.full((500, 500, 3), 180, dtype=np.uint8)
    holds = [hold(RED, (100, 400)), hold(RED, (150, 300)), hold(BLUE, (400, 400))]
    routes = re.extract_routes(holds, color_eps=28, spatial_eps_px=200)
    out = utils.draw_routes(img, routes)
    assert out.shape == img.shape
    assert out is not img  # drew on a copy
