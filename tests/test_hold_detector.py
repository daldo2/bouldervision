"""
Tests for the hold detection + color classification pipeline.

These tests focus on the parts that don't require downloading a model or a GPU:
the HSV color classifier and the config. They run fast and offline.

Run with:  pytest tests/
"""

import os
import sys

import numpy as np
import pytest

# Make the src/ modules importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import utils  # noqa: E402


@pytest.fixture
def config():
    return utils.load_config()


def solid_color_crop(bgr, size=50):
    """Create a `size`x`size` image filled with one BGR color."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def test_config_has_required_sections(config):
    """settings.yaml should expose the blocks the pipeline depends on."""
    for key in ("models", "detection", "colors", "draw_colors", "paths"):
        assert key in config, f"missing config section: {key}"


def test_classify_pure_red(config):
    # BGR for pure red is (0, 0, 255).
    crop = solid_color_crop((0, 0, 255))
    color, coverage = utils.classify_color(crop, config["colors"])
    assert color == "red"
    assert coverage > 0.8  # nearly the whole crop matched


def test_classify_pure_blue(config):
    crop = solid_color_crop((255, 0, 0))  # BGR blue
    color, _ = utils.classify_color(crop, config["colors"])
    assert color == "blue"


def test_classify_pure_green(config):
    crop = solid_color_crop((0, 255, 0))  # BGR green
    color, _ = utils.classify_color(crop, config["colors"])
    assert color == "green"


def test_classify_black(config):
    crop = solid_color_crop((0, 0, 0))  # black
    color, _ = utils.classify_color(crop, config["colors"])
    assert color == "black"


def test_empty_crop_is_unknown(config):
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    color, coverage = utils.classify_color(empty, config["colors"])
    assert color == "unknown"
    assert coverage == 0.0


def test_draw_color_falls_back_to_unknown(config):
    bgr = utils.draw_color_for("not_a_color", config["draw_colors"])
    assert bgr == tuple(config["draw_colors"]["unknown"])


def test_draw_box_does_not_crash(config):
    img = solid_color_crop((128, 128, 128), size=100)
    # Should modify the image in place without raising.
    utils.draw_box(img, (10, 10, 60, 60), "red 0.90", (0, 0, 255))
