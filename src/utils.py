"""
utils.py — shared helpers for BoulderVision.

Three groups of helpers live here so the rest of the codebase stays focused:

  1. Config loading          (read settings.yaml once, reuse everywhere)
  2. Color classification     (turn an image crop into a named color)
  3. Drawing                  (put labeled boxes on an image)

Everything is plain functions — no classes — to keep it approachable.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Project root = the folder one level above this file's `src/` directory.
# We compute it so paths work no matter what directory you run scripts from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "settings.yaml")


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------
def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load settings.yaml into a plain dict.

    Why a function instead of just importing a global: it makes the config
    location explicit and testable, and lets callers pass a custom path.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> str:
    """Turn a project-relative path (from config) into an absolute path.

    Config stores paths like "data/output/output.jpg" relative to the project
    root; this makes them absolute so they resolve regardless of the current
    working directory.
    """
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)


# ---------------------------------------------------------------------------
# 2. Color classification (HSV)
# ---------------------------------------------------------------------------
def classify_color(
    crop_bgr: np.ndarray,
    color_ranges: Dict[str, List[dict]],
) -> Tuple[str, float]:
    """Decide the dominant named color of an image crop.

    HOW IT WORKS:
      - We convert the crop from BGR (OpenCV's default) to HSV. HSV separates
        *hue* (which color) from *brightness*, which makes color matching far
        more robust to lighting than raw RGB/BGR.
      - For each candidate color we build a binary mask of pixels that fall
        inside that color's HSV range(s), then count how many pixels matched.
      - The color with the most matching pixels wins. We also return the
        fraction of the crop it covered, as a rough confidence.

    Args:
        crop_bgr: an (H, W, 3) BGR image region (one detected hold).
        color_ranges: the `colors:` block from settings.yaml.

    Returns:
        (color_name, coverage_fraction). If nothing matches well, returns
        ("unknown", 0.0).
    """
    # Guard against empty crops (can happen with degenerate boxes).
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0

    # Convert once; every color check reuses this HSV version of the crop.
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    best_color = "unknown"
    best_count = 0

    # Check each color. A color can have several ranges (e.g. red wraps the
    # hue circle at 0/179), so we OR their masks together.
    for color_name, ranges in color_ranges.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            lower = np.array(r["lower"], dtype=np.uint8)
            upper = np.array(r["upper"], dtype=np.uint8)
            # inRange marks pixels inside [lower, upper] as 255, else 0.
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = cv2.bitwise_or(mask_total, mask)

        count = int(cv2.countNonZero(mask_total))
        if count > best_count:
            best_count = count
            best_color = color_name

    # Coverage = what fraction of the crop the winning color filled.
    coverage = best_count / total_pixels if total_pixels else 0.0

    # If even the best color barely showed up, we don't trust it.
    if coverage < 0.10:
        return "unknown", coverage

    return best_color, coverage


# ---------------------------------------------------------------------------
# 3. Drawing
# ---------------------------------------------------------------------------
def draw_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    bgr_color: Tuple[int, int, int],
) -> None:
    """Draw one labeled bounding box onto `image` (modifies it in place).

    `box` is (x1, y1, x2, y2) in pixel coordinates.
    """
    x1, y1, x2, y2 = box

    # The rectangle around the hold.
    cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color, thickness=2)

    # Draw a filled label background so text is readable over any wall color.
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        image,
        (x1, y1 - text_h - baseline - 4),
        (x1 + text_w + 4, y1),
        bgr_color,
        thickness=-1,  # -1 = filled
    )
    # The label text, drawn in white on top of the colored background.
    cv2.putText(
        image,
        label,
        (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def draw_color_for(color_name: str, draw_colors: Dict[str, list]) -> Tuple[int, int, int]:
    """Look up the BGR drawing color for a color name, falling back to gray."""
    bgr = draw_colors.get(color_name, draw_colors.get("unknown", [200, 200, 200]))
    return tuple(int(c) for c in bgr)


def read_image(path: str) -> np.ndarray:
    """Load an image from disk, raising a clear error if it's missing/unreadable."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image at '{path}'. Check the path and file format."
        )
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """Write an image to disk, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, image)


# ---------------------------------------------------------------------------
# 4. Color vectors for route clustering (Phase 2)
# ---------------------------------------------------------------------------
# The HSV name classifier above maps a hold to a fixed name (red/blue/...). That
# is great for a human-readable label, but fixed names don't generalize across
# gyms (different lighting, white balance, palettes). For grouping holds into
# routes we instead work with each hold's *actual* color as a vector in CIELAB —
# a perceptually uniform space where Euclidean distance ≈ how different two
# colors look to the eye — and cluster those vectors PER IMAGE (see
# route_extractor.py). The helpers below produce those vectors.
#
# Note: OpenCV's 8-bit Lab packs L into 0-255 and a,b into 0-255 (128 = neutral).
# We keep that scale throughout; clustering thresholds are tuned to match it.

def white_balance(image: np.ndarray) -> np.ndarray:
    """Gray-world white balance: cancel a global color cast from lighting/camera.

    The gray-world assumption says the average color of a varied scene should be
    gray; if it isn't, we scale each channel to make it so. This removes much of
    the gym-to-gym lighting difference BEFORE we read hold colors, which is what
    lets the same color land in the same place across different photos.
    """
    result = image.astype(np.float32)
    channel_means = result.reshape(-1, 3).mean(axis=0)  # B, G, R means
    gray = float(channel_means.mean())
    scale = gray / (channel_means + 1e-6)
    result *= scale
    return np.clip(result, 0, 255).astype(np.uint8)


def center_inset(crop_bgr: np.ndarray, frac: float = 0.6) -> np.ndarray:
    """Return the central `frac` of a crop (both axes), keeping it non-empty.

    Detector boxes are axis-aligned, so a box around a round/irregular hold
    necessarily includes wall, shadow, and mat in its corners. Reading color
    over the whole box lets that background dominate — which is how a saturated
    blue volume ends up sampled as a dark neutral and named "black". Sampling
    only the box center keeps us on the hold itself. For tiny crops where the
    inset would vanish, we return the crop unchanged.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    iy, ix = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    inner = crop_bgr[iy:h - iy, ix:w - ix]
    return inner if inner.size else crop_bgr


def hold_foreground_mask(crop_bgr: np.ndarray, inset: float = 0.08, iters: int = 3):
    """GrabCut foreground (the hold) vs background (wall/pocket) for a box crop.

    The axis-aligned box around a hold includes wall in its corners and, for a
    pocketed hold, a dark hole in the middle — both pollute the color. GrabCut,
    seeded with the box border as probable-background, separates the hold itself
    so we sample color only off the hold. Returns a bool HxW mask, or None when
    the crop is too small or the result is degenerate (caller then falls back).
    """
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return None
    m = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    mx, my = int(w * inset) + 1, int(h * inset) + 1
    rect = (mx, my, max(1, w - 2 * mx), max(1, h - 2 * my))
    try:
        cv2.grabCut(crop_bgr, m, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = (m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)
    return fg if fg.sum() >= 0.05 * fg.size else None


def select_region_pixels(
    crop_bgr: np.ndarray,
    sat_min: int = 45,
    val_min: int = 40,
    val_max: int = 235,
    min_fraction: float = 0.05,
    center_frac: float = 0.6,
    use_mask: bool = False,
) -> np.ndarray:
    """The BGR pixels we trust as "the hold" color, for a box crop.

    Shared selection logic behind both the Lab-median naming and the w2c
    color-name naming, so they see *identical* pixels (a fair comparison):
      - Region: a GrabCut foreground mask (excludes wall + pocket) when
        `use_mask`, else the box CENTER (`center_frac`) as a coarse stand-in —
        the box corners are mostly wall/shadow that drags color toward neutral.
      - We prefer *chromatic* pixels: saturated enough (`sat_min`) and neither
        crushed-dark nor blown-out (`val_min`..`val_max`) — skips chalk dust,
        deep shadow, specular highlights. If too few chromatic pixels (an
        achromatic hold), fall back to in-range region pixels, then the region,
        then everything.

    Returns an (N, 3) uint8 BGR array; empty (0, 3) for an empty crop.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.empty((0, 3), np.uint8)

    region = hold_foreground_mask(crop_bgr) if use_mask else None
    if region is None:
        crop_bgr = center_inset(crop_bgr, center_frac)
        region = np.ones(crop_bgr.shape[:2], dtype=bool)

    bgr = crop_bgr.reshape(-1, 3)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat, val = hsv[:, 1], hsv[:, 2]
    reg = region.reshape(-1)

    chromatic = reg & (sat >= sat_min) & (val >= val_min) & (val <= val_max)
    if int(chromatic.sum()) >= max(1, int(min_fraction * int(reg.sum()))):
        return bgr[chromatic]
    valid = reg & (val >= val_min) & (val <= val_max)
    if int(valid.sum()):
        return bgr[valid]
    return bgr[reg] if int(reg.sum()) else bgr


def dominant_color_lab(
    crop_bgr: np.ndarray,
    sat_min: int = 45,
    val_min: int = 40,
    val_max: int = 235,
    min_fraction: float = 0.05,
    center_frac: float = 0.6,
    use_mask: bool = False,
) -> Optional[np.ndarray]:
    """Return a hold crop's dominant color as a Lab vector (OpenCV 8-bit scale).

    Takes the MEDIAN Lab over `select_region_pixels` (chromatic-preferred hold
    pixels), so a few stray pixels (a bolt, a chalk smear) don't drag the result
    and an achromatic hold still separates black (low L) from white (high L).

    Returns None for an empty crop.
    """
    px = select_region_pixels(crop_bgr, sat_min, val_min, val_max,
                              min_fraction, center_frac, use_mask)
    if px.size == 0:
        return None
    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return np.median(lab, axis=0).astype(np.float64)


def lab_to_bgr(lab: np.ndarray) -> Tuple[int, int, int]:
    """Convert an OpenCV 8-bit Lab vector back to a BGR tuple (for drawing)."""
    px = np.clip(np.round(lab), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    bgr = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_routes(image: np.ndarray, routes) -> np.ndarray:
    """Draw each route on a copy of `image`: holds in the route's own color,
    connected bottom-to-top, with a number + color name label at the start hold.

    `routes` is a list of route_extractor.Route. Returns the annotated copy.
    """
    annotated = image.copy()

    for idx, route in enumerate(routes, start=1):
        bgr = lab_to_bgr(route.lab)            # draw the route in its real color
        ordered = route.ordered_holds          # bottom-to-top climbing order

        # Connect consecutive holds to suggest the sequence.
        pts = [(int(h.center[0]), int(h.center[1])) for h in ordered]
        for a, b in zip(pts, pts[1:]):
            cv2.line(annotated, a, b, bgr, thickness=2, lineType=cv2.LINE_AA)

        # Outline every hold in the route's color.
        for hold in ordered:
            x1, y1, x2, y2 = hold.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, thickness=2)

        # Label the route at its start (lowest) hold.
        if pts:
            label = f"#{idx} {route.color_name}".strip()
            draw_box(annotated, ordered[0].box, label, bgr)

    return annotated


def reference_labs(
    draw_colors: Dict[str, list], hue_anchors: Optional[Dict[str, float]] = None
) -> Dict[str, np.ndarray]:
    """Build name -> Lab reference points from the draw_colors palette.

    Used only to attach a human-readable name to a discovered color cluster; it
    does NOT drive the grouping itself.

    The on-screen draw colors are not calibrated to how real holds actually read
    in Lab — e.g. pure-blue [255,0,0] sits at hue ~-54 deg, but real blue holds
    cluster near -80, so the draw-blue anchor drifts into purple territory and
    steals violet holds. `hue_anchors` (name -> hue in degrees, from measuring
    real holds) rotates just those references onto the real cluster hue, keeping
    their chroma and lightness, so naming improves without touching the overlay.
    """
    refs: Dict[str, np.ndarray] = {}
    for name, bgr in draw_colors.items():
        if name == "unknown":
            continue
        px = np.uint8([[list(bgr)]])
        lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)
        if hue_anchors and name in hue_anchors:
            a, b = lab[1] - 128.0, lab[2] - 128.0
            chroma = float(np.hypot(a, b))
            rad = np.radians(float(hue_anchors[name]))
            lab[1] = 128.0 + chroma * np.cos(rad)
            lab[2] = 128.0 + chroma * np.sin(rad)
        refs[name] = lab
    return refs


def _chroma_hue(lab: np.ndarray) -> Tuple[float, float]:
    """Chroma (distance from neutral) and hue angle (radians) of an 8-bit Lab.

    OpenCV packs a,b around 128 = neutral. Chroma = how colorful (0 for gray);
    hue = the color's angle on the a*/b* wheel, independent of lightness.
    """
    a, b = float(lab[1]) - 128.0, float(lab[2]) - 128.0
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


def nearest_color_name(
    lab: np.ndarray,
    refs: Dict[str, np.ndarray],
    chroma_min: float = 12.0,
    rescue: Optional[dict] = None,
) -> str:
    """Name a Lab color, matching hue for colored holds and lightness for grays.

    Plain Euclidean distance in Lab conflates lightness, saturation and hue, so
    a muted real-world blue (dark, low-chroma) lands nearer pure *black* than
    pure *blue*. Instead we split the decision:

      - A color is "neutral" if its chroma is below `chroma_min`. Neutral holds
        are named by the nearest neutral reference in lightness (L) only — this
        is what separates black from white.
      - A "colored" hold is named by the nearest reference in *hue angle*, which
        ignores how dark or washed-out the photo made it. Only chromatic
        references compete here, so a dim blue still reads as blue.

    `rescue` (optional) recovers washed-out but hue-consistent holds that would
    otherwise neutralize to grey — e.g. this gym's faded turquoise, which reads
    chroma ~7-10 (below `chroma_min`) yet clusters tightly around the cyan hue.
    A hold with chroma in [rescue["chroma_min"], chroma_min) whose hue is within
    `rescue["hue_tol"]` degrees of one of the eligible `rescue["colors"]` anchors
    is named that color instead of neutral. Only colors with a *tight* real-world
    hue cluster belong here; true greys (chroma <~5, random hue) stay neutral.

    Falls back to plain Lab distance if the palette has no usable references.
    '' if there are no references at all.
    """
    if not refs:
        return ""

    target_chroma, target_hue = _chroma_hue(lab)
    neutral_refs, colored_refs = {}, {}
    for name, ref_lab in refs.items():
        chroma, hue = _chroma_hue(ref_lab)
        (colored_refs if chroma >= chroma_min else neutral_refs)[name] = (ref_lab, hue)

    def hue_gap(name: str) -> float:
        d = abs(target_hue - colored_refs[name][1])
        return min(d, 2 * np.pi - d)  # wrap-around on the color wheel

    if target_chroma < chroma_min and neutral_refs:
        if rescue and colored_refs and target_chroma >= rescue.get("chroma_min", chroma_min):
            eligible = [n for n in rescue.get("colors", []) if n in colored_refs]
            if eligible:
                best = min(eligible, key=hue_gap)
                if np.degrees(hue_gap(best)) <= rescue.get("hue_tol", 0.0):
                    return best
        return min(neutral_refs, key=lambda n: abs(lab[0] - neutral_refs[n][0][0]))

    if colored_refs:
        return min(colored_refs, key=hue_gap)

    # No reference of the needed kind: fall back to nearest full-Lab point.
    return min(refs, key=lambda n: float(np.linalg.norm(lab - refs[n])))
