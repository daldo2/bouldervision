"""
route_extractor.py — Phase 2: group detected holds into routes (problems).

THE IDEA (why this generalizes across gyms):
    A route = "same color = same problem" *on one wall*. So we don't try to match
    each hold to a fixed global palette (that breaks under different lighting and
    gym color schemes). Instead we read each hold's ACTUAL color as a vector in
    CIELAB and CLUSTER those vectors per image. Because lighting shifts every
    hold in a photo roughly the same way, the *relative* grouping survives even
    when absolute values drift. Fixed color names are used only to label the
    clusters for humans, never to form them.

PIPELINE:
    holds (box + Lab color)
      ──▶ cluster by color   (DBSCAN in Lab space → "these holds share a color")
      ──▶ split by position  (DBSCAN on centroids → separate two same-color
                              problems on different wall sections)
      ──▶ Route objects, each named by its nearest reference color

Everything here is detector-agnostic: give it `Hold`s and it works, whether the
boxes came from the fine-tuned model or anywhere else. That is why it is fully
unit-tested without loading any model.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

# Make `import utils` work whether run as a module or a loose script (same trick
# hold_detector.py uses). utils only needs cv2/numpy/yaml — no heavy ML deps.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Hold:
    """One detected hold plus its dominant color as a Lab vector."""
    box: Tuple[int, int, int, int]      # (x1, y1, x2, y2) in pixels
    confidence: float
    lab: np.ndarray                     # dominant color, OpenCV 8-bit Lab
    name: str = ""                      # display color name (filled in later)
    rect_box: Optional[Tuple[float, float, float, float]] = None  # box in rectified (frontal) coords

    @property
    def center(self) -> Tuple[float, float]:
        """Centroid of the bounding box (original image pixels)."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def spatial_box(self) -> Tuple[float, float, float, float]:
        """Box used for spatial reasoning — rectified if a homography was applied."""
        return self.rect_box if self.rect_box is not None else self.box

    @property
    def spatial_center(self) -> Tuple[float, float]:
        """Centroid used for spatial clustering (rectified frontal plane if set)."""
        x1, y1, x2, y2 = self.spatial_box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def scale(self) -> float:
        """Mean box side — a proxy for distance/foreshortening.

        On an angled photo a hold far down the wall projects smaller, so its
        scale is a local yardstick: measuring gaps between holds in units of
        scale (rather than raw pixels) keeps "same route" consistent whether
        the holds are near the camera or foreshortened far away. Uses the
        rectified box when a homography has un-warped the wall.
        """
        x1, y1, x2, y2 = self.spatial_box
        return max(1.0, ((x2 - x1) + (y2 - y1)) / 2.0)


@dataclass
class Route:
    """A set of holds believed to form a single climbing problem."""
    color_name: str                     # nearest reference color, for display
    lab: np.ndarray                     # representative color of the route
    holds: List[Hold] = field(default_factory=list)

    @property
    def hold_count(self) -> int:
        return len(self.holds)

    @property
    def ordered_holds(self) -> List[Hold]:
        """Holds bottom-to-top — a first guess at the climbing sequence.

        Image y grows downward, so the lowest hold (the start) has the largest y.
        We sort by y descending; ties broken left-to-right for stability.
        """
        return sorted(self.holds, key=lambda h: (-h.center[1], h.center[0]))


# ---------------------------------------------------------------------------
# Clustering steps
# ---------------------------------------------------------------------------
def cluster_by_color(holds: List[Hold], eps: float, min_holds: int = 1) -> List[List[Hold]]:
    """Group holds whose Lab colors are within `eps` of each other (DBSCAN).

    `eps` is a distance in OpenCV's Lab scale: smaller = stricter (splits close
    shades), larger = looser (merges them). `min_holds` is DBSCAN's min_samples;
    keep it at 1 so a lone hold of a color still forms its own group here — the
    spatial step decides whether a group is route-worthy.
    """
    if not holds:
        return []

    features = np.array([h.lab for h in holds], dtype=np.float64)
    labels = DBSCAN(eps=eps, min_samples=min_holds, metric="euclidean").fit_predict(features)

    groups: Dict[int, List[Hold]] = {}
    for hold, label in zip(holds, labels):
        groups.setdefault(int(label), []).append(hold)
    # With min_samples=1 there is no noise (-1); if a caller raises it, drop noise.
    return [g for label, g in sorted(groups.items()) if label != -1]


def cluster_by_name(
    holds: List[Hold], references: Dict[str, np.ndarray], chroma_min: float = 12.0,
    rescue: Optional[dict] = None,
) -> List[List[Hold]]:
    """Group holds by their discrete nearest color NAME (red/blue/black/...).

    An alternative to cluster_by_color for DENSE walls. DBSCAN on raw Lab chains
    many holds into a single cluster through near-neutral ones — e.g. ~180 holds
    collapse into one "black" route. Bucketing by the already-tuned color name
    avoids that chaining; the spatial step still separates two same-named routes
    set apart on the wall.
    """
    groups: Dict[str, List[Hold]] = {}
    for h in holds:
        name = utils.nearest_color_name(h.lab, references, chroma_min, rescue)
        groups.setdefault(name, []).append(h)
    return [groups[k] for k in sorted(groups)]


def split_by_position(holds: List[Hold], eps_px: float, min_holds: int = 1) -> List[List[Hold]]:
    """Split holds that share a color into spatially-coherent clusters.

    Two different problems can reuse the same color on different parts of the
    wall. DBSCAN on hold centroids separates them: `eps_px` is how close (in
    pixels) holds must be to count as the same route, `min_holds` drops isolated
    outliers as noise (label -1) when set above 1.
    """
    if not holds:
        return []
    if len(holds) == 1:
        return [holds] if min_holds <= 1 else []

    centers = np.array([h.spatial_center for h in holds], dtype=np.float64)
    labels = DBSCAN(eps=eps_px, min_samples=min_holds, metric="euclidean").fit_predict(centers)

    clusters: Dict[int, List[Hold]] = {}
    for hold, label in zip(holds, labels):
        if label == -1:
            continue  # spatial outlier — not part of any route
        clusters.setdefault(int(label), []).append(hold)
    return [c for _, c in sorted(clusters.items())]


def split_by_position_adaptive(
    holds: List[Hold], scale_eps: float, min_holds: int = 1
) -> List[List[Hold]]:
    """Like split_by_position, but perspective-aware for angled photos.

    Instead of a fixed pixel radius, two holds count as the same route when the
    gap between them is within `scale_eps` *hold-widths* — using the average of
    their `scale`s as the local yardstick. As the wall foreshortens, both gaps
    and hold sizes shrink together, so this ratio stays stable across the image
    where a single pixel `eps` would over-merge far holds and split near ones.

    Implemented as DBSCAN on a precomputed, scale-normalized distance matrix.
    """
    if not holds:
        return []
    if len(holds) == 1:
        return [holds] if min_holds <= 1 else []

    centers = np.array([h.spatial_center for h in holds], dtype=np.float64)
    scales = np.array([h.scale for h in holds], dtype=np.float64)
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    avg_scale = (scales[:, None] + scales[None, :]) / 2.0
    normalized = dist / avg_scale

    labels = DBSCAN(eps=scale_eps, min_samples=min_holds, metric="precomputed").fit_predict(normalized)

    clusters: Dict[int, List[Hold]] = {}
    for hold, label in zip(holds, labels):
        if label == -1:
            continue
        clusters.setdefault(int(label), []).append(hold)
    return [c for _, c in sorted(clusters.items())]


# ---------------------------------------------------------------------------
# Bridge from detector output to Holds
# ---------------------------------------------------------------------------
def build_holds(
    image_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int, float]],
    balance: bool = True,
    use_mask: bool = False,
) -> List["Hold"]:
    """Turn detector boxes into Holds carrying each region's dominant Lab color.

    `boxes` are (x1, y1, x2, y2, confidence) — exactly what hold_detector emits.
    With `balance=True` we white-balance the whole image first so colors read
    consistently. `use_mask` samples color from a GrabCut hold mask (excludes
    wall/pocket) instead of the box center. Boxes that crop to nothing are skipped.
    """
    img = utils.white_balance(image_bgr) if balance else image_bgr
    holds: List[Hold] = []
    for (x1, y1, x2, y2, conf) in boxes:
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = max(0, x2), max(0, y2)
        lab = utils.dominant_color_lab(img[cy1:cy2, cx1:cx2], use_mask=use_mask)
        if lab is None:
            continue
        holds.append(Hold(box=(x1, y1, x2, y2), confidence=float(conf), lab=lab))
    return holds


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def extract_routes(
    holds: List[Hold],
    color_eps: float = 28.0,
    spatial_eps_px: float = 250.0,
    spatial_min_holds: int = 1,
    references: Optional[Dict[str, np.ndarray]] = None,
    adaptive_spatial: bool = False,
    spatial_scale_eps: float = 8.0,
    chroma_min: float = 12.0,
    group_by: str = "lab",
    rescue: Optional[dict] = None,
) -> List[Route]:
    """Turn a flat list of holds into named routes.

    Steps: cluster by color, split each color group spatially, then name each
    resulting route by its nearest reference color (if `references` given —
    build them with utils.reference_labs(config["draw_colors"])).

    `adaptive_spatial` switches the spatial step to a perspective-aware,
    hold-size-relative split (`spatial_scale_eps` in hold-widths) — better for
    angled photos; the fixed-pixel `spatial_eps_px` is used otherwise.

    Returns routes sorted by hold count (largest first) for stable output.
    """
    def split(group: List[Hold]) -> List[List[Hold]]:
        if adaptive_spatial:
            return split_by_position_adaptive(group, spatial_scale_eps, spatial_min_holds)
        return split_by_position(group, spatial_eps_px, spatial_min_holds)

    routes: List[Route] = []

    # "name" buckets holds by their discrete color name (robust on dense walls);
    # "lab" is the original per-image DBSCAN on raw Lab. Name mode needs refs.
    if group_by == "name" and references is not None:
        color_groups = cluster_by_name(holds, references, chroma_min, rescue)
    else:
        color_groups = cluster_by_color(holds, eps=color_eps)

    for color_group in color_groups:
        for route_holds in split(color_group):
            rep_lab = np.median([h.lab for h in route_holds], axis=0).astype(np.float64)
            name = utils.nearest_color_name(rep_lab, references, chroma_min, rescue) if references else ""
            for h in route_holds:
                h.name = name
            routes.append(Route(color_name=name, lab=rep_lab, holds=route_holds))

    routes.sort(key=lambda r: r.hold_count, reverse=True)
    return routes
