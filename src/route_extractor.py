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

    @property
    def center(self) -> Tuple[float, float]:
        """Centroid of the bounding box — used for spatial clustering."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


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

    centers = np.array([h.center for h in holds], dtype=np.float64)
    labels = DBSCAN(eps=eps_px, min_samples=min_holds, metric="euclidean").fit_predict(centers)

    clusters: Dict[int, List[Hold]] = {}
    for hold, label in zip(holds, labels):
        if label == -1:
            continue  # spatial outlier — not part of any route
        clusters.setdefault(int(label), []).append(hold)
    return [c for _, c in sorted(clusters.items())]


# ---------------------------------------------------------------------------
# Bridge from detector output to Holds
# ---------------------------------------------------------------------------
def build_holds(
    image_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int, float]],
    balance: bool = True,
) -> List["Hold"]:
    """Turn detector boxes into Holds carrying each region's dominant Lab color.

    `boxes` are (x1, y1, x2, y2, confidence) — exactly what hold_detector emits.
    With `balance=True` we white-balance the whole image first so colors read
    consistently. Boxes that crop to nothing are skipped.
    """
    img = utils.white_balance(image_bgr) if balance else image_bgr
    holds: List[Hold] = []
    for (x1, y1, x2, y2, conf) in boxes:
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = max(0, x2), max(0, y2)
        lab = utils.dominant_color_lab(img[cy1:cy2, cx1:cx2])
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
) -> List[Route]:
    """Turn a flat list of holds into named routes.

    Steps: cluster by color, split each color group spatially, then name each
    resulting route by its nearest reference color (if `references` given —
    build them with utils.reference_labs(config["draw_colors"])).

    Returns routes sorted by hold count (largest first) for stable output.
    """
    routes: List[Route] = []

    for color_group in cluster_by_color(holds, eps=color_eps):
        for route_holds in split_by_position(color_group, spatial_eps_px, spatial_min_holds):
            rep_lab = np.median([h.lab for h in route_holds], axis=0).astype(np.float64)
            name = utils.nearest_color_name(rep_lab, references) if references else ""
            for h in route_holds:
                h.name = name
            routes.append(Route(color_name=name, lab=rep_lab, holds=route_holds))

    routes.sort(key=lambda r: r.hold_count, reverse=True)
    return routes
