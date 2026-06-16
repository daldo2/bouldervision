"""
route_extractor.py — Phase 2 (scaffold).

Goal: take the colored holds produced by hold_detector.py and group them into
distinct *routes* (a.k.a. problems). The core idea:

    same color  ⇒  probably the same route
    BUT two different problems can share a color on different wall sections,
    so we also cluster by SPATIAL position to split them apart.

This module is a scaffold: the data structures and the color-grouping step are
implemented, and the spatial-clustering step has a clear TODO with the intended
approach (DBSCAN on hold centroids). Fill it in during Phase 2.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Hold:
    """One detected hold, enriched with its classified color."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    color: str
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        """Centroid of the hold's bounding box — used for spatial clustering."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Route:
    """A set of holds believed to form a single climbing problem."""
    color: str
    holds: List[Hold] = field(default_factory=list)

    @property
    def hold_count(self) -> int:
        return len(self.holds)


def group_by_color(holds: List[Hold]) -> Dict[str, List[Hold]]:
    """First pass: bucket holds by their classified color.

    This is the cheap, obvious grouping. It over-groups when two routes share a
    color, which the spatial step below is meant to fix.
    """
    buckets: Dict[str, List[Hold]] = defaultdict(list)
    for hold in holds:
        if hold.color == "unknown":
            continue  # don't route holds we couldn't classify
        buckets[hold.color].append(hold)
    return buckets


def split_by_position(color: str, holds: List[Hold]) -> List[Route]:
    """Second pass: split a single color bucket into spatially-coherent routes.

    INTENDED APPROACH (Phase 2 TODO):
      - Compute each hold's centroid (Hold.center).
      - Run DBSCAN (sklearn.cluster.DBSCAN) on those centroids. DBSCAN is a good
        fit because it finds clusters without us specifying how many routes
        there are, and it labels far-away outliers as noise (-1).
      - Each resulting cluster becomes one Route; drop noise points.

    For now we return the whole color bucket as a single route so the rest of
    the pipeline has something to consume.
    """
    # TODO(phase-2): replace this single-route fallback with DBSCAN clustering.
    return [Route(color=color, holds=holds)]


def extract_routes(holds: List[Hold]) -> List[Route]:
    """Full Phase 2 entry point: holds → list of routes."""
    routes: List[Route] = []
    for color, color_holds in group_by_color(holds).items():
        routes.extend(split_by_position(color, color_holds))
    # Order routes by size (biggest first) for stable, useful output.
    routes.sort(key=lambda r: r.hold_count, reverse=True)
    return routes
