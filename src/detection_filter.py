"""detection_filter.py — sort raw detections into holds vs. non-route furniture.

The detector returns a box for everything it thinks is a climbing feature, but
not everything on a wall belongs to a *route*:

  - volumes        large bolt-on shapes; often a neutral colour shared by many
                   routes, so they must not form their own colour route.
  - start/zone     small solid-black circular stickers next to holds.
    markers
  - difficulty     long thin strips of tape marking a route's grade.
    tape

Only true holds should flow into route grouping. This module makes that split
with simple, per-gym-tunable heuristics. It is a STOPGAP: the clean, reliable
fix is a detector retrained with these as their own classes (see PROJECT_PLAN).

Pure + offline: no torch, no model. Takes boxes + the image, returns labels.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402

# Detection kinds. Only HOLD feeds route extraction; the rest are drawn/reported
# separately so the user can see what was set aside and why.
HOLD = "hold"
VOLUME = "volume"
MARKER = "marker"
TAPE = "tape"

Box = Tuple[int, int, int, int]


@dataclass
class Detection:
    """One raw detection, plus the kind we sorted it into."""

    box: Box                 # (x1, y1, x2, y2) in pixels
    confidence: float
    cls: int                 # detector class id (-1 if unknown)
    kind: str = HOLD


def _wh(box: Box) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1), max(0, y2 - y1)


def classify_detection(
    box: Box,
    cls: int,
    image_bgr: np.ndarray,
    fcfg: dict,
) -> str:
    """Decide whether a detection is a hold, volume, marker, or tape.

    Order matters: tape (extreme shape) and volume (class/size) are decided from
    geometry alone; the marker test is last and deliberately strict (small AND
    square AND dark AND near-neutral) so it does not swallow real black holds.
    """
    w, h = _wh(box)
    if w == 0 or h == 0:
        return HOLD  # degenerate; let downstream skip it
    img_area = float(image_bgr.shape[0] * image_bgr.shape[1])
    area_frac = (w * h) / img_area
    long_side, short_side = max(w, h), max(1, min(w, h))
    aspect = long_side / short_side

    # 1. Difficulty tape: a long, thin strip — nothing else is this elongated.
    if aspect >= fcfg["tape_aspect"]:
        return TAPE

    # 2. Volume: the model's own class, or simply too big to be a hold.
    if cls == fcfg["volume_class"] or area_frac >= fcfg["volume_area_frac"]:
        return VOLUME

    # 3. Start/zone marker: small, round-ish, dark, near-neutral black dot.
    if area_frac <= fcfg["marker_max_area_frac"] and aspect <= fcfg["marker_max_aspect"]:
        x1, y1, x2, y2 = box
        crop = image_bgr[max(0, y1):y2, max(0, x1):x2]
        lab = utils.dominant_color_lab(crop)
        if lab is not None:
            chroma, _ = utils._chroma_hue(lab)
            if lab[0] <= fcfg["marker_dark_l_max"] and chroma <= fcfg["marker_chroma_max"]:
                return MARKER

    return HOLD


def classify_detections(
    boxes: List[Tuple[int, int, int, int, float, int]],
    image_bgr: np.ndarray,
    fcfg: dict,
) -> List[Detection]:
    """Label a list of (x1, y1, x2, y2, conf, cls) detections by kind."""
    out: List[Detection] = []
    for (x1, y1, x2, y2, conf, cls) in boxes:
        box = (int(x1), int(y1), int(x2), int(y2))
        kind = classify_detection(box, int(cls), image_bgr, fcfg)
        out.append(Detection(box=box, confidence=float(conf), cls=int(cls), kind=kind))
    return out


def holds_only(dets: List[Detection]) -> List[Tuple[int, int, int, int, float]]:
    """Boxes for route extraction: the holds, in build_holds' 5-tuple shape."""
    return [(*d.box, d.confidence) for d in dets if d.kind == HOLD]


def of_kind(dets: List[Detection], kind: str) -> List[Detection]:
    return [d for d in dets if d.kind == kind]
