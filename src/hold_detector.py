"""
hold_detector.py — Phase 1 proof of concept.

Pipeline implemented here:

    image ──▶ YOLOv8 detection ──▶ for each box: crop ──▶ classify HSV color
          ──▶ draw colored, labeled boxes ──▶ save output + print summary

This file is written to be read top-to-bottom by a Python developer who has
*no* computer-vision background. Every non-obvious step explains WHAT it does
and WHY.

IMPORTANT (read this first):
    The MVP loads the *generic* pretrained YOLOv8n model. That model was
    trained on the COCO dataset (people, cars, cups...) and has never seen a
    climbing hold. So on a real wall photo it will detect few or zero "holds".
    That is expected. The point of this script is to prove the full
    detect → crop → color-classify → draw pipeline runs end to end. Swapping in
    a hold-trained model later (see models/README.md) makes the detections
    real without changing this code.

Run it:
    python src/hold_detector.py path/to/climbing_wall.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import List, Tuple

import numpy as np

# Make `import utils` work whether you run this as `python src/hold_detector.py`
# or `python -m src.hold_detector`. We add this file's own directory to the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils  # noqa: E402  (import after sys.path tweak, intentional)

# NOTE on imports: we do NOT import ultralytics/torch at module load.
# Those are heavy (hundreds of MB) and only needed for the actual detection
# step. Importing them lazily means the rest of this module — and the whole
# color-classification pipeline — imports and runs fine on a machine where the
# deep-learning stack isn't installed yet (e.g. offline, or a CI box). You only
# hit the requirement at the moment you genuinely need a model.

# Directory where we keep local model weights, so an offline machine can use a
# pre-placed yolov8n.pt instead of trying (and failing) to download one.
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def resolve_weights(model_path: str) -> str:
    """Decide which weights file to hand to YOLO.

    Ultralytics, given a bare name like "yolov8n.pt", will try to download it.
    That fails on an offline machine. So before falling back to that behavior we
    look for the file (a) as given, and (b) inside our local models/ directory.
    Whichever exists on disk is used as-is; if none do, we return the original
    name and let ultralytics attempt its download (which needs a network).
    """
    if os.path.isfile(model_path):
        return model_path

    local_copy = os.path.join(MODELS_DIR, os.path.basename(model_path))
    if os.path.isfile(local_copy):
        return local_copy

    # Nothing on disk — return the bare name; ultralytics will try to fetch it.
    return model_path


def load_detector(model_path: str):
    """Load (and, if needed, auto-download) the YOLOv8 detection model.

    The `ultralytics` import lives *inside* this function on purpose (see the
    note above): only callers that actually run detection pay for the heavy
    dependency, and a missing install produces a clear, actionable message
    instead of an import error at the top of the file.

    Ultralytics caches weights after the first download, so later runs are
    fast. A bare name like "yolov8n.pt" is downloaded; a path (or a file we
    find in models/) is loaded directly — see `resolve_weights`.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "The 'ultralytics' package (and its torch dependency) is not "
            "installed, so YOLO detection can't run.\n"
            "Install the full stack with:\n"
            "    pip install -r requirements.txt\n"
            "Everything else in BoulderVision (color classification, drawing, "
            "tests) works without it."
        ) from exc

    weights = resolve_weights(model_path)
    print(f"[1/5] Loading detection model: {weights}")
    if not os.path.isfile(weights):
        print(
            "      (weights not found locally — ultralytics will try to "
            "download them; this step needs a network connection)"
        )
    return YOLO(weights)


def detect_holds(
    model: YOLO,
    image: np.ndarray,
    confidence: float,
    iou: float,
    max_detections: int,
) -> List[Tuple[int, int, int, int, float]]:
    """Run YOLO on the image and return a list of detected boxes.

    Each returned box is (x1, y1, x2, y2, confidence) in pixel coordinates.
    We deliberately ignore YOLO's *class label* here: with the generic model
    the class would be a COCO category (irrelevant), and with a future
    hold-only model every detection is just "hold". Color — not class — is
    what matters for routes, and we compute that ourselves next.
    """
    print("[2/5] Running detection...")

    # `model(...)` returns a list of Results (one per image); we passed one image.
    results = model(
        image,
        conf=confidence,     # drop boxes less confident than this
        iou=iou,             # non-max suppression: merge heavily overlapping boxes
        max_det=max_detections,
        verbose=False,       # we print our own summary instead of YOLO's
    )

    boxes: List[Tuple[int, int, int, int, float]] = []
    result = results[0]

    # result.boxes holds every detection. `.xyxy` gives corner coordinates,
    # `.conf` the confidence score. We move them to CPU and convert to plain
    # Python numbers so the rest of the code doesn't depend on torch tensors.
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        boxes.append((int(x1), int(y1), int(x2), int(y2), conf))

    return boxes


def detect_objects(
    model: YOLO,
    image: np.ndarray,
    confidence: float,
    iou: float,
    max_detections: int,
    augment: bool = False,
) -> List[Tuple[int, int, int, int, float, int]]:
    """Like detect_holds, but KEEPS the class id: (x1, y1, x2, y2, conf, cls).

    The route pipeline needs the class to tell volumes (e.g. class 1 in best.pt)
    from holds, then post-filters markers/tape by shape — see detection_filter.
    Phase-1 `run()` still uses detect_holds and ignores class, as before.

    `augment=True` runs YOLO's test-time augmentation (multi-scale + flips) for
    higher recall — slower, used by the annotation bootstrap, off in production.
    """
    print("[2/5] Running detection...")
    results = model(image, conf=confidence, iou=iou, max_det=max_detections,
                    augment=augment, verbose=False)
    out: List[Tuple[int, int, int, int, float, int]] = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
        out.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
    return out


def classify_and_annotate(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int, float]],
    config: dict,
) -> Tuple[np.ndarray, List[Tuple[str, float, float]]]:
    """For each detected box, classify its color and draw a labeled rectangle.

    Returns:
      - annotated: a COPY of the image with boxes drawn (original untouched).
      - detections: a list of (color, color_coverage, detection_confidence)
                    used to build the console summary.
    """
    print("[3/5] Classifying hold colors...")

    color_ranges = config["colors"]
    draw_colors = config["draw_colors"]

    # Work on a copy so the pixels we crop for color analysis stay pristine
    # (drawing on the image would contaminate later crops if boxes overlap).
    annotated = image.copy()
    detections: List[Tuple[str, float, float]] = []

    for (x1, y1, x2, y2, conf) in boxes:
        # Crop the detected region out of the ORIGINAL image for color analysis.
        # Clamp coordinates so a box touching the image edge can't go negative.
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = max(0, x2), max(0, y2)
        crop = image[cy1:cy2, cx1:cx2]

        # Turn that crop into a named color (red/blue/...) via HSV analysis.
        color_name, coverage = utils.classify_color(crop, color_ranges)

        # Pick the BGR color we'll draw this box in, and build its label.
        bgr = utils.draw_color_for(color_name, draw_colors)
        label = f"{color_name} {conf:.2f}"

        utils.draw_box(annotated, (x1, y1, x2, y2), label, bgr)
        detections.append((color_name, coverage, conf))

    return annotated, detections


def print_summary(detections: List[Tuple[str, float, float]]) -> None:
    """Print a human-readable summary of what was found, grouped by color."""
    print("[5/5] Detection summary")
    print("-" * 40)

    if not detections:
        # The common Phase-0 case: the generic model found nothing hold-like.
        print("No holds detected.")
        print(
            "Tip: the MVP uses generic yolov8n (COCO), which doesn't know\n"
            "climbing holds. Fine-tune a hold model (see models/README.md)\n"
            "to get real detections."
        )
        return

    print(f"Total detections: {len(detections)}")

    # Count how many holds of each color we found.
    color_counts = Counter(color for color, _, _ in detections)
    print("\nHolds per color:")
    for color, count in color_counts.most_common():
        print(f"  {color:<8} {count}")


def run(image_path: str, config_path: str | None = None) -> str:
    """End-to-end: load model, detect, classify, draw, save. Returns output path."""
    # Load all tunable settings (thresholds, color ranges) from YAML.
    config = utils.load_config(config_path) if config_path else utils.load_config()

    # Read the input image off disk (raises a clear error if missing).
    image = utils.read_image(image_path)

    # Build and run the detector.
    model = load_detector(config["models"]["hold_detector"])
    det_cfg = config["detection"]
    boxes = detect_holds(
        model,
        image,
        confidence=det_cfg["confidence"],
        iou=det_cfg["iou"],
        max_detections=det_cfg["max_detections"],
    )

    # Color-classify each box and draw the annotated image.
    annotated, detections = classify_and_annotate(image, boxes, config)

    # Save the result next to the other samples.
    output_path = utils.resolve_path(config["paths"]["output_image"])
    print(f"[4/5] Saving annotated image to: {output_path}")
    utils.save_image(annotated, output_path)

    # Tell the user what we found.
    print_summary(detections)
    return output_path


def run_holds(image_path: str, config_path: str | None = None) -> str:
    """Detect holds and draw each one in its OWN detected color, without grouping
    them into routes. Useful for inspecting raw detection + color quality.

    Volumes / markers / tape are still set aside and overlaid in neutral colors.
    Returns the output path.
    """
    import route_extractor as rex
    import detection_filter as dfilt

    config = utils.load_config(config_path) if config_path else utils.load_config()
    image = utils.read_image(image_path)

    model = load_detector(config["models"]["hold_detector"])
    det = config["detection"]
    raw = detect_objects(model, image, det["confidence"], det["iou"], det["max_detections"])
    dets = dfilt.classify_detections(raw, image, config["filter"])

    print("[3/4] Reading per-hold colors (no route grouping)...")
    use_mask = config.get("color_naming", {}).get("use_mask", False)
    holds = rex.build_holds(image, dfilt.holds_only(dets), use_mask=use_mask)
    refs = utils.reference_labs(config["draw_colors"],
                                config.get("color_naming", {}).get("hue_anchors"))
    chroma_min = config.get("color_naming", {}).get("chroma_min", 16.0)
    rescue = config.get("color_naming", {}).get("rescue")
    draw_colors = config["draw_colors"]

    annotated = image.copy()
    counts: dict = {}
    for h in holds:
        name = utils.nearest_color_name(h.lab, refs, chroma_min, rescue)
        counts[name] = counts.get(name, 0) + 1
        utils.draw_box(annotated, h.box, name, utils.draw_color_for(name, draw_colors))
    aside_style = {dfilt.VOLUME: (150, 150, 150), dfilt.MARKER: (200, 0, 200),
                   dfilt.TAPE: (200, 200, 0), dfilt.DOWNCLIMB: (255, 0, 255)}
    aside = {k: 0 for k in aside_style}
    for d in dets:
        if d.kind in aside_style:
            aside[d.kind] += 1
            utils.draw_box(annotated, d.box, d.kind, aside_style[d.kind])

    output_path = utils.resolve_path(config["paths"]["holds_image"])
    print(f"[4/4] Saving per-hold map to: {output_path}")
    utils.save_image(annotated, output_path)

    print("-" * 40)
    print(f"{len(holds)} holds by color: " + ", ".join(f"{n}={c}" for n, c in sorted(counts.items(), key=lambda x: -x[1])))
    set_aside = ", ".join(f"{n} {k}{'s' if n != 1 else ''}" for k, n in aside.items() if n)
    if set_aside:
        print(f"Set aside (not holds): {set_aside}")
    return output_path


def run_routes(image_path: str, config_path: str | None = None, corners=None) -> str:
    """End-to-end Phase 1+2: detect holds, read their colors, group into routes,
    draw the route map, save it. Returns the output path.

    `corners` (4 [x, y] wall points) overrides config to rectify a steeply
    angled wall before spatial grouping. None falls back to config.

    This is the full pipeline the project is building toward. It needs a real
    hold detector to be useful — with generic yolov8n it still runs, just on
    whatever boxes COCO produces.
    """
    import route_extractor as rex  # imported here; pulls sklearn, not torch
    import detection_filter as dfilt

    config = utils.load_config(config_path) if config_path else utils.load_config()
    image = utils.read_image(image_path)

    # 1. Detect candidate features (keep the class id for the volume split).
    model = load_detector(config["models"]["hold_detector"])
    det = config["detection"]
    raw = detect_objects(model, image, det["confidence"], det["iou"], det["max_detections"])

    # 1b. Sort detections: only true holds become routes; volumes, start/zone
    #     markers and difficulty tape are set aside (and reported below).
    dets = dfilt.classify_detections(raw, image, config["filter"])
    hold_boxes = dfilt.holds_only(dets)
    aside = {k: len(dfilt.of_kind(dets, k))
             for k in (dfilt.VOLUME, dfilt.MARKER, dfilt.TAPE, dfilt.DOWNCLIMB)}

    # 2. Read each hold's real color (Lab) and 3. group into routes.
    print("[3/5] Reading hold colors and grouping into routes...")
    use_mask = config.get("color_naming", {}).get("use_mask", False)
    holds = rex.build_holds(image, hold_boxes, use_mask=use_mask)

    # 2b. Optional perspective rectification for steeply angled photos: warp hold
    #     positions into a frontal plane so spatial grouping is not foreshortened.
    pcfg = config.get("perspective", {})
    wall_corners = corners if corners is not None else (pcfg.get("corners") or None)
    if wall_corners and (corners is not None or pcfg.get("enabled")):
        import perspective
        H, out_size = perspective.compute_homography(wall_corners)
        perspective.rectify_holds(holds, H)
        print(f"     perspective: rectified {len(holds)} holds to {out_size[0]}x{out_size[1]} frontal plane")
        if pcfg.get("save_preview", True):
            preview = perspective.warp_image(image, H, out_size)
            ppath = utils.resolve_path(config["paths"]["routes_image"]).replace(".jpg", "_rectified.jpg")
            utils.save_image(preview, ppath)
            print(f"     perspective: wrote preview {ppath}")

    refs = utils.reference_labs(config["draw_colors"],
                                config.get("color_naming", {}).get("hue_anchors"))
    rcfg = config["routes"]
    routes = rex.extract_routes(
        holds,
        color_eps=rcfg["color_eps"],
        spatial_eps_px=rcfg["spatial_eps_px"],
        spatial_min_holds=rcfg["spatial_min_holds"],
        references=refs,
        adaptive_spatial=rcfg.get("adaptive_spatial", False),
        spatial_scale_eps=rcfg.get("spatial_scale_eps", 8.0),
        chroma_min=config.get("color_naming", {}).get("chroma_min", 16.0),
        group_by=rcfg.get("group_by", "lab"),
        rescue=config.get("color_naming", {}).get("rescue"),
    )

    # 4. Draw and save the route map. Overlay set-aside detections in distinct
    #    neutral colors so their classification can be eyeballed for mistakes.
    annotated = utils.draw_routes(image, routes)
    aside_style = {dfilt.VOLUME: (150, 150, 150), dfilt.MARKER: (200, 0, 200),
                   dfilt.TAPE: (200, 200, 0), dfilt.DOWNCLIMB: (255, 0, 255)}
    for d in dets:
        if d.kind in aside_style:
            utils.draw_box(annotated, d.box, d.kind, aside_style[d.kind])
    output_path = utils.resolve_path(config["paths"]["routes_image"])
    print(f"[4/5] Saving route map to: {output_path}")
    utils.save_image(annotated, output_path)

    # 5. Summary.
    print("[5/5] Route summary")
    print("-" * 40)
    if not routes:
        print("No routes found (no holds detected — is a hold-trained model set?).")
    else:
        print(f"Found {len(routes)} route(s) from {len(holds)} holds:")
        for i, r in enumerate(routes, start=1):
            print(f"  #{i}  {r.color_name or '?':<8} {r.hold_count} holds")
    set_aside = ", ".join(f"{n} {k}{'s' if n != 1 else ''}" for k, n in aside.items() if n)
    if set_aside:
        print(f"Set aside (not routes): {set_aside}")
    return output_path


def main() -> None:
    """Parse the command-line arguments and run the chosen pipeline."""
    parser = argparse.ArgumentParser(
        description="Detect climbing holds in an image and classify them by color."
    )
    parser.add_argument("image", help="Path to the input climbing-wall image.")
    parser.add_argument(
        "--routes",
        action="store_true",
        help="Group detected holds into routes (Phase 2) and draw a route map.",
    )
    parser.add_argument(
        "--holds",
        action="store_true",
        help="Draw each detected hold in its own color WITHOUT grouping into routes.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a settings.yaml (defaults to config/settings.yaml).",
    )
    parser.add_argument(
        "--corners",
        default=None,
        help='Rectify a steeply angled wall before grouping. 4 wall corners as '
             '"x,y x,y x,y x,y" (image pixels, any order). Implies --routes.',
    )
    args = parser.parse_args()

    corners = None
    if args.corners:
        corners = [tuple(float(v) for v in pair.split(",")) for pair in args.corners.split()]
        if len(corners) != 4:
            parser.error("--corners needs exactly 4 'x,y' points")

    if args.holds:
        output_path = run_holds(args.image, args.config)
    elif args.routes or corners:
        output_path = run_routes(args.image, args.config, corners=corners)
    else:
        output_path = run(args.image, args.config)
    print("-" * 40)
    print(f"Done. Open {output_path} to view the result.")


if __name__ == "__main__":
    main()
