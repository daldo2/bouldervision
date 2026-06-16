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


def main() -> None:
    """Parse the command-line argument (image path) and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Detect climbing holds in an image and classify them by color."
    )
    parser.add_argument("image", help="Path to the input climbing-wall image.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a settings.yaml (defaults to config/settings.yaml).",
    )
    args = parser.parse_args()

    output_path = run(args.image, args.config)
    print("-" * 40)
    print(f"Done. Open {output_path} to view the result.")


if __name__ == "__main__":
    main()
