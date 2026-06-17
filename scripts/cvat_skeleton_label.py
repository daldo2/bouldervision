"""Emit a CVAT 'climber' skeleton label (17 COCO joints) for the Raw label editor.

CVAT will not draw imported COCO keypoints unless a matching Skeleton label already
exists — otherwise import fails with "Label 'nose' is not registered". This prints
(and writes) the label JSON to paste into a CVAT project's label editor:

  CVAT project -> Labels -> 'Raw' tab -> paste -> Done.

Then Import dataset (COCO Keypoints 1.0) maps the pre-labels by joint name. The
joint NAMES must match the COCO category in predictions_coco.json (they do here).
Point coordinates only shape the little editor preview — they don't affect import.

Run:  python scripts/cvat_skeleton_label.py
"""
from __future__ import annotations

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT_ROOT, "data", "pose_annotation", "cvat_skeleton_label.json")

# COCO-17 names in order, with a rough standing-figure layout (0-100 canvas) just
# for the editor preview.
JOINTS = [
    ("nose", 50, 10), ("left_eye", 54, 8), ("right_eye", 46, 8),
    ("left_ear", 58, 10), ("right_ear", 42, 10),
    ("left_shoulder", 60, 24), ("right_shoulder", 40, 24),
    ("left_elbow", 66, 40), ("right_elbow", 34, 40),
    ("left_wrist", 70, 55), ("right_wrist", 30, 55),
    ("left_hip", 57, 54), ("right_hip", 43, 54),
    ("left_knee", 59, 76), ("right_knee", 41, 76),
    ("left_ankle", 60, 95), ("right_ankle", 40, 95),
]
EDGES = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def main() -> int:
    # Sublabel (point) ids start at 1; node-id in the SVG == sublabel id. The
    # skeleton root carries NO numeric id (server-assigned) — providing one
    # collides with the sublabel ids and CVAT rejects the SVG references.
    nid = {name: i + 1 for i, (name, _, _) in enumerate(JOINTS)}
    pos = {name: (x, y) for name, x, y in JOINTS}

    sublabels = [
        {"name": name, "id": nid[name], "type": "points", "attributes": []}
        for name, _, _ in JOINTS
    ]

    svg = ""
    for a, b in EDGES:  # lines first so circles render on top
        (x1, y1), (x2, y2) = pos[a], pos[b]
        svg += (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" '
                f'data-type="edge" data-node-from="{nid[a]}" data-node-to="{nid[b]}" '
                f'stroke-width="0.5"></line>')
    for name, x, y in JOINTS:
        svg += (f'<circle r="1.5" stroke="black" fill="#b3b3b3" cx="{x}" cy="{y}" '
                f'stroke-width="0.1" data-type="element node" '
                f'data-node-id="{nid[name]}" data-label-name="{name}"></circle>')

    label = [{
        "name": "climber", "color": "#fa3253", "type": "skeleton",
        "attributes": [], "sublabels": sublabels, "svg": svg,
    }]

    with open(OUT, "w") as f:
        json.dump(label, f, indent=2)
    print(json.dumps(label, indent=2))
    print(f"\nWritten to {OUT}")
    print("Paste this into CVAT: project -> Labels -> 'Raw' -> replace contents -> Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
