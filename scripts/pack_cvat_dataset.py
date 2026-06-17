"""Package data/pose_annotation/ into a CVAT-ready COCO Keypoints dataset zip.

CVAT will only DRAW the pre-labelled skeleton if a Skeleton label exists. Importing
the bare predictions_coco.json onto a plain task shows blank images (no label to
attach points to). The reliable fix is to import a whole DATASET into an empty
CVAT project — CVAT then auto-creates the skeleton label from the COCO category.

This builds the exact layout CVAT expects:

  pose_cvat_dataset.zip
  ├── images/<all frames>
  └── annotations/person_keypoints_default.json   (copied from predictions_coco.json)

Run:  python scripts/pack_cvat_dataset.py
Then in CVAT: Projects -> + -> Create from dataset -> COCO Keypoints 1.0 -> upload
this zip. The skeleton label "climber" (17 COCO joints) is created automatically.
"""
from __future__ import annotations

import json
import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT_ROOT, "data", "pose_annotation")


def main() -> int:
    coco_path = os.path.join(SRC, "predictions_coco.json")
    img_dir = os.path.join(SRC, "images")
    if not os.path.exists(coco_path):
        print("Missing predictions_coco.json — run scripts/export_pose_frames.py first.")
        return 1

    coco = json.load(open(coco_path))
    listed = {im["file_name"] for im in coco["images"]}
    out_zip = os.path.join(SRC, "pose_cvat_dataset.zip")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        # Only pack images that actually have an annotation entry (skip orphans).
        packed = 0
        for fname in sorted(listed):
            fp = os.path.join(img_dir, fname)
            if os.path.exists(fp):
                z.write(fp, arcname=f"images/{fname}")
                packed += 1
        z.writestr("annotations/person_keypoints_default.json", json.dumps(coco))

    print(f"Packed {packed} images + annotations -> {out_zip}")
    print("CVAT: Projects -> + -> Create from dataset -> 'COCO Keypoints 1.0' -> "
          "upload this zip. The 'climber' skeleton label is created for you.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
