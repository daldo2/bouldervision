# Models

Trained model weights live in this directory. **Weights are not committed to
git** (see the root `.gitignore`) because they are large and reproducible.

## What goes here

| File | Purpose | How to get it |
|------|---------|---------------|
| `yolov8n.pt` | Generic pretrained detector (Phase 0/1 MVP) | Auto-downloaded by Ultralytics on first run |
| `yolov8x-pose.pt` | Pose estimation (Phase 3) | Auto-downloaded by Ultralytics on first run |
| `holds_yolov8n.pt` | **Hold-specific** detector fine-tuned on labeled climbing holds | Produced in Phase 1 (see below) |

The auto-downloaded models land in Ultralytics' cache, but you can also place
them here and reference them by path in `config/settings.yaml`.

## Training a hold-specific detector (Phase 1)

The MVP uses generic `yolov8n.pt`, which was trained on COCO and does **not**
know what a climbing hold is. To get real detections you fine-tune on your own
labeled holds:

1. Label ~200–500 holds across a handful of wall photos (Roboflow / CVAT).
   One class, `hold`, is enough to start.
2. Export in YOLO format (images + `.txt` label files + `data.yaml`).
3. Train:
   ```bash
   yolo detect train model=yolov8n.pt data=path/to/data.yaml epochs=100 imgsz=640
   ```
4. Copy the best weights here:
   ```bash
   cp runs/detect/train/weights/best.pt models/holds_yolov8n.pt
   ```
5. Update `config/settings.yaml`:
   ```yaml
   models:
     hold_detector: "models/holds_yolov8n.pt"
   ```

After that, `hold_detector.py` will detect actual holds instead of COCO objects.
