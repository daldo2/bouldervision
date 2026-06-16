# 🧗 BoulderVision

Computer-vision powered analysis of bouldering videos. BoulderVision detects
climbing holds, groups them into routes by color, tracks the climber's body
pose, and (in later phases) estimates the force each limb carries.

## What it does

| Capability | Status |
|------------|--------|
| Detect climbing holds in an image | Phase 1 (MVP) |
| Classify each hold by color (red / blue / yellow / purple / green / black) | Phase 1 |
| Group same-colored holds into a single route/problem | Phase 2 |
| Track the climber's 17-keypoint skeleton across video frames | Phase 3 |
| Detect which holds the climber is touching per frame | Phase 3 |
| Estimate per-limb force distribution | Phase 4 (research) |
| REST API for video upload + analysis | Phase 5 |
| React Native mobile app | Phase 6 |

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full phased roadmap.

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the hold detector on an image
python src/hold_detector.py path/to/climbing_wall.jpg

# ...or use the CLI entry point
python scripts/run_analysis.py --image path/to/climbing_wall.jpg
```

The first run downloads the pretrained `yolov8n.pt` weights (~6 MB) and writes
an annotated image to `data/output/output.jpg`.

> **Note on the MVP detector:** Phase 1 uses a *pretrained* YOLOv8n model that
> was trained on the generic COCO dataset, not on climbing holds. It will not
> reliably find real holds yet — it is wired up so the full pipeline
> (detect → crop → color-classify → draw) runs end to end. Training a
> hold-specific model is Phase 1's follow-up task. See `models/README.md`.

## Project layout

```
bouldervision/
├── config/settings.yaml     # model paths, thresholds, color ranges
├── data/
│   ├── input/               # images/videos to analyze (sample wall.jpg here)
│   └── output/              # generated annotated results (gitignored)
├── models/                  # trained weights live here (gitignored)
├── src/                     # library code
│   ├── hold_detector.py     # hold detection + color classification
│   ├── pose_estimator.py    # skeleton tracking
│   ├── route_extractor.py   # group holds into routes
│   ├── video_pipeline.py    # full-video orchestration
│   └── utils.py             # drawing + color + I/O helpers
├── notebooks/               # exploration
├── scripts/run_analysis.py  # CLI entry point
└── tests/
```

## Requirements

- Python 3.9+
- See [`requirements.txt`](requirements.txt)

## License

TBD.
