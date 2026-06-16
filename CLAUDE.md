# BoulderVision — context for Claude

Computer-vision analysis of bouldering videos: detect climbing holds, classify
them by color, group same-colored holds into routes, track climber pose, and
(later) estimate per-limb force. Full roadmap in `PROJECT_PLAN.md`.

## How to run

The project uses a local virtualenv at `.venv` (Python 3.14, with torch +
ultralytics already installed). Heavy ML deps are NOT system-installed.

```bash
source .venv/bin/activate
python -m src.hold_detector data/input/wall.jpg   # writes data/output/output.jpg
pytest -q                                          # 8 offline tests
```

Run modules with `python -m src.<name>` (package imports), not as loose scripts.

## Layout

```
config/settings.yaml   # model paths, thresholds, HSV color ranges — tune here, not in code
data/input/            # images/videos to analyze (sample wall.jpg lives here)
data/output/           # generated results (gitignored)
models/                # weights (yolov8n.pt cached here; *.pt gitignored)
src/hold_detector.py   # detect -> crop -> HSV color-classify -> draw  (Phase 1, works)
src/utils.py           # config load, classify_color, draw helpers (pure, tested)
src/route_extractor.py # Phase 2 scaffold
src/pose_estimator.py  # Phase 3 scaffold
src/video_pipeline.py  # full-video orchestration scaffold
tests/                 # offline-safe: exercise color logic, never load a model
```

## Conventions / things to know

- **Offline-safe by design:** `ultralytics`/`torch` are imported lazily inside
  `load_detector()`. The module imports and all color logic + tests run without
  them; only actual YOLO detection needs them. Keep it that way.
- **Weights resolution:** `resolve_weights()` prefers a local `models/*.pt`
  before attempting a network download. Works offline once weights are cached.
- **MVP caveat:** detection uses generic pretrained `yolov8n` (COCO), which does
  not know climbing holds — few/zero real detections expected until a
  hold-specific model is fine-tuned (Phase 1 follow-up, see `models/README.md`).
- All tunables (thresholds, colors, paths) live in `config/settings.yaml`.

## Status (as of 2026-06-16)

- Phase 0 done; Phase 1 pipeline runs end-to-end on a synthetic test image.
- GitHub: https://github.com/daldo2/bouldervision (public). Branch: `master`.

## Next steps

1. Test on a REAL climbing-wall photo (drop in `data/input/`), tune HSV ranges.
2. Phase 2: cluster detected holds by color into routes (`route_extractor.py`).
3. Optional: fine-tune a hold-specific YOLO model for real detections.
