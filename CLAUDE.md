# BoulderVision — context for Claude

Computer-vision analysis of bouldering videos: detect climbing holds, classify
them by color, group same-colored holds into routes, track climber pose, and
(later) estimate per-limb force. Full roadmap in `PROJECT_PLAN.md`.

## How to run

The project uses a local virtualenv at `.venv` (Python 3.14). Installed ML deps:
torch + ultralytics (detection, yolov8 pose), **rtmlib + onnxruntime** (RTMPose
pose backend — the default), mediapipe (alternate pose backend). NOT system-installed.

```bash
source .venv/bin/activate
python -m src.hold_detector data/input/wall.jpg            # Phase 1: detect + color
python -m src.hold_detector data/input/<img>.jpg --routes  # Phase 1+2: routes (also --holds)
python -m src.video_pipeline data/input/<vid>.mp4 --out out.mp4  # Phase 3: pose + contacts + beta
pytest -q                                                  # 81 offline tests
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

## Status (as of 2026-06-17)

GitHub: https://github.com/daldo2/bouldervision (public). Branch: `master`. **81 tests**
(offline; model runs are separate). The trained hold detector `models/best.pt`
arrived and is wired (`models.hold_detector: best.pt`) — detection works well.

- **Phase 1 (holds + color):** done. Real detector in use.
- **Phase 2 (routes):** done & tuned. Per-image CIELAB clustering, spatial split,
  hue-aware naming. Palette gained **pink + cyan** (the gym uses them). Color
  threshold tuned via the eval harness — `scripts/eval_colors.py` + `eval/labels.yaml`
  (recall 0.70→0.90). Run `--routes`; `--holds` draws per-hold colors without grouping.
- **Phase 2.5 (real-photo robustness):** `src/detection_filter.py` sets aside
  volumes/markers/tape (only holds form routes); adaptive spatial split (hold-width
  relative); homography rectification for angled walls (`--corners`). See
  `docs/annotation-retrain.md` for the planned richer-class detector retrain.
- **Phase 3 (video: pose + contacts + beta):** WORKING on real footage.
  `python -m src.video_pipeline <vid> --out out.mp4`. Pose via **RTMPose** (default,
  top-down, best for small/odd-posed climbers — beat yolov8 & MediaPipe in an A/B);
  switch with `models.pose_backend`. Contacts: extremity points + velocity gate +
  sticky resolution + occlusion/implausible-keypoint detection + mode filter.
  Outputs: annotated video, contact summary, and a **move sequence ("beta")**.
  Extras: `--route-color` (analyze one route), `--stabilize` (handheld).

### Pending / next
1. **Richer-class detector retrain** — user shooting ~50–100 more gym photos
   (~2026-06-18). Workflow: `scripts/export_for_annotation.py` → correct in
   Roboflow (volume/downclimb/marker classes) → fine-tune. See `docs/annotation-retrain.md`.
2. **Climbing-fine-tuned POSE model** — the deeper accuracy fix for extreme poses
   (RTMPose got us most of the way; off-the-shelf 2D pose can't fully nail a wide
   split). Needs keypoint annotation on climbing frames.
3. **Phase 4 (force estimation)** — deferred; now has pose+contacts timeline as input.

### Environment notes
- No local GPU. `.venv` (Python 3.14): torch+ultralytics, rtmlib+onnxruntime, mediapipe.
- RTMPose on CPU ≈ 10–15 min per ~2 min clip; render ONE video at a time (don't
  fan out heavy jobs — it oversubscribes the 16 cores).
- Real gym photos & videos stay local (gitignored); only synthetic `wall.jpg` is tracked.
