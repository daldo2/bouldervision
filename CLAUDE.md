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
  `python -m src.video_pipeline <vid> --out out.mp4`. Pose via **RTMPose Wholebody**
  (133-kpt; `models.rtmpose_wholebody: true`) — uses real big-toe / fingertip
  **contact points** instead of extrapolating from ankle/wrist; keeps the 17 body
  joints for the skeleton, ignores face. RTMPose beat yolov8 & MediaPipe in a
  **quantitative A/B** (`scripts/ab_pose_backends.py`: contact-joint PCK 85% vs 69%).
  Contacts: per-limb point + velocity gate + engage dwell + sticky resolution +
  occlusion detection + mode filter. **Contact distances are scale-invariant** —
  expressed as a fraction of torso size (`*_frac` in config), so behaviour is the
  same at any resolution / climber size. Overlay: thin skeleton, small body dots,
  hand = 3 finger rings, foot = heel+toe dots. Outputs: annotated video, contact
  summary, **move sequence ("beta")**. Extras: `--route-color`, `--stabilize`.

### Pending / next
1. **Richer-class detector retrain — TOP PRIORITY (next session).** Holds vs
   volumes/markers/downclimb/tape are the current limiter (on slabs everything
   snapped to one box). User shooting ~50–100 gym photos 2026-06-18. Workflow:
   `scripts/export_for_annotation.py` → correct in Roboflow → fine-tune.
   See `docs/annotation-retrain.md`.
2. **POSE: evaluate Wholebody off-the-shelf, fine-tune only if needed.** The
   pretrained 133-kpt model now gives toe/finger points (no training). Eval harness:
   `scripts/eval_pose.py` (PCK vs hand-corrected GT); annotation bootstrap via
   `scripts/export_pose_frames.py` + `pack_cvat_dataset.py` + `cvat_skeleton_label.py`
   (CVAT). If a custom model is needed, use a lean ~23-kpt schema (no face),
   likely yolov8-pose (easy custom kpts + mobile export).
3. **Phase 4 (force estimation)** — deferred; has pose+contacts timeline as input.

### Environment notes
- No local GPU. `.venv` (Python 3.14): torch+ultralytics, rtmlib+onnxruntime, mediapipe.
- RTMPose on CPU ≈ 10–15 min per ~2 min clip; render ONE video at a time (don't
  fan out heavy jobs — it oversubscribes the 16 cores).
- Real gym photos & videos stay local (gitignored); only synthetic `wall.jpg` is tracked.
