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

- GitHub: https://github.com/daldo2/bouldervision (public). Branch: `master`. 37 tests, all offline.
- **Phase 1 (holds + color):** done & wired. HSV palette expanded for real walls
  (added orange/white); `scripts/inspect_hsv.py` for tuning.
- **Phase 2 (routes):** done & wired. Holds grouped by clustering their real
  CIELAB colors per image (generalizes across gyms), then split spatially.
  Run with `--routes`. Drawing + bottom-to-top ordering implemented.
- **Phase 3 (pose/contacts):** all model-free logic done & tested — `touched_holds`
  (limb→hold proximity), `smooth_keypoint_sequence`, `analyze_video` loop wired
  (lazy models), `summarize_contacts`/`holds_used`. Needs real footage to run.
- **Full pipeline command:** `python -m src.hold_detector <img> --routes`
  (detect → Lab color → cluster routes → draw). Ready the moment `models/holds.pt` exists.

### Waiting on (external, as of 2026-06-16)
1. **`models/holds.pt`** — user training a YOLOv8 hold detector on Colab (free T4,
   public Roboflow "Climbing Holds and Volumes" dataset, ~100 epochs). When the
   `best.pt` arrives: drop it as `models/holds.pt`, set `hold_detector` in
   `config/settings.yaml`, run on the 8 real gym photos in `data/input/`
   (gitignored), tune `routes.color_eps` / `spatial_eps_px`.
2. **Climbing videos** — user sending in the evening (static camera, whole climber
   in frame, 2-3s empty-wall start frame). Then run Phase 3, tune `pose.touch_distance_px`.

### Environment notes
- No local GPU (training is cloud-only). `.venv` (Python 3.14) has torch+ultralytics.
- Real gym photos & videos stay local (gitignored); only synthetic `wall.jpg` is tracked.

## Next steps
1. (Blocked on `best.pt`) Run Phase 1+2 on real photos; tune color/spatial eps.
2. (Blocked on videos) Run Phase 3 on footage; tune touch distance; polish overlay.
3. Phase 4 (force estimation) — research phase, needs pose data from real videos first.
