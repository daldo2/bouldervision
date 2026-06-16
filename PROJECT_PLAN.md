# BoulderVision — Project Plan

A phased roadmap for building BoulderVision, from a single-image proof of
concept to a mobile app backed by a video-analysis API.

Each phase lists its **goal**, **concrete tasks**, the **deliverable** that
marks it "done", and the **key risks / unknowns** to watch.

---

## Guiding principles

1. **Always have something that runs.** Every phase ends with a script or
   endpoint you can execute, even if the output is rough.
2. **Defer training as long as possible.** Use pretrained models and classical
   CV (HSV color math) before investing in data labeling and GPU training.
3. **Separate detection from interpretation.** "Where are the holds" (vision)
   is kept apart from "which holds form a route" (logic), so each can be
   improved independently.
4. **Measure before optimizing.** Don't reach for a bigger model or force
   model until the simple version is demonstrably the bottleneck.

---

## Phase 0 — Setup & Learning *(current)*

**Goal:** A clean, runnable repository scaffold and a working dev environment.

**Tasks**
- [x] Create repository structure (`src/`, `data/`, `config/`, `tests/`, …).
- [x] Pin dependencies in `requirements.txt`.
- [x] Add `config/settings.yaml` for model paths, thresholds, and color ranges.
- [x] Write the Phase 1 `hold_detector.py` proof of concept.
- [x] Run the detector on one test image and eyeball the output.
- [ ] Read up on YOLOv8 (Ultralytics) and the HSV color model.

**Deliverable:** `python src/hold_detector.py <image>` produces
`data/output/output.jpg` with labeled boxes and prints a detection summary.

**Risks / unknowns**
- Pretrained YOLOv8n knows nothing about climbing holds → low recall expected.
  That's fine for Phase 0; it proves the *pipeline* works.

---

## Phase 1 — Hold Detection MVP

**Goal:** Reliably find holds in a still image and label each by color.

**Tasks**
- Run pretrained `yolov8n` on climbing-wall images to establish a baseline.
- **Collect & label a small hold dataset** (~200–500 boxes) using a tool such
  as [Roboflow](https://roboflow.com) or [CVAT](https://www.cvat.ai/). One
  class is enough for now: `hold`.
- Fine-tune `yolov8n` (or `yolov8s`) on the labeled holds.
- Build the HSV color classifier (no training needed): crop each detected box,
  find the dominant hue, map it to a named color.
- Tune HSV ranges in `config/settings.yaml` against real photos.

**Deliverable:** An image with one colored bounding box per hold, each labeled
with its detected color and confidence.

**Risks / unknowns**
- Holds are visually diverse (jugs, crimps, slopers, volumes) → may need more
  labeled data than expected.
- Lighting and wall color shift HSV values. Mitigation: classify on the most
  saturated pixels and allow per-gym calibration.
- Tiny/distant holds may fall below YOLO's resolution → consider tiling.

---

## Phase 2 — Route Extraction

**Goal:** Turn a bag of colored holds into discrete routes ("problems").

**Approach (implemented):** Instead of matching holds to a fixed global palette
(which doesn't generalize across gyms — different lighting/white balance/color
schemes), we read each hold's actual color as a CIELAB vector and **cluster
those vectors per image**. Route grouping is relative, so it survives lighting
drift. Fixed names only label the clusters for humans. See
`src/route_extractor.py` (`cluster_by_color` → `split_by_position`).

**Tasks**
- [x] Read each hold's dominant color robustly (Lab median, ignoring chalk /
  shadows / highlights) — `utils.dominant_color_lab`.
- [x] Gray-world white balance to reduce gym-to-gym lighting cast.
- [x] Cluster holds by color (DBSCAN in Lab) into route candidates.
- [x] Spatial split (DBSCAN on centroids) for two routes sharing a color.
- [x] Name each route by nearest reference color; full `extract_routes` + tests.
- [ ] Order holds bottom-to-top to suggest a climbing sequence.
- [ ] Visualize each route distinctly (connected line / number labels).
- [ ] Tune `color_eps` / `spatial_eps_px` on real detections once the model lands.

**Deliverable:** An image where each route is drawn as a labeled, connected set
of holds.

**Risks / unknowns**
- Two different problems can use the *same* color on the same wall → color
  alone is ambiguous. Spatial clustering is the primary mitigation.
- "Volumes" (large wooden features) are often shared by all routes and may need
  special handling.

---

## Phase 2.5 — Wall furniture & real-photo robustness

**Why:** On real gym photos the detector returns boxes for things that are NOT
route holds, which pollute route grouping:

| Thing | What it is | Signal |
|-------|-----------|--------|
| **Volumes** | big bolt-on shapes, neutral color, shared by routes | model class `1`; also large box area |
| **Start/zone markers** | small solid-black circular stickers next to holds | small + square + dark + near-neutral |
| **Difficulty tape** | thin strips marking a grade | extreme box aspect ratio |
| **Down-climb holds** | descent holds, white/black, painted **down-arrow** | the arrow glyph; usually a vertical column at the wall edge |

**Stopgap (implemented):** `src/detection_filter.py` sorts raw detections into
`hold / volume / marker / tape` using the model's class id plus geometry/color
heuristics; only holds feed route extraction. Tunable under `filter:` in
`config/settings.yaml`. Set-aside items are drawn in neutral colors and counted
in the summary so misclassifications are visible. **Limits:** when YOLO splits
one volume into several sub-boxes, the size test misses them; marker/tape
heuristics are fragile per-gym; down-climb holds are **not** handled here.

**Proper fix — retrain with a richer class set (depends on a labeled dataset):**
- [ ] Annotate ~100–200 photos of **our** walls with classes:
  `hold`, `volume`, `downclimb`, `marker`, `tape`.
  (Bootstrap labels from `best.pt` predictions, then correct in Roboflow.)
- [ ] Retrain YOLOv8 on Colab (same flow that produced `best.pt`).
- [ ] Drop classes other than `hold` from route grouping by *class*, not
  heuristics; surface `volume`/`downclimb` as their own overlays.
- [ ] Re-tune / retire the `filter:` heuristics once the model is reliable.

**Perspective (angled photos):** holds far down an angled wall foreshorten, so
pixel-space spatial clustering merges or over-splits routes.
- [x] Scale-adaptive spatial split: measure hold spacing in *hold-widths*
  (`routes.adaptive_spatial` + `spatial_scale_eps`), not raw pixels, so the
  threshold is resolution- and distance-independent. `split_by_position_adaptive`
  in `route_extractor.py`. On a front-on wall this also fixed fragmentation
  (15 singleton "routes" → 4 coherent colour routes).
- [ ] Medium term: rectify the wall plane via homography (detect/select 4 wall
  corners → warp to a frontal view) before spatial clustering. Needed for
  *extreme* angles (e.g. a whole wall shot down a corridor) where holds project
  into one compact band and 2D clustering is inherently ambiguous.
- [ ] Long term: prefer near-frontal shots in the capture guidance.

**Deliverable:** Route maps from real, imperfect gym photos with volumes,
markers, tape and down-climb holds excluded from routes.

---

## Phase 3 — Pose Estimation on Video

**Goal:** Track the climber's body through a video and tie limbs to holds.

**Tasks**
- [x] Per-frame keypoint extraction wired (`pose_estimator.estimate_pose`,
  lazy model load).
- [x] Smooth keypoints across frames — confidence-weighted moving average
  (`smooth_keypoint_sequence`).
- [x] Decide which holds hands/feet touch (keypoint-to-box edge proximity,
  `touched_holds`) + unit tests.
- [x] Per-frame timeline + aggregation (`video_pipeline.analyze_video`,
  `summarize_contacts`, `holds_used`).
- [ ] Validate + tune `touch_distance_px` on real footage (pending videos).
- [ ] Polish the annotated-video overlay (skeleton bones, hold labels).

**Status:** All model-free logic is implemented and tested offline. The video
loop and pose model run as soon as real footage + a pose model are available
(yolov8x-pose auto-downloads).

**Deliverable:** An annotated video with the skeleton overlaid and the
currently-touched holds highlighted.

**Risks / unknowns**
- Occlusion: the climber's body hides limbs and holds → keypoint confidence
  drops. Use confidence thresholds and temporal smoothing.
- "Touching" is fuzzy — proximity ≠ weight-bearing contact. Phase 4 refines.
- `yolov8x-pose` is heavy; may need GPU or a smaller pose model for speed.

---

## Phase 4 — Force Estimation *(research phase)*

**Goal:** Estimate how much load each limb carries at a given moment.

**Tasks**
- Literature review on biomechanical force estimation from monocular pose
  (static equilibrium models, contact-force optimization).
- Build a first-order physics model: estimate the body center of mass (COM)
  from keypoints + segment mass ratios, apply gravity, and solve a simple
  static balance across the contact points (hands/feet on holds).
- Output an estimated load percentage per limb, per frame.
- Sanity-check against intuition (e.g. on an overhang, arms carry more).

**Deliverable:** Per-frame, per-limb load estimates (JSON + overlay).

**Risks / unknowns**
- Monocular video gives no true depth → forces are under-constrained. Expect
  estimates, not measurements. Be explicit about uncertainty.
- Segment mass ratios are population averages, not the specific climber.
- This phase is exploratory; the "model" may stay qualitative (high/med/low).

---

## Phase 5 — Backend API

**Goal:** Expose the pipeline as a service.

**Tasks**
- FastAPI app with `POST /analyze` accepting a video upload.
- Run analysis asynchronously via a job queue (e.g. Celery + Redis, or RQ).
- `GET /jobs/{id}` returns status + results JSON when complete.
- Store outputs (annotated media, JSON) in object storage.
- Containerize with Docker; document GPU vs CPU deployment.

**Deliverable:** A running API: upload a video, poll for a JSON result.

**Risks / unknowns**
- Video analysis is slow and memory-hungry → strict job timeouts and size
  limits needed.
- GPU availability/cost in deployment.

---

## Phase 6 — Mobile App

**Goal:** Let a climber record/upload a video and view the analysis.

**Tasks**
- React Native app (Expo to start).
- Record or pick a video → upload to the Phase 5 API.
- Poll the job endpoint; show progress.
- Render results: route overlay, pose, per-limb load timeline.

**Deliverable:** A mobile app that takes a climbing video and shows analysis.

**Risks / unknowns**
- Mobile upload of large videos over cellular → compress/trim client-side.
- Rendering overlays smoothly on-device.

---

## Tech stack summary

| Layer | Choice | Why |
|-------|--------|-----|
| Detection / pose | Ultralytics YOLOv8 | Strong pretrained models, easy fine-tuning |
| Classical CV | OpenCV + NumPy | HSV color classification, drawing |
| Config | YAML | Human-editable thresholds & color ranges |
| API | FastAPI | Async, typed, fast to build |
| Queue | Celery/RQ + Redis | Offload slow video jobs |
| Mobile | React Native (Expo) | Cross-platform, single codebase |

## Suggested order of attack

Phases are mostly sequential, but Phase 2 (route extraction) and Phase 3 (pose)
are independent and can be developed in parallel once Phase 1 produces reliable
hold boxes.
