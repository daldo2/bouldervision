# Annotation & retrain — richer-class hold detector

**Goal:** retrain the detector with the classes `hold, volume, downclimb, marker,
tape` so volumes/markers/down-climb holds are recognized by the *model* instead
of by fragile heuristics. This is the durable fix for "volumes caught as holds".

**Why we need it (evidence):** the current `best.pt` only has hold/volume and it
labels big white volumes as **class 0 = hold** (largest box only ~3.3% of the
frame, below the size heuristic). No post-processing can fix that safely — the
model is the limiter.

---

## Step 0 — collect photos
Aim for **~50–100 photos** of the gym (more = better), varied: different walls,
angles, lighting. Drop them in `data/input/`. Even correcting a modest set helps,
because we **fine-tune** the already-strong `best.pt` rather than train from zero.

Tips for good route photos: shoot **front-on**, in **sections** (not one ultra-wide
frame), evenly lit. That also helps color naming downstream.

## Step 1 — generate bootstrap labels
```bash
source .venv/bin/activate
python scripts/export_for_annotation.py
```
This pre-labels every photo (detection + hold/volume/marker/tape filter) into
`data/annotation/{images,labels}/` + `data.yaml`, YOLOv8 format. You **correct**
these, not draw from scratch. (Bundle is gitignored — it copies private photos.)

## Step 2 — upload to Roboflow
1. Roboflow → **Create Project → Object Detection**.
2. **Upload** `data/annotation/images/` **and** `data/annotation/labels/` together
   → it auto-detects YOLOv8 and shows the pre-drawn boxes.
3. Class names come from `data.yaml`: `hold, volume, downclimb, marker, tape`.

## Step 3 — correct (the actual work, in priority order)
- **Volumes the model called `hold` → reclassify to `volume`** (the main win —
  especially the big white ones).
- **Add `downclimb`** boxes on descent holds (the ones with a painted down-arrow).
  These are not auto-detected; draw them by hand.
- Fix any stray `marker` / `tape`.
- Spot-check that real holds are still `hold`.

## Step 4 — export & train
1. Roboflow → **Generate** a version → **Export → YOLOv8** (download or copy the
   snippet).
2. Colab (same free T4 notebook used for `best.pt`): **fine-tune from `best.pt`**,
   `nc=5`, ~100 epochs. (Fine-tune, do NOT train from scratch — keeps the strong
   hold detection.)

## Step 5 — install the new model
1. Save the resulting `best.pt` as `models/holds.pt`.
2. Point `models.hold_detector` in `config/settings.yaml` at it.
3. The class ids matter downstream — `detection_filter.py` keys off class `1`
   for volumes (`filter.volume_class`). Keep `volume = 1`, or update that config.
4. Re-run and re-check:
   ```bash
   python -m src.hold_detector data/input/<photo>.jpeg --routes
   python scripts/eval_colors.py        # color naming shouldn't regress
   ```

## After retrain — what should improve
- Volumes set aside correctly (no more big-white-volume-as-holds).
- Markers/tape from the model, so the heuristics in `filter:` can be relaxed.
- `downclimb` holds available to exclude from routes (new capability).
