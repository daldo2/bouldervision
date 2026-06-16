# Models

Trained model weights live in this directory. **Weights are not committed to
git** (see the root `.gitignore`) because they are large and reproducible.

## What goes here

| File | Purpose | How to get it |
|------|---------|---------------|
| `yolov8n.pt` | Generic pretrained detector (Phase 0/1 MVP) | Auto-downloaded by Ultralytics on first run |
| `yolov8x-pose.pt` | Pose estimation (Phase 3) | Auto-downloaded by Ultralytics on first run |
| `holds.pt` | **Hold-specific** detector fine-tuned on labeled climbing holds | Produced in Phase 1 (see below) |

The auto-downloaded models land in Ultralytics' cache, but you can also place
them here and reference them by path in `config/settings.yaml`.

## Training a hold-specific detector (Phase 1)

The MVP uses generic `yolov8n.pt`, which was trained on COCO and does **not**
know what a climbing hold is. To get real detections you fine-tune a model that
learns the `hold`/`volume` class. Color is NOT learned here — our HSV classifier
(`src/utils.py`) colors each detected hold, so the model stays simple.

We have no local GPU, so we train in the cloud (free Colab GPU) using a public
hold dataset, then optionally fine-tune on our own gym photos (the "hybrid"
approach).

### Step 1 — base model on a public dataset (cloud)
Run **`notebooks/02_train_holds.ipynb`** on Google Colab (Runtime → GPU). It:
downloads the public *Climbing Holds and Volumes* dataset from Roboflow,
trains `yolov8s`, evaluates, and lets you download `best.pt`.

You need a free Roboflow **Private API key** (Settings → API Keys). It is
entered via a hidden prompt in the notebook — never hard-code it or commit it.

### Step 2 — install the trained model locally
1. Move the downloaded `best.pt` here as `models/holds.pt`.
2. Update `config/settings.yaml`:
   ```yaml
   models:
     hold_detector: "models/holds.pt"
   ```
3. Run: `python -m src.hold_detector data/input/<photo>.jpeg`

### Step 3 — fine-tune on our gym (optional, improves accuracy)
Annotate ~30–50 of our own wall photos in Roboflow, add them to the dataset,
and re-run the notebook to fine-tune to our specific walls.

> On a GPU machine you can use the script form instead:
> `python scripts/train_holds.py --data path/to/data.yaml --epochs 100`
