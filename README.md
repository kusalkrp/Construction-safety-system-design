# Construction Safety Monitor

A two-stage YOLOv8 system that detects PPE compliance violations on construction sites in real time.
It flags what it knows, admits what it doesn't, and escalates only when confidence warrants it —
designed for trustworthiness over raw accuracy.

---

## The Core Question

> *"Is this worker compliant, non-compliant, or is the system unable to tell?"*

Every baseline system outputs `safe / unsafe`. This system outputs a **structured violation report**,
a **0–100 compliance score**, and **human-readable alerts** — because that is what a site manager
actually needs.

---

## System Architecture

```
Input Frame
    │
    ▼
Stage 1: YOLOv8m — Person Detector  (COCO pretrained, fine-tuned)
    │  outputs: person bounding boxes
    ▼
Stage 2: YOLOv8m — PPE Detector     (per-person crop)
    │  classes: helmet_on, no_helmet, vest_on, no_vest, mask_on
    ▼
Stage 3: SafetyChecker              (Rules 1–6, rule_confidence)
    │  outputs: ViolationReport per worker
    ▼
Stage 4: SiteScorer                 (0–100 compliance score)
    │  outputs: score + colour band
    ▼
Output: annotated frame + JSON report + compliance score
```

**Why two-stage:** PPE classification on per-person crops is easier to train with limited data.
Failure modes are isolated — person detection and PPE detection can be debugged independently.
The compliance logic is transparent and independently testable (no YOLO calls inside `SafetyChecker`).

---

## Safety Rules

Six formal rules are defined in [`SAFETY_RULES.md`](SAFETY_RULES.md) and
machine-readable in [`rules.yaml`](rules.yaml).

| Rule | Trigger | Severity |
|---|---|---|
| 1 — No helmet | `no_helmet` conf ≥ 0.50, bbox ≥ 40px | CRITICAL |
| 1 (elevated) | same + upper 40% of frame | CRITICAL-ELEVATED |
| 2 — No vest (outdoor) | `no_vest` conf ≥ 0.50, outdoor scene | HIGH |
| 2 — No vest (indoor) | `no_vest` conf ≥ 0.50, indoor scene | WARNING |
| 3 — Partial compliance | PPE conf 0.35–0.65 | WARNING → review queue |
| 4 — Far-field worker | Person bbox height < 40px | UNVERIFIABLE |
| 5 — Occlusion gap | Person detected, no PPE overlap | WARNING → review queue |
| 6 — Crowd non-compliance | ≥ 4 workers, ≥ 50% violating | SITE ALERT |

---

## Compliance Scoring

Score range 0–100, updated per frame:

```
score = 100
      − (critical_violations × 25)
      − (high_violations × 15)
      − (warning_violations × 5)
      × crowd_multiplier (1.3 if Rule 6 triggers)
      × temporal_decay   (violations > 60 s decay 50%/min)
```

| Score | Band | Meaning |
|---|---|---|
| 80–100 | COMPLIANT (green) | Normal operations |
| 60–79 | CAUTION (amber) | Supervisor attention needed |
| 40–59 | AT RISK (orange) | Active intervention required |
| 0–39 | CRITICAL (red) | Stop-work consideration |

---

## Dataset

Four sources merged into one 6-class dataset:

| Source | Images | Format | Key classes contributed |
|---|---|---|---|
| Roboflow Construction Site Safety (base) | ~3,000 | YOLO txt | All 6 classes |
| Kaggle PPE Kit Detection | ~1,263 | YOLO txt | no_helmet, no_vest, helmet_on, vest_on |
| Kaggle Safety Vests Detection | ~3,897 | Pascal VOC CSV → converted | vest_on, no_vest |
| Roboflow Hard Hat Sample (scaffolding) | ~2,192 | YOLO txt | helmet_on, vest_on, person |

- **Class remapping:** each source remapped to 6-class schema (see `dataset/mappings/`)
- **Violation ratio:** ~60% violation annotations intentionally — optimises for recall on safety-critical classes
- **Split:** 75% train / 15% val / 10% test
- **Offline augmentation:** shadow overlay, fog/haze, Gaussian blur, brightness/contrast, Gaussian noise, horizontal flip — train split only, factor ×2

Full documentation: [`docs/DATASET.md`](docs/DATASET.md)

---

## Training

- **Model:** YOLOv8m pretrained on COCO 2017 — fine-tuned end-to-end
- **Hardware:** NVIDIA RTX 3060 6 GB VRAM
- **Config:** `imgsz=640`, `batch=8`, `epochs=50`, `AdamW`, `lr0=0.001`, `patience=15`
- **Notebook:** [`training/notebooks/training.ipynb`](training/notebooks/training.ipynb) — run all cells top-to-bottom

---

## Evaluation Results

> Populated after training. See [`training/notebooks/training.ipynb`](training/notebooks/training.ipynb) Section 7–9 and [`docs/per_class_metrics.csv`](docs/per_class_metrics.csv).

| Class | mAP@0.5 | Recall | Precision |
|---|---|---|---|
| `no_helmet` | — | — | — |
| `no_vest` | — | — | — |
| `helmet_on` | — | — | — |
| `vest_on` | — | — | — |
| `person` | — | — | — |
| `mask_on` | — | — | — |
| **Overall** | — | — | — |

---

## Known Failure Modes

| Failure | What happens | Production fix |
|---|---|---|
| Night / low-light | Detection confidence drops below threshold — goes to review queue, no false alert | IR-capable cameras or supplementary lighting |
| Far-field workers (< 40px bbox) | Flagged `UNVERIFIABLE` — no false compliance, no false alert | PTZ camera with auto-zoom |
| Partial occlusion | Flagged `OCCLUDED_WORKER` — human review recommended | Multi-angle CCTV |
| Dark helmet in low contrast | `no_helmet` false negative — partially mitigated by lower production threshold (0.40) | Dedicated low-contrast training data |
| Dense crowds / overlapping boxes | Cluster flagged `CROWD_DETECTION_AMBIGUOUS` | Stereoscopic cameras |
| Night in floodlit scenes | Model untested in this common condition — acknowledged limitation | Targeted floodlit dataset collection |
| Geography bias | Dataset is global — PPE styles may vary regionally | Region-specific collection if deploying locally |

---

## Quickstart

### Prerequisites

- Python 3.10+
- NVIDIA GPU (RTX 3060 or better recommended for training)
- Docker (for API serving)
- [Roboflow account](https://roboflow.com) — free tier is sufficient

---

### Step 1 — Install dependencies

```bash
cd construction-safety-system-design
pip install -r requirements.txt
pip install roboflow label-studio jupyter
```

---

### Step 2 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your Roboflow API key:

```
ROBOFLOW_API_KEY=your_actual_key_here
MODEL_WEIGHTS_PATH=runs/train/weights/best.pt
```

---

### Step 3 — Download the base dataset (Roboflow)

```bash
python scripts/download_dataset.py
```

Downloads Roboflow Construction Site Safety (latest version), remaps 10 → 6 classes,
and writes to `dataset/raw/roboflow_base/`.

---

### Step 4 — Download Kaggle datasets

Install the Kaggle CLI and place your API key at `C:\Users\YOUR_USERNAME\.kaggle\kaggle.json`
([get it here](https://www.kaggle.com/settings) → API → Create New Token).

```bash
pip install kaggle

# Dataset 1: PPE Kit Detection (~1,263 images — no_helmet / no_vest violations, YOLO format)
kaggle datasets download -d ketakichalke/ppe-kit-detection-construction-site-workers \
    -p dataset/raw/kaggle_ppe_kit/ --unzip

# Dataset 2: Safety Vests Detection (~3,897 images — vest_on / no_vest, Pascal VOC CSV format)
kaggle datasets download -d adilshamim8/safety-vests-detection-dataset \
    -p dataset/raw/kaggle_safety_vests/ --unzip
```

---

### Step 5 — Download Roboflow scaffolding extension

Already downloaded to `dataset/raw/roboflow_scaffolding/` (2,192 images — helmet_on, vest_on, person).

---

### Step 6 — Mapping configs are pre-filled (audit complete)

All four mapping files in `dataset/mappings/` have been audited against the actual downloaded files:

| Dataset | Format | Mapping file | Notes |
|---|---|---|---|
| `roboflow_base/` | YOLO txt | `roboflow_base.yaml` | Identity mapping, already 6-class |
| `kaggle_ppe_kit/` | YOLO txt | `kaggle_ppe_kit.yaml` | 11 classes → 6, class IDs verified |
| `kaggle_safety_vests/` | Pascal VOC CSV | `kaggle_safety_vests.yaml` | CSV — use `convert_csv_to_yolo.py`, not `remap_labels.py` |
| `roboflow_scaffolding/` | YOLO txt | `roboflow_scaffolding.yaml` | 8-class (corrupted data.yaml) — mapping verified from label files |

No manual editing needed. Proceed to Step 7.

---

### Step 7 — Convert and remap each dataset to the 6-class schema

```bash
# Dataset 1: Roboflow base — identity remap (already 6-class)
python scripts/remap_labels.py \
    --source  dataset/raw/roboflow_base/ \
    --mapping dataset/mappings/roboflow_base.yaml \
    --output  dataset/remapped/roboflow_base/

# Dataset 2: Kaggle PPE Kit — YOLO format, 11→6 class remap
python scripts/remap_labels.py \
    --source  dataset/raw/kaggle_ppe_kit/ \
    --mapping dataset/mappings/kaggle_ppe_kit.yaml \
    --output  dataset/remapped/kaggle_ppe_kit/

# Dataset 3: Safety Vests — CSV format, convert first (no remap_labels.py needed)
python scripts/convert_csv_to_yolo.py \
    --source  dataset/raw/kaggle_safety_vests/ \
    --output  dataset/remapped/kaggle_safety_vests/

# Dataset 4: Roboflow scaffolding — YOLO format, 8→6 class remap
python scripts/remap_labels.py \
    --source  dataset/raw/roboflow_scaffolding/ \
    --mapping dataset/mappings/roboflow_scaffolding.yaml \
    --output  dataset/remapped/roboflow_scaffolding/
```

**After each run:** check the per-class log output. If `no_helmet` or `no_vest` shows 0 annotations,
the mapping YAML has a wrong index — fix it and re-run before continuing.

---

### Step 8 — Merge all sources into one dataset

```bash
python scripts/merge_datasets.py \
    --sources \
        dataset/raw/roboflow_base/ \
        dataset/remapped/kaggle_ppe_kit/ \
        dataset/remapped/kaggle_safety_vests/ \
        dataset/remapped/roboflow_scaffolding/ \
    --output dataset/merged/ \
    --val-split 0.15 \
    --test-split 0.10
```

Expected result: ~8,500 total images, `no_helmet` ≥ 1,200 and `no_vest` ≥ 2,000 train annotations.

---

### Step 9 — Validate the merged dataset

```bash
python scripts/validate_dataset.py \
  --data dataset/merged/data.yaml \
  --save-report
```

Fix any issues before augmentation. Do not proceed if validation reports class ID errors or zero-count violation classes.

---

### Step 10 — Augment training split

```bash
python scripts/augment_dataset.py \
  --data dataset/merged/data.yaml \
  --factor 2
```

Applies 6 offline augmentations to the train split only. Val and test splits are never augmented.
Expected: ~6,400 train images → ~12,800 after augmentation.

---

### Step 12 — Train (in the notebook)

```bash
jupyter notebook training/notebooks/training.ipynb
```

Open in your browser and **run all cells top-to-bottom**. Training takes ~1–3 hours on an RTX 3060.

The notebook covers:
- Baseline model (YOLOv8n, no custom extension) for comparison
- Full training (YOLOv8m + custom extension + augmentation)
- Training curves and loss plots inline
- Per-class evaluation: mAP@0.5, mAP@0.5:0.95, precision, recall, F1, confusion matrix
- Per-class commentary (what the numbers mean, real-world cost, threshold recommendation)
- Baseline vs full model comparison table and chart
- Failure case gallery (auto-detected false negatives on test set)
- Precision-recall threshold trade-off discussion

Trained weights are saved to `runs/train/weights/best.pt`.

---

### Step 13 — Upload weights to GitHub Release

```bash
gh release create v1.0 runs/train/weights/best.pt \
  --title "v1.0 — initial training" \
  --notes "YOLOv8m trained on Roboflow v12 + custom extension"
```

Weights are stored as a GitHub Release asset, not in the repository.
To use on another machine: download the release asset and set `MODEL_WEIGHTS_PATH` in `.env`.

---

### Step 14 — Run the Gradio demo

```bash
python demo.py
```

Open `http://localhost:7860`. Upload a construction site image to get:
- Annotated scene with green (compliant), red (violation), and yellow (unverifiable) boxes
- Compliance score with colour band
- Full structured violation report

---

### Step 15 — Run the API (optional)

```bash
mkdir serving/weights
cp runs/train/weights/best.pt serving/weights/

docker-compose -f serving/docker-compose.yml up
```

API available at `http://localhost:8000`.

```bash
# Health check
curl http://localhost:8000/health

# Analyse an image
curl -X POST http://localhost:8000/analyse \
  -F "file=@path/to/image.jpg" | python -m json.tool
```

---

## Running Tests

```bash
pytest tests/ -v
```

29 unit tests covering all 6 safety rules, score formula, temporal decay, and edge cases.

---

## Repository Structure

```
construction-safety/
├── scripts/           # data preparation — download, scrape, label, augment, validate
├── training/
│   ├── configs/       # train_config.yaml
│   └── notebooks/     # training.ipynb — sole training entry point
├── inference/         # SafetyChecker, SiteScorer, Annotator, pipeline
├── serving/           # FastAPI + Dockerfile + docker-compose
├── tests/             # pytest unit tests
├── docs/              # DATASET.md, plots, metrics CSVs
├── demo.py            # Gradio demo
├── rules.yaml         # all thresholds — no magic numbers in inference code
└── SAFETY_RULES.md    # formal rule specification
```

---

## Design Decisions & Trade-offs

| Decision | Choice | Reason |
|---|---|---|
| Two-stage pipeline | Person → PPE-on-crop | Isolates failure modes; easier to train PPE classifier with limited data |
| YOLOv8m over nano | Medium model | Better small-object detection for far-field PPE; fits 6 GB VRAM |
| COCO pretrained weights | Transfer learning | Person class already in COCO; 3× faster convergence |
| 60% violation ratio | Intentional imbalance | Maximises recall on safety-critical classes — false negative > false positive in cost |
| rule_confidence composite | Detection conf + bbox size + edge proximity | More trustworthy than raw YOLO score alone |
| Heuristic scene classifier | Sky-pixel HSV detection | Zero model weight, sufficient accuracy for Rule 2 risk modulation |
| Compliance score (0–100) | Score over binary flag | Site managers think in scores and trends, not event counts |
| Lower production threshold (0.40) | For violation classes only | Asymmetric cost: false negative = injury; false positive = supervisor time |

---

## Future Work

- Night-time / floodlit scene training data (IR camera input)
- PTZ auto-zoom integration for far-field workers (Rule 4 production upgrade)
- Worker re-identification across frames for persistent violation tracking
- True zone mapping (visitor vs. active worker) beyond proximity heuristic
- Region-specific dataset collection for Sri Lanka deployment context
- Real-time RTSP stream support for live CCTV integration
