# Construction Safety Monitor

A YOLOv8-based system that detects PPE compliance violations on construction sites in real time.
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

Detailed architecture notes are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

```
Input Frame
    │
    ▼
YOLOv8m — Single-pass full-frame detection
    │  classes: helmet_on, no_helmet, vest_on, no_vest, person, mask_on
    ▼
SafetyChecker              (Rules 1–6, rule_confidence, suppression logic)
    │  outputs: ViolationReport per worker
    ▼
SiteScorer                 (0–100 compliance score + temporal decay)
    │  outputs: score + colour band
    ▼
Annotator                  (solid box = person anchor, dashed box = PPE violation location)
    │
Output: annotated frame + structured report + compliance score
```

**Single-pass design:** YOLO runs once on the full frame and returns all boxes (persons + all PPE
classes simultaneously). `SafetyChecker` then associates PPE boxes to person boxes by IoU and
geometry — it never calls YOLO. This isolates the compliance logic and makes it independently
testable. The compliance rules, thresholds, and scoring formula are all independently unit-tested
with 29 tests, none of which touch the model.

**PPE association strategy:**
- Expanded person bbox (60% upward) for IoU search — compensates for torso-anchored detections
- `is_above_person` check — associates helmets sitting above the person box when IoU is near zero
- Conflict suppression — when `helmet_on` and `no_helmet` overlap the same worker, the safe class
  wins. Previously this silently discarded the report; it now surfaces as a `partial_compliance`
  WARNING so the uncertainty is visible to operators
- Anatomical position filter — `no_helmet` center below 60% frame height, or `no_vest` center in
  the top 25% of frame, are rejected as physically impossible mislocalizations (silently discarded)
- Per-class highest-confidence — when multiple boxes of the same class overlap a worker, the
  strongest detection drives the rule rather than an arbitrary iteration-order winner
- Rule 3 deduplication — one `partial_compliance` entry regardless of how many borderline
  detections are present; bonus classes (`mask_on`) are excluded; detections already handled by
  Rules 1/2 (conf ≥ `violation_conf_min`) are not also flagged as partial compliance

---

## Safety Rules

Six formal rules are defined in [`SAFETY_RULES.md`](SAFETY_RULES.md) and
machine-readable in [`rules.yaml`](rules.yaml).

| Rule | Trigger | Severity |
|---|---|---|
| 1 — No helmet | `no_helmet` conf ≥ 0.40, bbox ≥ 40px | CRITICAL |
| 1 (elevated) | same + upper 60% of frame | CRITICAL-ELEVATED |
| 2 — No vest (outdoor) | `no_vest` conf ≥ 0.40, outdoor scene | HIGH |
| 2 — No vest (indoor) | `no_vest` conf ≥ 0.40, indoor scene | WARNING |
| 3 — Partial compliance | Required PPE conf 0.35–0.65; or conflicting safe+violation signal on same worker | WARNING → review queue |
| 4 — Far-field worker | Person bbox height < 40px | UNVERIFIABLE |
| 5 — Occlusion gap | No PPE overlap, OR helmet detected but no vest-class, OR vest detected but no helmet-class | WARNING → review queue |
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

**`rule_confidence` formula** — composite signal used for tiering alerts:
```
rule_confidence = 0.60 × detection_conf
               + 0.25 × min(1.0, bbox_height / 80)
               + 0.15 × (1.0 − edge_proximity_ratio)
```
Tiers: HIGH ≥ 0.70 → immediate alert | MEDIUM ≥ 0.45 → review queue | LOW < 0.45 → logged only

---

## Dataset

Seven sources merged into one 6-class dataset. Full documentation: [`DATASET.md`](DATASET.md).

| Source | Images | Key classes contributed |
|---|---|---|
| Roboflow Construction Site Safety (base) | 194 | All 6 classes — balanced baseline |
| Kaggle PPE Kit Detection | 1,415 | no_helmet, no_vest, helmet_on, vest_on |
| Kaggle Safety Vests Detection | 3,897 | vest_on, no_vest — dominant vest source |
| Roboflow Scaffolding | 1,999 | helmet_on, vest_on, person — elevation scenes |
| Roboflow Worker | 768 | person only — 2,279 person boxes (fixes weak person class) |
| Roboflow No-Helmet | 212 | no_helmet — 384 boxes (closes class shortfall) |
| Roboflow No-Vest | 351 | no_vest — additional violation coverage |
| **Total raw** | **8,836** | |

- **Class remapping:** each source remapped to 6-class schema (see `dataset/mappings/`)
- **Train/val/test split:** 73.9% / 15.0% / 10.0% (6,517 raw train + 13,477 augmented = 20,106 total train)
- **Offline augmentation:** shadow overlay, fog/haze, Gaussian blur, brightness/contrast, Gaussian noise,
  horizontal flip — factor ×2 on train split only
- **Fine-tuning:** train5 (run as `train53`) fine-tuned from train4 weights using raw-only split
  (`data_raw.yaml`, 6,517 images) with online augmentation — avoids 72-hour full-dataset training

---

## Training

Two training runs. The fine-tuned model (`train53`) is the active production model.

| Run | Weights | Config | Purpose |
|---|---|---|---|
| train4 | `runs/train4/weights/best.pt` | YOLOv8m, epoch 50, lr0=0.001, batch=8 | Initial full training |
| train53 | `runs/train53/weights/best.pt` | Fine-tune from train4, epoch 20, lr0=0.0001, batch=16 | Person class fix + no_helmet boost |

- **Model:** YOLOv8m pretrained on COCO 2017
- **Hardware:** NVIDIA RTX 3060 6 GB VRAM (local training)
- **Config:** `imgsz=640`, `AdamW`, `patience=8`, `augment=True`
- **Notebook (train4):** [`training/notebooks/training.ipynb`](training/notebooks/training.ipynb)
- **Notebook (train53):** [`training/notebooks/Retrain with finetuning.ipynb`](training/notebooks/Retrain%20with%20finetuning.ipynb)

---

## Evaluation Results

Evaluated on held-out test set (883 images) using `runs/train53/weights/best.pt`.

### Per-Class Results

| Class | Type | mAP@0.5 | Recall | Precision |
|---|---|---|---|---|
| `helmet_on` | Safe | **95.6%** | 90.2% | 94.8% |
| `no_helmet` | ⚠ Violation | **77.9%** | 70.0% | 69.3% |
| `vest_on` | Safe | **96.5%** | 94.1% | 87.8% |
| `no_vest` | ⚠ Violation | **93.3%** | 88.1% | 84.0% |
| `person` | Neutral | **79.6%** | 83.6% | 70.7% |
| `mask_on` | Safe (bonus) | **71.9%** | 58.5% | 75.0% |
| **Overall** | | **85.8%** | **80.8%** | **80.3%** |

### Comparison: train4 → train53

| Metric | train4 | train53 | Delta |
|---|---|---|---|
| mAP@50 | 79.5% | **85.8%** | +6.3pp |
| Recall | 80.4% | **80.8%** | +0.4pp |
| Precision | 70.7% | **80.3%** | +9.6pp |

### Violation Class Focus

The two safety-critical classes (no_helmet, no_vest) are the most important to evaluate honestly:

- **no_helmet:** 77.9% mAP@50, 70.0% recall — weakest class. Root cause: model sometimes generates
  mislocalized `no_helmet` boxes at torso level when the person is close-up and the helmet is at the
  top edge of frame. Mitigated by above-adjacent suppression and anatomical position filter at inference.
  Genuine hard cases (back-of-head angles, dark helmets in low contrast) remain as known failure modes.

- **no_vest:** 93.3% mAP@50, 88.1% recall — strong. Vest is large and colourful; easier to detect.

---

## Honest Assessment — Where It Works and Where It Doesn't

### Where the model performs well

**Vest detection is reliable.** `vest_on` and `no_vest` both score above 93% mAP@50. Hi-vis vests are large, brightly coloured, and occupy a consistent torso region — the model has learned this robustly across outdoor scenes, scaffolding imagery, and augmented lighting conditions. In normal daylight construction footage, vest compliance is the most dependable signal the system produces.

**Helmet-on detection is strong.** `helmet_on` at 95.6% mAP@50 means the model rarely misses a worker who is wearing a helmet. False negatives on the safe class are low, which matters: the system won't incorrectly flag compliant workers at high volume.

**Crowd-level and site-level scoring holds up.** When multiple workers are present, the compliance score and crowd escalation (Rule 6) aggregate individual detections sensibly. The temporal decay and sliding window logic work as designed for video input — violations that clear up stop affecting the score within the decay window.

**The suppression logic catches the model's own mislocalizations.** The anatomical position filter and above-adjacent suppression were added specifically because the model produces mislocalized `no_helmet` boxes at torso level in close-up shots. These filters prevent those from generating false CRITICAL alerts, and in testing they eliminated the most common false positive category without hurting recall.

---

### Where the model struggles

**`no_helmet` detection is the weakest safety-critical class.** At 70.0% recall, roughly 3 in 10 real no-helmet violations are missed. This is the class that matters most — a missed no-helmet is a missed head injury risk. The root causes are structural: the class has the fewest training annotations (1,835 vs. 7,095 for vest_on), and the model confuses close-up hair and dark backgrounds with a worn helmet. Fine-tuning improved it from ~900 to 1,835 annotations but the shortfall from the 2,000 target remains.

**Person detection fails on close-up, cropped frames.** When a worker fills most of the frame, the model frequently misses the `person` class entirely — because the training data is dominated by full-body and mid-field shots. The fallback path (derive violation reports from PPE boxes directly) handles this at runtime, but it is a workaround, not a fix. It means the system cannot produce worker IDs or apply Rules 4–6 when the person anchor is missing.

**Night, low-light, and indoor fluorescent scenes are unreliable.** No real night-time imagery was available for training. Augmentation simulates some lighting variation, but detection confidence drops significantly below threshold in genuinely dark frames — the system silently falls back to the review queue rather than generating alerts. A site using overhead CCTV in an unlit area at night should not rely on this model without IR cameras.

**`mask_on` is too sparse to trust in isolation.** At 58.5% recall it is only useful as a supporting signal. It is not enforced as a safety rule and should not be.

**Far-field workers are an acknowledged blind spot.** Workers whose bounding box is under 40px tall cannot be assessed for PPE state — the system flags them UNVERIFIABLE. In wide-angle CCTV footage covering a large site, a significant fraction of workers may fall into this category. This is not a model failure but a camera placement and resolution constraint.

**The system has not been tested on regional PPE variants.** All training data is from global public datasets. Locally specific PPE (different helmet colours, alternative vest styles, partial-coverage vests) may produce lower confidence scores and more partial-compliance flags than expected.

---



## Known Failure Modes

| Failure | What happens | Mitigation applied | Production fix |
|---|---|---|---|
| Mislocalized `no_helmet` at torso | False CRITICAL alert on compliant worker | Anatomical position filter + above-adjacent suppression | More close-up training data |
| `helmet_on` / `no_helmet` conflict | Same worker gets both classes | Conflict suppression — safe class wins, surfaces as `partial_compliance` WARNING | Retrain with harder negative examples |
| Far-field workers (< 40px bbox) | Flagged UNVERIFIABLE — no false compliance | Rule 4 (explicit capability flag) | PTZ camera with auto-zoom |
| Person class missed (close-up) | Fallback to PPE-box-only reports | Fallback path + suppression in fallback | Improved person training data |
| Night / low-light | Conf drops below threshold → review queue | Conservative default | IR-capable cameras |
| Partial occlusion | OCCLUDED_WORKER flag | Rule 5 | Multi-angle CCTV |
| Dense crowds / overlapping boxes | Crowd rule fires | Rule 6 escalation | Stereoscopic cameras |

---

## Quickstart

### Prerequisites

- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended — project uses `ai-env`)
- NVIDIA GPU (RTX 3060 or better recommended for training; CPU works for inference only)
- Docker (for API serving only — not required for local inference or demo)

### Step 1 — Create conda environment and install dependencies

```bash
conda create -n ai-env python=3.10 -y
conda activate ai-env
cd construction-safety-system-design
pip install -r requirements.txt
```

Or with an existing Python 3.10+ environment:
```bash
pip install -r requirements.txt
```

### Step 2 — Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```
ROBOFLOW_API_KEY=your_key_here
MODEL_WEIGHTS_PATH=runs/train53/weights/best.pt
RULES_CONFIG_PATH=rules.yaml
LOG_LEVEL=INFO
```

### Step 3 — Download datasets

```bash
python scripts/download_dataset.py          # Roboflow base dataset

# Kaggle datasets (requires kaggle CLI + API key at ~/.kaggle/kaggle.json)
kaggle datasets download -d ketakichalke/ppe-kit-detection-construction-site-workers \
    -p dataset/raw/kaggle_ppe_kit/ --unzip
kaggle datasets download -d adilshamim8/safety-vests-detection-dataset \
    -p dataset/raw/kaggle_safety_vests/ --unzip
```

### Step 4 — Remap and merge

```bash
python scripts/remap_labels.py --source dataset/raw/roboflow_base/     --mapping dataset/mappings/roboflow_base.yaml     --output dataset/remapped/roboflow_base/
python scripts/remap_labels.py --source dataset/raw/kaggle_ppe_kit/     --mapping dataset/mappings/kaggle_ppe_kit.yaml     --output dataset/remapped/kaggle_ppe_kit/
python scripts/remap_labels.py --source dataset/raw/roboflow_scaffolding/ --mapping dataset/mappings/roboflow_scaffolding.yaml --output dataset/remapped/roboflow_scaffolding/

python scripts/merge_datasets.py \
    --sources dataset/remapped/roboflow_base/ dataset/remapped/kaggle_ppe_kit/ \
              dataset/remapped/kaggle_safety_vests/ dataset/remapped/roboflow_scaffolding/ \
              dataset/remapped/roboflow_worker/ dataset/remapped/roboflow_no_helmet/ \
              dataset/remapped/roboflow_no_vest/ \
    --output dataset/merged/ --val-split 0.15 --test-split 0.10

python scripts/make_raw_split.py
python scripts/augment_dataset.py --data dataset/merged/data.yaml --factor 2
```

### Step 5 — Train

```bash
# Initial training (train4)
jupyter notebook training/notebooks/training.ipynb

# Fine-tuning (train53) — run after train4 completes
jupyter notebook "training/notebooks/Retrain with finetuning.ipynb"
```

### Step 6 — Run inference

**Single image (CLI):**
```bash
conda activate ai-env
python -m inference.pipeline --source path/to/image.jpg
```

Output: annotated image written to `output/` + printed violation report + compliance score.

To use a specific weights file:
```bash
python -m inference.pipeline --source path/to/image.jpg --weights runs/train4/weights/best.pt
```

**Gradio demo (browser UI):**
```bash
python demo.py
```

Open `http://localhost:7860`. Upload a construction site image to get:
- Annotated scene: solid box = person anchor, dashed box = PPE violation location
- Compliance score with colour band (COMPLIANT / CAUTION / AT RISK / CRITICAL)
- Full structured violation report with recommended actions

### Step 7 — Run the API (optional)

```bash
docker-compose -f serving/docker-compose.yml up
curl -X POST http://localhost:8000/analyse -F "file=@path/to/image.jpg" | python -m json.tool
```

---

## Running Inference in Google Colab

Use this when you have no local GPU or want to demo the system without a full local install.

### Option A — Run inference on uploaded images

```python
# Cell 1 — install dependencies
!pip install ultralytics opencv-python-headless gradio pydantic-settings slowapi python-dotenv

# Cell 2 — clone the repo (or upload your own copy)
!git clone https://github.com/YOUR_USERNAME/construction-safety-system-design.git
%cd construction-safety-system-design

# Cell 3 — upload weights file
from google.colab import files
uploaded = files.upload()   # upload runs/train53/weights/best.pt

import os, shutil
os.makedirs("runs/train53/weights", exist_ok=True)
shutil.move(list(uploaded.keys())[0], "runs/train53/weights/best.pt")

# Cell 4 — set environment variables (no .env file needed in Colab)
os.environ["MODEL_WEIGHTS_PATH"] = "runs/train53/weights/best.pt"
os.environ["RULES_CONFIG_PATH"] = "rules.yaml"
os.environ["LOG_LEVEL"] = "INFO"

# Cell 5 — run inference on an uploaded image
uploaded_img = files.upload()   # upload any construction site image
img_path = list(uploaded_img.keys())[0]

from inference.pipeline import ConstructionSafetyPipeline
from PIL import Image
import numpy as np

pipeline = ConstructionSafetyPipeline(weights_path="runs/train53/weights/best.pt")
frame = np.array(Image.open(img_path).convert("RGB"))
result = pipeline.analyse(frame)

print(result.formatted_report)

# Cell 6 — display annotated output
from IPython.display import display
import PIL.Image
display(PIL.Image.fromarray(result.annotated_frame))
```

### Option B — Run the Gradio demo in Colab

```python
# After completing Cells 1–4 above:

# Cell 5 — launch demo with public URL
import subprocess
proc = subprocess.Popen(["python", "demo.py"])

# Gradio will print a public share URL — open it in your browser
# e.g. https://abc123.gradio.live
```

### Option C — Batch test on the test set

```python
# After completing Cells 1–4 above, run the batch test script:
!python scripts/batch_test.py
```

> **GPU note:** Colab's free T4 GPU is sufficient for inference. For training, use Colab Pro
> or run locally on an RTX 3060+. Training notebooks are in `training/notebooks/`.

---

## Running Tests

```bash
pytest tests/ -v
```

37 unit tests covering all 6 safety rules, score formula, temporal decay, suppression logic,
and edge cases. No tests touch the model — all rule logic is independently testable.

New tests added in the latest revision cover: Rule 3 deduplication, Rule 3 exclusion of bonus
classes and Rules 1/2 violations, fallback conflict → `partial_compliance` path, per-class
highest-confidence selection, Rule 5 partial PPE coverage (helmet-only, vest-only, mask-only).

---

## Repository Structure

```
construction-safety/
├── scripts/           # data preparation — download, remap, merge, augment, validate
├── training/
│   ├── configs/       # train_config.yaml
│   └── notebooks/     # training.ipynb (train4) + Retrain with finetuning.ipynb (train53)
├── inference/         # SafetyChecker, SiteScorer, Annotator, pipeline, scene_classifier
├── serving/           # FastAPI + Dockerfile + docker-compose
├── tests/             # 29 pytest unit tests
├── ARCHITECTURE.md    # detailed architecture and design notes
├── DATASET.md         # dataset sources, mapping, and split details
├── dataset/
│   ├── raw/           # original downloaded sources (never modified)
│   ├── remapped/      # label-remapped copies (6-class schema)
│   ├── mappings/      # YAML class mapping per source
│   └── merged/        # final unified dataset + train_raw/ split for fine-tuning
├── runs/
│   ├── train4/        # weights/best.pt — mAP@50=79.5%
│   └── train53/       # weights/best.pt — mAP@50=85.8% (active)
├── demo.py            # Gradio demo
├── rules.yaml         # all thresholds — no magic numbers in inference code
├── SAFETY_RULES.md    # formal rule specification with suppression logic
```

---

## Design Decisions & Trade-offs

| Decision | Choice | Reason |
|---|---|---|
| Single-pass YOLO | All classes in one model | Simpler deployment; association by geometry is more robust than crop-based two-stage |
| YOLOv8m | Medium model | Better small-object detection for far-field PPE; fits 6 GB VRAM with batch=16 |
| COCO pretrained weights | Transfer learning from train4 → train53 | 10× lower lr for fine-tuning; converges in 20 epochs vs 50 |
| `rule_confidence` composite | Detection conf + bbox size + edge proximity | More trustworthy than raw YOLO score; prevents edge-box false alerts |
| Heuristic scene classifier | Sky-pixel HSV detection | Zero model weight, zero inference cost, sufficient for Rule 2 risk modulation |
| Compliance score (0–100) | Score over binary flag | Site managers think in scores and trends, not event counts |
| `violation_conf_min = 0.40` | Lowered from 0.50 | Asymmetric cost: false negative = injury risk; false positive = supervisor time |
| Anatomical position filter | `no_helmet` center must be in top 60% of person box | Rejects mislocalized detections where model draws violation box on torso |
| Above-adjacent suppression | `helmet_on` above `no_helmet` → suppress regardless of confidence | Spatial evidence overrides confidence ordering for physically impossible layouts |
| Two training runs | train4 (full aug dataset) → train53 (fine-tune raw only) | Avoids 72h full retrain; fine-tune on 6,517 raw images with online aug in ~9h |

---

## Future Work

- Night-time / floodlit scene training data (IR camera input)
- PTZ auto-zoom integration for far-field workers (Rule 4 production upgrade)
- Worker re-identification across frames for persistent violation tracking
- True zone mapping (visitor vs. active worker) beyond proximity heuristic
- Region-specific dataset collection for local deployment context
- Real-time RTSP stream support for live CCTV integration
- Undersample `kaggle_safety_vests` to improve violation class ratio (currently 22.9%, target 31%)
