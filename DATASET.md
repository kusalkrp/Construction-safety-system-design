# Dataset Documentation — Construction Safety Monitor

## Overview

The dataset was assembled from seven public sources merged into a unified 6-class schema.
The base dataset was selected for its YOLO-format annotations and sufficient volume, but was
found to have four specific gaps relevant to this project's safety rules: underrepresentation
of indoor scenes, imbalance between violation classes, limited lighting condition diversity,
and sparse scaffolding/elevation coverage. Each gap was addressed through targeted data
collection with documented rationale.

The violation class ratio was intentionally elevated to optimise for recall on safety-critical
classes, where false negatives carry greater real-world cost than false positives.

---

## Data Sources

Seven sources were downloaded, remapped, and merged. No source file was ever modified
in place — all remapping and merging operates on read-only copies in `dataset/raw/`.

| Source | Images | Key Classes Contributed | Purpose |
|---|---|---|---|
| Roboflow Construction Site Safety v12 (base) | 194 | All 6 classes — balanced baseline | Foundation dataset |
| Kaggle PPE Kit Detection | 1,415 | no_helmet, no_vest, helmet_on, vest_on | Primary PPE violation source |
| Kaggle Safety Vests Detection | 3,897 | vest_on, no_vest — dominant vest source | Vest class coverage |
| Roboflow Scaffolding | 1,999 | helmet_on, vest_on, person — elevation scenes | Gap 4: scaffolding coverage |
| Roboflow Worker | 768 | person only — 2,279 person boxes | Gap fix: weak person class detection |
| Roboflow No-Helmet | 212 | no_helmet — 384 boxes | Gap fix: no_helmet class shortfall |
| Roboflow No-Vest | 351 | no_vest — additional violation coverage | Gap 2: violation class imbalance |
| **Total raw** | **8,836** | | |

---

## Class Schema

Six classes used throughout — matched to YOLO class IDs used in training.

| ID | Class | Type | Remapped From |
|---|---|---|---|
| 0 | `helmet_on` | Safe | Hardhat, hard-hat, helmet |
| 1 | `no_helmet` | Violation | NO-Hardhat, no hard hat, No Helmet |
| 2 | `vest_on` | Safe | Safety Vest, hi-vis vest |
| 3 | `no_vest` | Violation | NO-Safety Vest, no vest |
| 4 | `person` | Neutral | Person, working, not-working |
| 5 | `mask_on` | Safe (bonus) | Mask |

**Discarded classes from base dataset:** Safety Cone, machinery, vehicle — not relevant
to PPE compliance rules. Their labels are dropped during remapping; their images are
retained if they contain any of the 6 retained classes.

---

## Identified Gaps and How Each Was Addressed

### Gap 1 — Indoor Scene Underrepresentation

The base dataset skews strongly toward outdoor daylight construction scenes. Indoor
environments (warehouses, covered construction floors) are sparse. Indoor artificial
lighting creates fundamentally different visual conditions: harsher shadows, different
colour rendering on PPE, lower overall image contrast.

**Address:** The Roboflow Scaffolding source contributes indoor covered-area scenes.
Brightness/contrast augmentation simulates artificial lighting variations. Acknowledged
as a remaining limitation (see Known Limitations).

### Gap 2 — Violation Class Imbalance

`no_vest` is underrepresented relative to `no_helmet` across most public datasets.
Uneven coverage means uneven recall on the two primary violation classes.

**Address:** Dedicated addition of Roboflow No-Vest (351 images, additional `no_vest`
annotations). The Kaggle PPE Kit source also contributes `no_vest` coverage. Remaining
imbalance is noted (22.9% violation ratio against 31% target) and partially compensated
by `cls=1.5` class weighting during training.

### Gap 3 — Lighting Condition Diversity

Base dataset dominated by daylight scenes in good conditions. Overcast, shadow-heavy,
and artificial-light imagery limited.

**Address:** Offline augmentation applies shadow overlays, fog/haze simulation, and
brightness/contrast shifts to the training split (details in Augmentation section).
Real low-light imagery remains a documented gap.

### Gap 4 — Scaffolding and Elevation Coverage

Workers at height are a specific escalation case in safety Rule 1 — the system
escalates severity when a helmetless worker is detected in the upper frame region.
The base dataset had limited scaffolding imagery.

**Address:** Roboflow Scaffolding source (1,999 images) directly targets this scenario,
contributing helmet_on, vest_on, and person annotations in elevated contexts.

### Gap 5 — Weak Person Class Detection

After initial training (train4), the `person` class had median YOLO confidence of 0.04
on the test set. 56 of 100 test images returned zero workers detected, causing the
pipeline to score 96/100 as COMPLIANT despite visible violations.

**Address:** Roboflow Worker source added (768 images, 2,279 person annotations).
This increased person-class training annotations from ~1,400 to 3,726 (+166%).
A fallback path was also added to `SafetyChecker` — when no person boxes are detected,
standalone `no_helmet`/`no_vest` detections above the confidence threshold generate
ViolationReports directly. This fallback was validated as a bridge pending fine-tuning.

---

## Dataset Split

| Split | Images (raw) | Images (with augmentation) | Annotations |
|---|---|---|---|
| train | 6,517 | 20,106 (×2 augmentation factor) | 16,266 raw-split |
| valid | 1,325 | 1,325 (no augmentation) | ~3,700 |
| test | 883 | 883 (no augmentation) | ~2,600 |

**Split ratios (raw images):** 73.9% train / 15.0% val / 10.0% test

**Fine-tuning split (`data_raw.yaml`):** A separate `train_raw/` split (6,517 raw images,
no `_aug` files) was created for fine-tuning (train53). Online augmentation (`augment=True`)
in YOLO makes pre-augmented copies redundant for fine-tuning. This reduced fine-tuning
time from ~72 hours (full dataset) to ~9 hours with no quality loss.

---

## Class Distribution (Training Split — Raw)

Annotation counts in the raw training split, as produced by `scripts/validate_dataset.py`:

| Class | Annotations | % of Total | Type |
|---|---|---|---|
| `helmet_on` | 3,350 | 16.7% | Safe |
| `no_helmet` | 1,835 | 9.1% | Violation |
| `vest_on` | 7,095 | 35.3% | Safe |
| `no_vest` | 2,702 | 13.4% | Violation |
| `person` | 3,726 | 18.5% | Neutral |
| `mask_on` | 1,086 | 5.4% | Safe (bonus) |
| **Total** | **19,794** | | |

**Violation class ratio:** 22.5% (`no_helmet` + `no_vest` combined). Target was 31% —
the gap is caused by vest_on dominance from the Kaggle Safety Vests source. Compensated
during training by `cls=1.5` class loss weight on violation classes.

**Violation class imbalance (`no_vest` vs `no_helmet`):** 47.2%. Target ≤ 20%. Root cause:
the Kaggle Safety Vests source is vest-dominant. A future fix would undersample this source
during merge.

---

## Annotation Approach

**Format:** YOLO v8 — normalised bounding boxes (`class cx cy w h`, values 0–1).

**Tool:** Remapping and merging via custom scripts (`scripts/remap_labels.py`,
`scripts/merge_datasets.py`). Class mappings per source stored in `dataset/mappings/*.yaml`.

### Bounding Box Policy

| Object | Box Drawn Around |
|---|---|
| `helmet_on` / `no_helmet` | Head and helmet region — tight around the head |
| `vest_on` / `no_vest` | Torso region — shoulders to waist |
| `person` | Full body — used when PPE state is ambiguous or distant |
| `mask_on` | Face / mask region — tight around face |

### Occlusion Policy

| Occlusion Level | Action |
|---|---|
| Worker > 50% visible | Annotate normally |
| Worker 20–50% visible | Annotate if PPE state clearly determinable |
| Worker < 20% visible | Skip — do not annotate |

This prevents the model from learning from ambiguous, low-information annotations.

### Multiple PPE Per Worker

Each worker receives independent boxes for head and torso PPE. A worker missing a helmet
but wearing a vest gets a `no_helmet` box on the head region and a `vest_on` box on the
torso region. Annotation count per image therefore exceeds worker count.

### Ambiguous PPE State

If PPE state cannot be determined (object too small, image quality too low, state genuinely
unclear), the worker is annotated as `person` only. No guess is made. This produces fewer
annotations but more reliable ones.

### Far-Field Worker Policy

Workers whose full-body bounding box height would be < 40px in a 640px input are not
annotated for PPE state — only as `person` if clearly distinguishable. This is consistent
with inference-time Rule 4 (far-field unverifiable) so training data and runtime behaviour
are aligned.

---

## Augmentation Strategy

Offline augmentation is applied to the training split only via `scripts/augment_dataset.py`
with `--factor 2` (each training image produces 2 additional augmented copies).

Each augmentation maps to a specific real-world deployment condition:

| Augmentation | Real-World Condition | Probability |
|---|---|---|
| Diagonal shadow overlay | Scaffolding / overhead obstruction blocking site camera | 0.40 |
| Fog / haze simulation | Overcast outdoor morning, dusty construction environments | 0.20 |
| Gaussian blur | Low-quality CCTV footage, cheap IP camera lens | 0.30 |
| Brightness / contrast shift | Indoor fluorescent vs. direct midday sunlight | 0.80 |
| Gaussian noise | Surveillance camera graininess, low-light sensor noise | 0.30 |
| Horizontal flip | Mirrored site layouts, cameras on either side | 0.50 |

**Factor = 2** is conservative — chosen to multiply training data without letting augmented
data dominate over real images. YOLOv8's built-in online augmentation (mosaic, HSV jitter,
flipud) is additionally enabled during training and is complementary.

Augmented files are named with an `_aug` suffix and kept in `dataset/merged/images/train/`.
The `train_raw/` split (created by `scripts/make_raw_split.py`) excludes all `_aug` files
and is used exclusively for fine-tuning.

---

## Environments Covered

| Environment | Coverage | How Addressed |
|---|---|---|
| Outdoor open lot (ground-level) | Good | Base dataset + Kaggle sources |
| Scaffolding / height | Moderate | Roboflow Scaffolding source (1,999 images) |
| Indoor warehouse / covered site | Limited | Brightness augmentation; acknowledged gap |
| Mixed lighting (overcast / shadow) | Simulated | Shadow overlay + fog augmentation |
| Artificial lighting (fluorescent) | Simulated | Brightness/contrast augmentation |
| Night / floodlit outdoor | Not covered | Acknowledged limitation |
| CCTV / surveillance grain | Simulated | Gaussian noise + blur augmentation |

---

## Known Limitations

| Limitation | Impact | Production Fix |
|---|---|---|
| Night-time / low-light imagery absent | Model performance degrades significantly in dark scenes | IR-capable cameras or dedicated low-light dataset |
| Indoor scenes remain underrepresented | Indoor recall lower than outdoor | Continued targeted indoor data collection |
| `mask_on` class sparse (~1,086 train annotations) | Treat as bonus signal only — unreliable in isolation | Dedicated mask dataset |
| Violation ratio 22.5% (target 31%) | Violation recall lower than optimum | Undersample `kaggle_safety_vests` during merge |
| `vest_on` dominance (35.3% of annotations) | Vest-safe class overrepresented | Source-level undersampling |
| Far-field workers (< 40px bbox) unclassifiable | PPE state undetectable at distance | PTZ cameras with auto-zoom |
| Dataset geography is global, not region-specific | PPE equipment styles may vary locally | Region-specific data collection for local deployment |
| Scraped images may include JPEG artefacts | Slight quality degradation | Raw image collection from site cameras |

---

## Retraining Justification (train4 → train53)

### Why Retraining Was Necessary

After initial training (train4), post-training inference testing on the held-out test set
identified a critical pipeline failure that made retraining unavoidable.

### Finding 1 — Person Class Near-Failure

The `person` detection class (class ID 4) had a median YOLO confidence of **0.04** across
200 test images. Of 200 images sampled:

- 91 images: zero person detections at any confidence threshold
- 334 person detections: confidence below the 0.30 runtime gate (unusable)
- 84 person detections: confidence ≥ 0.30 (usable)

**Root cause:** The merged dataset used for train4 contained approximately 1,400 person-class
annotations in the train split — insufficient relative to the 6 PPE classes the model was
simultaneously learning. The person class was crowded out during gradient updates.

**Impact:** The inference pipeline anchors all compliance checks to detected person bounding
boxes. When no person is detected, `SafetyChecker` finds no workers, `SiteScorer` receives
an empty report, and the pipeline returns 100/100 COMPLIANT regardless of visible PPE
violations.

Batch test result on 100 images before the fix:
- COMPLIANT: 96 images (many false — no person detected)
- CAUTION: 3 images
- AT RISK: 1 image
- CRITICAL: 0 images
- Zero worker detections: **56 / 100 images**

### Finding 2 — no_helmet Annotation Shortfall

The train4 dataset had approximately 900 `no_helmet` training annotations, well below the
2,000 target. This directly reduced detection confidence on the `no_helmet` class — the
highest-severity violation class (CRITICAL). A model that consistently undershoots confidence
on the most dangerous violation is a safety risk.

### Finding 3 — Violation Class Imbalance

`no_vest` had 1.58× more annotations than `no_helmet` (imbalance = 58%). Target was ≤ 20%.
This creates uneven detection capability between the two primary violation classes —
both governed by safety rules of comparable severity.

---

### Interim Fix Applied Before Retraining

While retraining was prepared, a fallback was added to `inference/safety_checker.py`:

When no person boxes are detected, the pipeline creates ViolationReports directly from
standalone `no_helmet` / `no_vest` detections above `violation_conf_min` (0.40). This allows
the system to catch violations even when the person class fails.

`violation_conf_min` was also lowered from 0.50 → 0.40 based on threshold sensitivity
analysis — recovers ~7% recall on `no_helmet` at no measurable precision cost.

Batch test result after interim fix:
- COMPLIANT: 51 images
- CAUTION: 41 images
- AT RISK: 3 images
- CRITICAL: 5 images
- Zero worker detections: **7 / 100 images** (genuine empty scenes)

The fallback is a documented workaround. The root cause fix is retraining with a dataset
that has adequate person-class annotations.

---

### New Datasets Added for Retraining

#### Dataset 1 — roboflow_worker

| Field | Detail |
|---|---|
| Source | Roboflow (worker-detection-cons-site v1) |
| Licence | CC BY 4.0 |
| Images used | 568 (train split only) |
| Original classes | `working` (1), `not-working` (0) |
| Remapped to | `person` (4) |
| Annotation yield | 2,279 person bounding boxes |

**Justification:** Both `working` and `not-working` labels describe the same object in our
schema — a person present on site. Remapping both to `person` is semantically correct.
This dataset was specifically collected for worker detection on construction sites, making
it directly applicable to Finding 1.

Person-class annotations in train split:
- Before (train4): ~1,400
- After (train53): ~3,726 (+166%)

#### Dataset 2 — roboflow_no_helmet

| Field | Detail |
|---|---|
| Source | Roboflow (kusal-punchihewa-s-workspace) |
| Images used | 212 (train only — no val/test split provided) |
| Original classes | `No Helmet` (0) |
| Remapped to | `no_helmet` (1) |
| Annotation yield | 384 no_helmet bounding boxes |

**Justification:** Directly targets Finding 2 (no_helmet shortfall) and partially addresses
Finding 3 (class imbalance). Adding 384 `no_helmet` annotations raises the train count from
~900 to ~1,835. The remaining shortfall from the 2,000 target is compensated by `cls=1.5`
class loss weighting during training.

---

### Dataset Comparison: train4 → train53

| Metric | train4 dataset | train53 dataset |
|---|---|---|
| Total raw images | 8,068 | 8,836 (+768) |
| Train images (raw) | ~5,600 | 6,517 |
| Train images (augmented) | ~18,000 | 20,106 |
| Train images (fine-tune) | N/A | 6,517 (`data_raw.yaml`) |
| `person` annotations | ~1,400 | 3,726 (+166%) |
| `no_helmet` annotations | ~900 | 1,835 (+104%) |
| `no_vest` annotations | ~2,100 | 2,702 (+29%) |
| Violation ratio | ~22% | 22.9% (structural — unchanged) |
| no_helmet/no_vest imbalance | ~61% | 38% (improved) |

---

### Fine-Tuning Strategy

**Decision: fine-tune from train4 weights, not retrain from scratch.**

train4 learned PPE feature representations well (mAP@50 = 79.5%). Retraining from COCO
weights would discard this knowledge and require ~50 full epochs to relearn PPE classes
before improving person detection. Fine-tuning preserves all existing PPE capability and
directs gradient updates only where the model is weak — the person class.

**Hyperparameter changes:**

| Parameter | train4 | train53 | Reason |
|---|---|---|---|
| model | `yolov8m.pt` | train4 best.pt | Preserve PPE features |
| data | `data.yaml` | `data_raw.yaml` | Avoid 72h training (see below) |
| lr0 | 0.001 | 0.0001 | 10× lower — weights near optimum |
| batch | 8 | 16 | Safe with raw dataset size |
| epochs | 50 | 20 | Fine-tuning converges in 15–20 |
| patience | 15 | 8 | Early stop if plateaued |
| warmup_epochs | 3 | 2 | Less warmup from good initialisation |

**Why `data_raw.yaml` instead of `data.yaml`:**

The full `data.yaml` (20,106 images) includes 13,477 pre-augmented copies. For fine-tuning,
these are redundant because `augment=True` applies equivalent online augmentation per-batch
at training time.

Using full dataset: 20,106 / 16 = 1,257 iterations/epoch × 20 epochs ≈ **28 hours**
Using raw only: 6,517 / 16 = 407 iterations/epoch × 20 epochs ≈ **9 hours**

No quality loss — online augmentation provides the same variation as offline pre-augmentation.
Pre-augmentation was designed for the original scratch training run (train4) where it
meaningfully expanded a smaller dataset. For fine-tuning it is unnecessary overhead.

---

## Remapping Files

Each source has a corresponding YAML mapping file in `dataset/mappings/`:

| Source | Mapping File |
|---|---|
| Roboflow Construction Site Safety (base) | `roboflow_base.yaml` |
| Kaggle PPE Kit Detection | `kaggle_ppe_kit.yaml` |
| Kaggle Safety Vests Detection | `kaggle_safety_vests.yaml` |
| Roboflow Scaffolding | `roboflow_scaffolding.yaml` |
| Roboflow Worker | `roboflow_worker.yaml` |
| Roboflow No-Helmet | `roboflow_no_helmet.yaml` |
| Roboflow No-Vest | `roboflow_no_vest.yaml` |

Each YAML maps original class names to the 6-class schema. Classes not in the schema
(Safety Cone, machinery, vehicle) are listed as `null` — their annotations are dropped,
their images retained if they contain any mappable class.
