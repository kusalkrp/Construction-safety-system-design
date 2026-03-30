# System Architecture — Construction Safety Monitor

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Inference Pipeline](#2-inference-pipeline)
3. [PPE Association Logic](#3-ppe-association-logic)
4. [Safety Rule Application](#4-safety-rule-application)
5. [Compliance Scoring](#5-compliance-scoring)
6. [Scene Classification](#6-scene-classification)
7. [Annotation Rendering](#7-annotation-rendering)
8. [Dataset Pipeline](#8-dataset-pipeline)
9. [Dataset Manipulation Methods](#9-dataset-manipulation-methods)
10. [Dataset Validation](#10-dataset-validation)

---

## 1. System Overview

The system has two independent pipelines: a **dataset pipeline** (offline, runs once) and an
**inference pipeline** (online, runs per frame). They share the 6-class schema and the
`rules.yaml` threshold file but are otherwise fully decoupled.

```mermaid
flowchart TD
    subgraph OFFLINE["Dataset Pipeline (one-time)"]
        RAW["Raw Datasets\n7 sources"]
        REMAP["remap_labels.py\nClass ID remapping"]
        MERGE["merge_datasets.py\nUnified split"]
        RAW_SPLIT["make_raw_split.py\ntrain_raw/ for fine-tuning"]
        AUG["augment_dataset.py\n×2 offline augmentation"]
        TRAIN["YOLOv8m Training\ntrain4 → train53"]
        WEIGHTS["runs/train53/weights/best.pt"]

        RAW --> REMAP --> MERGE --> RAW_SPLIT
        MERGE --> AUG
        AUG --> TRAIN
        RAW_SPLIT --> TRAIN
        TRAIN --> WEIGHTS
    end

    subgraph ONLINE["Inference Pipeline (per frame)"]
        INPUT["Input Frame (RGB)"]
        YOLO["YOLOv8m\nSingle-pass detection"]
        CHECKER["SafetyChecker\nRules 1–6"]
        SCORER["SiteScorer\n0–100 score"]
        ANNOTATOR["Annotator\nDraw boxes + overlay"]
        OUTPUT["Annotated Frame\n+ Structured Report\n+ Compliance Score"]

        INPUT --> YOLO --> CHECKER --> SCORER --> ANNOTATOR --> OUTPUT
    end

    WEIGHTS -->|"loaded at startup"| YOLO
    RULES["rules.yaml\nAll thresholds"] -->|"constants.py"| CHECKER
    RULES -->|"constants.py"| SCORER
```

**Design principle:** `SafetyChecker` never calls YOLO. `SiteScorer` never applies rules.
`Annotator` has no inference logic. `pipeline.py` is the only file that knows about all of them.

---

## 2. Inference Pipeline

The pipeline runs YOLO once on the full frame and returns all detection boxes simultaneously.
`SafetyChecker` then associates PPE boxes to person boxes by geometry — no second YOLO call.

```mermaid
flowchart TD
    FRAME["Input Frame\nRGB uint8 H×W×3"]
    BGR["Convert RGB→BGR\ncv2.cvtColor"]
    YOLO["YOLO inference\nmodel(frame_bgr, conf=0.30)"]
    PARSE["Parse results\nDetection dataclasses\n(class_name, conf, x1,y1,x2,y2)"]

    SPLIT_P["Person boxes\nclass_name == 'person'"]
    SPLIT_PPE["PPE boxes\nhelmet_on / no_helmet\nvest_on / no_vest / mask_on"]

    SCENE["classify_scene(frame_rgb)\nindoor / outdoor"]
    CHECK["SafetyChecker.analyse()\nRules 1–6 per worker"]
    FALLBACK{"Any person\nboxes?"}
    FALLBACK_PATH["Fallback: create reports\nfrom standalone PPE boxes\n_report_from_violation_det()"]
    NORMAL["Normal path:\nassociate PPE → persons\n_check_worker() per person"]

    SITE["_check_site_level()\nRule 6 crowd check"]
    SITE_REPORT["SiteReport\nworker_reports + site_alert"]

    SCORE["SiteScorer.compute()\n0–100 score + band"]
    DRAW["draw_annotations()\nBoxes + score overlay"]
    RESULT["AnalysisResult\nreport + score + annotated_frame"]

    FRAME --> BGR --> YOLO --> PARSE
    PARSE --> SPLIT_P & SPLIT_PPE
    SPLIT_P & SPLIT_PPE --> FALLBACK
    SCENE --> CHECK
    FALLBACK -->|"Yes"| NORMAL --> CHECK
    FALLBACK -->|"No"| FALLBACK_PATH --> CHECK
    CHECK --> SITE --> SITE_REPORT
    SITE_REPORT --> SCORE
    SITE_REPORT --> DRAW
    SCORE --> DRAW
    DRAW --> RESULT
```

### `AnalysisResult` fields

| Field | Type | Description |
|---|---|---|
| `site_report` | `SiteReport` | Per-worker ViolationReports + site alert |
| `score_result` | `ScoreResult` | 0–100 score, band, violation counts |
| `annotated_frame` | `np.ndarray` | RGB frame with boxes drawn |
| `formatted_report` | `str` | Human-readable text report |
| `timestamp` | `float` | Unix timestamp of analysis |

---

## 3. PPE Association Logic

The core challenge: YOLO anchors the `person` box at the torso, so the head is often above
or at the top edge of the bounding box. A naïve IoU search misses helmet detections.
Three association strategies are used in order:

```mermaid
flowchart TD
    PERSON["Person bounding box\nx1,y1,x2,y2"]
    EXPAND["expand_crop_for_ppe()\nExpand upward by 60% of box height\nCaptures head above torso anchor"]
    SEARCH["Search all PPE detections\nfor each PPE box:"]

    IOU["compute_iou()\nPPE box ∩ expanded person box\nIoU > PPE_PERSON_OVERLAP_IOU (0.10)?"]
    ABOVE["is_above_person()\nPPE box sits above person box?\n≥30% horizontal overlap\n& y_bottom < person y_top + 10px?"]
    UNMATCHED["PPE box unmatched\n→ not associated to this worker"]

    MATCHED["PPE box associated\nto this worker"]

    SUPPRESS["Suppression checks\nbefore creating violation"]
    CONFLICT["Conflict suppression\nhelmet_on IoU > 0.10 with no_helmet\n& helmet_on conf > no_helmet conf\n→ suppress no_helmet"]
    ADJACENT["Above-adjacent suppression\nhelmet_on above no_helmet\n→ suppress regardless of confidence"]
    ANATOM["Anatomical position filter\nno_helmet center > 60% of person bbox → reject\nno_vest center < 25% of frame → reject"]

    VIOLATION["Violation confirmed\n→ apply Rule 1 or Rule 2"]

    PERSON --> EXPAND --> SEARCH
    SEARCH --> IOU
    IOU -->|"Yes"| MATCHED
    IOU -->|"No"| ABOVE
    ABOVE -->|"Yes"| MATCHED
    ABOVE -->|"No"| UNMATCHED

    MATCHED --> SUPPRESS
    SUPPRESS --> CONFLICT
    SUPPRESS --> ADJACENT
    SUPPRESS --> ANATOM
    CONFLICT & ADJACENT & ANATOM -->|"Not suppressed"| VIOLATION
    CONFLICT & ADJACENT & ANATOM -->|"Suppressed"| UNMATCHED
```

### `expand_crop_for_ppe(bbox, frame_height, frame_width)`

```
expanded_y1 = max(0, y1 − (y2 − y1) × PERSON_CROP_HEAD_EXPAND)
expanded_y1 = max(0, y1 − (y2 − y1) × 0.60)
```

Expands the person bounding box upward by 60% of its height. This captures helmet detections
that sit above the torso-anchored person box, which is the primary cause of missed `no_helmet`
detections in close-up scenes.

### `is_above_person(ppe_bbox, person_bbox)`

Fires when two conditions are both true:
1. The PPE box bottom edge is within 10px above the person box top edge
2. The horizontal overlap between the PPE box and person box is ≥ 30% of the PPE box width

This is intentionally independent of confidence — spatial position is definitive evidence
that a helmet sits above a person. A mislocalized `no_helmet` box at high confidence is
suppressed when a `helmet_on` box sits above it, because the spatial layout is physically
unambiguous.

### `compute_iou(box_a, box_b)`

Standard intersection-over-union for two `(x1, y1, x2, y2)` boxes:

```
intersection = max(0, min(x2a, x2b) − max(x1a, x1b)) × max(0, min(y2a, y2b) − max(y1a, y1b))
union = area_a + area_b − intersection
IoU = intersection / union
```

---

## 4. Safety Rule Application

`_check_worker()` applies Rules 1–5 in sequence for each detected person.

```mermaid
flowchart TD
    WORKER["Worker detected\nwith associated PPE boxes"]

    R4{"Person bbox height\n< 40px?"}
    R4Y["Rule 4 — UNVERIFIABLE\nPPE state undetectable\nat this distance"]

    VISITOR{"Near frame edge?\ny1 or x1 < 30px\nfrom any edge?"}
    VISITORF["Classified as VISITOR\nINFO severity\nNot scored"]

    PPE{"Any PPE boxes\nassociated?"}
    R5["Rule 5 — PPE GAP\nPerson detected\nbut no PPE overlap\nWARNING → review queue"]

    HELMET{"no_helmet associated?\nconf ≥ 0.40?"}
    ANATOM_H{"no_helmet center\nin upper 60% of person bbox?"}
    CONFLICT_H{"helmet_on also\npresent with higher conf?"}
    ABOVE_H{"helmet_on above\nno_helmet?"}
    FRAME_POS{"Person y1 < frame_height × 0.60\n(upper frame = elevated worker)?"}

    R1_E["Rule 1 — CRITICAL-ELEVATED\nWorker at height\nwithout helmet"]
    R1["Rule 1 — CRITICAL\nNo helmet in active zone"]
    PARTIAL["Rule 3 — WARNING\nPARTIAL_COMPLIANCE_SUSPECTED\nConflicting signals"]

    VEST{"no_vest associated?\nconf ≥ 0.40?"}
    ANATOM_V{"no_vest center\nbelow top 25% of frame?"}
    SCENE{"Scene context?"}
    R2_OUT["Rule 2 — HIGH\nNo vest, outdoor scene"]
    R2_IN["Rule 2 — WARNING\nNo vest, indoor scene"]

    CONF{"Any PPE conf\nbetween 0.35–0.65?"}
    R3["Rule 3 — WARNING\nPartial compliance\nAmbiguous PPE state"]

    COMPLIANT["COMPLIANT\nAll checks passed"]

    WORKER --> R4
    R4 -->|"Yes"| R4Y
    R4 -->|"No"| VISITOR
    VISITOR -->|"Yes"| VISITORF
    VISITOR -->|"No"| PPE
    PPE -->|"None"| R5
    PPE -->|"Present"| HELMET

    HELMET -->|"Yes"| ANATOM_H
    ANATOM_H -->|"No — torso zone"| COMPLIANT
    ANATOM_H -->|"Yes — head zone"| CONFLICT_H
    CONFLICT_H -->|"Yes"| ABOVE_H
    ABOVE_H -->|"Yes"| PARTIAL
    ABOVE_H -->|"No"| PARTIAL
    CONFLICT_H -->|"No"| FRAME_POS
    FRAME_POS -->|"Yes"| R1_E
    FRAME_POS -->|"No"| R1

    HELMET -->|"No"| VEST
    VEST -->|"Yes"| ANATOM_V
    ANATOM_V -->|"No — head zone"| COMPLIANT
    ANATOM_V -->|"Yes"| SCENE
    SCENE -->|"outdoor"| R2_OUT
    SCENE -->|"indoor"| R2_IN

    VEST -->|"No"| CONF
    CONF -->|"Yes"| R3
    CONF -->|"No"| COMPLIANT
```

### Rule 6 — Site-Level Crowd Check (`_check_site_level`)

Runs after all per-worker reports are generated:

```mermaid
flowchart LR
    REPORTS["All ViolationReports\nfor this frame"]
    COUNT{"Workers detected\n≥ 4?"}
    RATIO{"Violating workers /\ntotal workers ≥ 50%?\n(rule_confidence ≥ 0.60)"}
    ALERT["SITE ALERT\nSystemic PPE non-compliance\nCrowd multiplier 1.3 applied"]
    NONE["No site-level alert"]

    REPORTS --> COUNT
    COUNT -->|"Yes"| RATIO
    COUNT -->|"No"| NONE
    RATIO -->|"Yes"| ALERT
    RATIO -->|"No"| NONE
```

### `compute_rule_confidence(detection_conf, bbox_height, bbox_x_center, frame_width)`

A composite signal that is more trustworthy than raw YOLO confidence:

```
rule_confidence = (WEIGHT_DETECTION_CONF × detection_conf)
                + (WEIGHT_BBOX_SIZE       × min(1.0, bbox_height / 80))
                + (WEIGHT_EDGE            × (1.0 − edge_proximity_ratio))

Weights: 0.60 + 0.25 + 0.15 = 1.00

edge_proximity_ratio = min(bbox_x_center, frame_width − bbox_x_center) / (frame_width / 2)
```

| Tier | Threshold | Action |
|---|---|---|
| HIGH | ≥ 0.70 | Immediate alert |
| MEDIUM | 0.45–0.69 | Review queue |
| LOW | < 0.45 | Logged only |

---

## 5. Compliance Scoring

`SiteScorer.compute()` aggregates all ViolationReports into a single 0–100 score.

```mermaid
flowchart TD
    REPORTS["ViolationReports\n(worker_reports list)"]
    COUNT["Count violations by severity\nn_critical, n_high, n_warning"]
    DECAY["Apply temporal decay\nviolations > 60s → ×0.5 per minute\n_compute_decay()"]
    BASE["Base score = 100\n− (n_critical × 25)\n− (n_high × 15)\n− (n_warning × 5)"]
    CROWD{"Rule 6\ntriggered?"}
    MULT["Apply crowd multiplier\nscore × (1 / 1.3)"]
    CLAMP["Clamp to [0, 100]\nmax(0, min(100, score))"]
    BAND["_get_band(score)\n80–100 COMPLIANT green\n60–79 CAUTION amber\n40–59 AT RISK orange\n0–39 CRITICAL red"]
    HISTORY["_record_history()\nStore at 10s intervals\nfor trend analysis"]
    RESULT["ScoreResult\nscore, band, band_colour\nn_critical, n_high, n_warning\ndisplay string"]

    REPORTS --> COUNT --> DECAY --> BASE
    BASE --> CROWD
    CROWD -->|"Yes"| MULT --> CLAMP
    CROWD -->|"No"| CLAMP
    CLAMP --> BAND --> HISTORY --> RESULT
```

**Temporal decay:** Violations timestamped at detection time. For video, old violations
gradually reduce their score impact rather than snapping to zero, giving a smooth signal
that reflects sustained non-compliance rather than instantaneous frame state.

**Trend analysis (`get_trend_summary`):** Computes mean, minimum, and standard deviation
over `score_history` and returns a human-readable summary (e.g., "Average 74/100, minimum
43/100 — site in CAUTION band for this session").

---

## 6. Scene Classification

`classify_scene(frame_rgb)` determines indoor vs. outdoor context for Rule 2 severity.

```mermaid
flowchart LR
    FRAME["Input Frame RGB"]
    CROP["Sample top 20% of frame\nframe_rgb[:int(H×0.20), :]"]
    HSV["Convert to HSV\ncv2.cvtColor(crop, COLOR_RGB2HSV)"]
    MASK["Sky pixel mask:\nH ∈ [90, 140] (sky-blue hue)\nS ≥ 30 (not grey/white)"]
    RATIO["sky_ratio = masked_pixels / total_pixels"]
    DECISION{"sky_ratio ≥ 0.15?"}
    OUT["outdoor"]
    IND["indoor"]

    FRAME --> CROP --> HSV --> MASK --> RATIO --> DECISION
    DECISION -->|"Yes"| OUT
    DECISION -->|"No"| IND
```

**Why a heuristic, not a model:** Sky blue in HSV space is a stable, high-specificity signal
on construction sites. The alternative — a separate scene-classification model — would add
weight, latency, and a second inference call for negligible accuracy gain. The heuristic has
zero cost, is deterministic, and is sufficient for the sole purpose it serves: modulating
Rule 2 severity between HIGH (outdoor) and WARNING (indoor).

---

## 7. Annotation Rendering

`draw_annotations()` is the final stage — it reads ViolationReports and draws on the frame
without any inference logic.

```mermaid
flowchart TD
    INPUT["Input frame RGB\n+ ViolationReports\n+ ScoreResult\n+ site_alert"]
    BGR["Convert RGB→BGR\nfor OpenCV drawing"]

    LOOP["For each ViolationReport:"]
    BOX["cv2.rectangle()\nSolid box = person anchor\nColoured by severity"]
    DASHED{"violation_bbox\npresent?"}
    DASH_BOX["_draw_dashed_rect()\nDashed box = actual PPE\ndetection location"]
    LABEL["_draw_label()\nFilled background + text\n#id SEVERITY: VIOLATION [TIER]"]

    SCORE_OVL["_draw_score_overlay()\nTop-right corner\nScore number + band label"]
    ALERT_BANNER["_draw_site_alert_banner()\nRed banner at bottom\nif Rule 6 triggered"]

    OUT_RGB["Convert BGR→RGB\nReturn annotated frame"]

    INPUT --> BGR --> LOOP
    LOOP --> BOX --> DASHED
    DASHED -->|"Yes"| DASH_BOX --> LABEL
    DASHED -->|"No"| LABEL
    LABEL --> LOOP
    LOOP --> SCORE_OVL --> ALERT_BANNER --> OUT_RGB
```

### Colour Convention

| Colour | BGR Value | Used For |
|---|---|---|
| Green `#00AA44` | (68, 170, 0) | COMPLIANT workers |
| Red `#FF2200` | (0, 34, 255) | CRITICAL / CRITICAL-ELEVATED |
| Orange `#FF6600` | (0, 102, 255) | HIGH severity |
| Yellow `#FFCC00` | (0, 204, 255) | WARNING / review queue |
| Blue `#4488FF` | (255, 136, 68) | Visitors |
| Dark red | (0, 0, 200) | SITE ALERT banner |

### Solid box vs. dashed box

- **Solid box** — drawn at the person bounding box. This is the anchor. It tells the viewer
  which worker the report refers to.
- **Dashed box** — drawn at `violation_bbox`, the coordinates of the actual `no_helmet` or
  `no_vest` detection. This shows exactly where the model saw the violation, allowing a
  human reviewer to verify whether the detection is plausible or a false positive.

### `_draw_dashed_rect()`

Draws a dashed rectangle by decomposing each side into alternating filled/gap segments:

```
For each side (4 edges):
  steps = length // (dash_len × 2)
  For each step i:
    t0 = i × 2 × dash_len / length        (dash start)
    t1 = (i × 2 + 1) × dash_len / length  (dash end)
    cv2.line(img, interpolated_start, interpolated_end, colour, 1)
```

---

## 8. Dataset Pipeline

```mermaid
flowchart TD
    subgraph SOURCES["Raw Sources (dataset/raw/)"]
        S1["roboflow_base\n194 images"]
        S2["kaggle_ppe_kit\n1,415 images"]
        S3["kaggle_safety_vests\n3,897 images"]
        S4["roboflow_scaffolding\n1,999 images"]
        S5["roboflow_worker\n768 images"]
        S6["roboflow_no_helmet\n212 images"]
        S7["roboflow_no_vest\n351 images"]
    end

    subgraph REMAP_STEP["Remapping (scripts/remap_labels.py)"]
        MAP1["roboflow_base.yaml"]
        MAP2["kaggle_ppe_kit.yaml"]
        MAP3["kaggle_safety_vests.yaml"]
        MAP4["roboflow_scaffolding.yaml"]
        MAP5["roboflow_worker.yaml"]
        MAP6["roboflow_no_helmet.yaml"]
        MAP7["roboflow_no_vest.yaml"]
        REMAPPED["dataset/remapped/\n7 source directories\nAll in 6-class schema"]
    end

    subgraph MERGE_STEP["Merging (scripts/merge_datasets.py)"]
        COLLECT["Collect all image+label pairs\nPrefix filenames with source name\nShuffle (seed=42)"]
        SPLIT_STEP["Split 73.9% / 15.0% / 10.0%\ntrain / valid / test"]
        MERGED["dataset/merged/\nimages/train/ labels/train/\nimages/valid/ labels/valid/\nimages/test/  labels/test/\ndata.yaml"]
    end

    subgraph RAW_STEP["Raw Split (scripts/make_raw_split.py)"]
        FILTER["Filter: exclude *_aug* filenames\nCopy remaining to train_raw/"]
        RAWSPLIT["dataset/merged/\nimages/train_raw/ labels/train_raw/\ndata_raw.yaml\n6,517 raw images"]
    end

    subgraph AUG_STEP["Augmentation (scripts/augment_dataset.py)"]
        AUG_APPLY["Apply to train/ only\n× factor 2\nOutput: {stem}_aug{i}.jpg\n+ copy of original label"]
        AUG_OUT["20,106 total train images\n(6,517 raw + 13,477 aug)"]
    end

    subgraph VALIDATE_STEP["Validation (scripts/validate_dataset.py)"]
        VAL_CHECK["Count per-class annotations\nFlag imbalance warnings\nReport split distribution"]
    end

    SOURCES --> REMAP_STEP
    S1 --> MAP1 & S2 --> MAP2 & S3 --> MAP3 & S4 --> MAP4 & S5 --> MAP5 & S6 --> MAP6 & S7 --> MAP7
    MAP1 & MAP2 & MAP3 & MAP4 & MAP5 & MAP6 & MAP7 --> REMAPPED
    REMAPPED --> COLLECT --> SPLIT_STEP --> MERGED
    MERGED --> FILTER --> RAWSPLIT
    MERGED --> AUG_APPLY --> AUG_OUT
    AUG_OUT --> VAL_CHECK
```

---

## 9. Dataset Manipulation Methods

### `remap_labels.py` — Class ID Remapping

Walks every `.txt` label file in a source dataset and rewrites class IDs using a mapping YAML.

**Mapping YAML format (`dataset/mappings/roboflow_worker.yaml`):**
```yaml
0: 4    # not-working → person
1: 4    # working → person
```

A value of `-1` discards the annotation. Classes not present in the YAML are also discarded.

**Supported directory layouts:**
```
Layout 1 (split at root):        Layout 2 (images/ + labels/):     Layout 3 (flat):
source/train/images/              source/images/train/               source/images/
source/train/labels/              source/labels/train/               source/labels/
source/valid/images/              source/images/valid/
source/valid/labels/              source/labels/valid/
```

`find_dataset_roots()` auto-detects the layout before remapping.

**Per-class reporting:** `remap_file()` tracks source class counts and destination class counts
and logs a before/after table per split so the engineer can verify the mapping is correct.

---

### `merge_datasets.py` — Multi-Source Merge

Merges N remapped source directories into one unified dataset with a clean train/val/test split.

```mermaid
flowchart LR
    SRC["N remapped\nsource directories"]
    COLLECT["collect_pairs(source)\nFind all (image, label) pairs\nHandle split and flat layouts"]
    PREFIX["Prefix filenames with source name\nkaggle_ppe_kit_img001.jpg\nPrevents filename collisions"]
    SHUFFLE["random.shuffle(seed=42)\nReproducible ordering"]
    SPLIT_MERGE["Split by ratio\nval_split=0.15, test_split=0.10\nRemainder → train"]
    COPY["safe_copy()\nCopy image + label\nCreate parent dirs if needed"]
    YAML["Write data.yaml\nYOLO format with absolute paths\n6 class names"]
    WARN["Warn if no_helmet or no_vest\n< 2,000 annotations in train"]

    SRC --> COLLECT --> PREFIX --> SHUFFLE --> SPLIT_MERGE --> COPY --> YAML --> WARN
```

**Duplicate stem handling:** If two sources produce the same filename after prefixing,
`merge_datasets.py` appends an incrementing counter (`_1`, `_2`) to the stem.

**`count_annotations(label_path)`:** Reads a single YOLO `.txt` file and returns a
`Counter` of class IDs. Used for the post-merge distribution report.

---

### `make_raw_split.py` — Fine-Tuning Split

Creates `train_raw/` by copying only non-augmented files from `train/`.

**Filter logic:**
```python
if "_aug" not in image_path.name:
    copy image → train_raw/images/
    copy label → train_raw/labels/
```

This is a one-way copy — `train_raw/` is always regenerated from `train/` and never
modified directly. Run after every re-merge.

---

### `augment_dataset.py` — Offline Augmentation

Applies to `train/` only. Each augmented copy gets a new label file (identical to the original,
since all augmentations except horizontal flip are label-preserving).

**Horizontal flip label correction:**
```
For each annotation line: class cx cy w h
Flipped cx = 1.0 − cx
All other fields unchanged
```

**Augmentation pipeline per copy:**
```mermaid
flowchart LR
    IMG["Original image"]
    SHADOW{"p=0.40\nshadow?"}
    FOG{"p=0.20\nfog?"}
    BLUR{"p=0.30\nblur?"}
    BRIGHT{"p=0.80\nbrightness?"}
    NOISE{"p=0.30\nnoise?"}
    FLIP{"p=0.50\nflip?"}
    OUT["_aug{i}.jpg\n+ copied label\n(label flipped if flip applied)"]

    IMG --> SHADOW --> FOG --> BLUR --> BRIGHT --> NOISE --> FLIP --> OUT
```

Each augmentation is applied independently — multiple augmentations can combine on a single
copy. A copy that hits all six probabilities will have shadow + fog + blur + brightness
adjustment + noise + horizontal flip applied simultaneously.

---

## 10. Dataset Validation

### `validate_dataset.py` — Class Distribution Audit

Counts annotations across all splits and reports against class targets.

**Validation rules enforced:**
1. `no_vest` count vs `no_helmet` count imbalance must be ≤ 20%
2. Combined violation class ratio (`no_helmet` + `no_vest`) must be ≥ 28%

**Output format:**
```
Split Distribution
--------------------------------
 Split - Images - Annotations
--------------------------------
 train - 6517   - 19794
 valid - 1325   - 3245
 test  - 883    - 2600
--------------------------------

Class Distribution (all splits)
--------------------------------------------------------
 Class     - Annotations - %     - Target - Delta
--------------------------------------------------------
 helmet_on - 3350        - 16.9% - 860    - 2490
 no_helmet - 1835        - 9.3%  - 680    - 1155
 vest_on   - 7095        - 35.8% - 770    - 6325
 no_vest   - 2702        - 13.7% - 500    - 2202
 person    - 3726        - 18.8% - 940    - 2786
 mask_on   - 1086        - 5.5%  - 120    - 966
--------------------------------------------------------
Imbalance Warnings:
  WARNING: no_vest/no_helmet imbalance = 47% — target ≤ 20%
  WARNING: violation class ratio 23.0% below target 31%
```

**Note on targets:** The `TARGET` column reflects initial planning targets, not hard
requirements. The actual distribution is determined by available public data. Deviations
are documented and compensated at training time via `cls=1.5` class loss weighting.

---

### `full_stats.py` — Per-Source Breakdown

Provides a source-by-source view of annotation counts in the raw (pre-merge) datasets.
Useful for diagnosing which source is responsible for a class imbalance.

**Output sections:**
1. **Source breakdown** — per-source image count and per-class annotation count before merge
2. **Merged dataset** — per-split breakdown with raw vs. augmented image counts and class
   distribution with ASCII bar charts
3. **Dataset health** — global metrics: violation ratio, no_helmet/no_vest imbalance,
   person annotation count, val/train and test/train ratios

```mermaid
flowchart LR
    RAW_SOURCES["RAW_SOURCES dict\n7 source → remapped path"]
    COUNT_LABELS["count_labels(lbl_dir)\nRecursive .txt parse\nCounter per class ID"]
    COUNT_IMAGES["count_images(img_dir)\nCount .jpg/.png/.bmp"]
    PER_SOURCE["Per-source table\nimages + annotations"]
    MERGED_STATS["Merged dataset stats\nper split"]
    HEALTH["Health metrics\nviolation ratio\nimbalance warnings"]

    RAW_SOURCES --> COUNT_LABELS & COUNT_IMAGES --> PER_SOURCE --> MERGED_STATS --> HEALTH
```

---

*All inference thresholds referenced above are defined in `rules.yaml` and loaded via
`inference/constants.py`. No magic numbers appear in inference code.*
