# Safety Rules Definition

## Overview

This document defines the 6 formal safety rules applied by the Construction Safety Monitor.

Each rule is specified with a precise, machine-checkable condition — not a vague description.
The design philosophy is: **express appropriate uncertainty rather than minimise false negatives
at the cost of usability**. The system flags what it knows, admits what it doesn't, and escalates
only when confidence warrants it.

All thresholds referenced below are defined in `rules.yaml` and loaded via `inference/constants.py`.
No magic numbers appear in inference code.

---

## Rule 1 — No Helmet in Active Zone

**Formal condition:**
`no_helmet` detection on a person bounding box with YOLO confidence ≥ 0.50,
where person bbox height ≥ 40px in a 640px input frame.

**Severity:** CRITICAL

**Severity escalation:**
If the person bounding box top edge (`y1`) is located in the upper 40% of the frame
(i.e. `y1 < frame_height × 0.60`), escalate severity to **CRITICAL-ELEVATED** —
indicating the worker is likely at height (scaffolding, elevated platform).

**What counts as a violation:**
- Worker on site floor or scaffolding with visible head and `no_helmet` detected
- Confidence ≥ 0.50 and bounding box height ≥ 40px

**What does NOT count:**
- Person bounding box height < 40px — far-field, classified as UNVERIFIABLE (Rule 4)
- Person detected at site perimeter (within 30px of frame edge) — classified as VISITOR

**Rationale:**
Head injuries are the leading cause of construction fatalities globally.
A worker at height without head protection represents the highest-risk PPE failure.

---

## Rule 2 — No High-Visibility Vest (Context-Aware)

**Formal condition:**
`no_vest` detection on a person bounding box with YOLO confidence ≥ 0.50.

**Severity (outdoor scene):** HIGH
**Severity (indoor scene):** WARNING

**Scene classification method:**
A heuristic examines the top 20% of the input frame in HSV colour space.
If ≥ 15% of sampled pixels fall in the sky-blue hue range (H: 90–140) with
saturation ≥ 30, the frame is classified as **outdoor**. Otherwise **indoor**.

This is a deliberate heuristic choice — zero additional model weight, zero inference cost,
and sufficiently accurate for risk modulation purposes.

**What counts as a violation:**
- Worker in active zone without hi-vis vest, in outdoor or high-traffic indoor scene
- Confidence ≥ 0.50

**What does NOT count:**
- Worker near site perimeter (visitor heuristic)

**Rationale:**
Hi-vis vests protect workers from vehicle and machinery strike.
Risk is context-dependent — outdoor means vehicle exposure; indoor means lower but non-zero risk.
The same absence carries different severity depending on where the worker is.

---

## Rule 3 — Partial Compliance (Low-Confidence Ambiguity)

**Formal condition:**
Any PPE item detected (helmet or vest) where the YOLO confidence is
between 0.35 and 0.65 (inclusive).

**Severity:** WARNING

**Output:** Flag as `PARTIAL_COMPLIANCE_SUSPECTED`. Add to human review queue.
Do not generate a CRITICAL or HIGH alert.

**What this detects:**
Vest worn open, helmet not fastened, PPE partially removed — conditions where
the model sees something PPE-shaped but cannot confirm it is correctly worn.

**Rationale:**
Partial compliance is a real safety risk but cannot be reliably confirmed by
a detection model alone. The honest response is to flag uncertainty, not guess.
A forced binary classification in this confidence range would produce unreliable alerts.

---

## Rule 4 — Far-Field Worker Unverifiable

**Formal condition:**
Person detected with bounding box height < 40px in a 640px input
(approximately 6% of frame height — worker too distant to classify PPE reliably).

**Severity:** UNVERIFIABLE (capability flag — not a violation flag)

**Output:** `PPE_STATE_UNVERIFIABLE — worker at distance, PPE cannot be assessed.`

**What this prevents:**
- Silently treating undetectable workers as compliant (dangerous — false safety)
- Flagging them as violations (alert fatigue — false alarm)

The system admits its limit rather than guessing in either direction.

**Production upgrade path:**
PTZ camera with auto-zoom triggered by person detection, or stereoscopic
depth sensing to compensate for far-field detection gaps.

---

## Rule 5 — Occlusion / Detection Gap

**Formal condition:**
Person bounding box detected, but no PPE bounding boxes overlap the person
region (IoU > 0.10 threshold with any PPE detection).

**Severity:** WARNING

**Output:** `PPE detection failed for worker — possible occlusion or detection gap.
Human review recommended.`

**What this prevents:**
A false negative from the PPE detector silently passing as "compliant."
The system flags the absence of evidence rather than treating it as evidence of compliance.
The distinction between "no PPE detected" and "PPE confirmed absent" is safety-critical.

**Rationale:**
Workers may be occluded behind equipment, other workers, or scaffolding members.
The model cannot distinguish between "PPE not present" and "PPE not visible."
Flagging this gap for human review is the correct response.

---

## Rule 6 — Site-Level Crowd Non-Compliance

**Formal condition:**
≥ 4 workers detected in a single frame AND ≥ 50% of detected workers have at least
one active violation (Rule 1 or Rule 2) with rule_confidence ≥ 0.60.

**Severity:** SITE ALERT (escalated from individual violation alerts)

**Output:**
Replaces individual worker alerts with a single site-level report:
```
[SITE ALERT] Systemic PPE non-compliance detected
  Workers detected: N
  Workers non-compliant: X (Y%)
  Active violations: no_helmet ×N, no_vest ×N
  Compliance score: Z/100 — CRITICAL
  Recommended action: Site-wide work stoppage for PPE briefing.
  Supervisor: review site PPE supply immediately.
```

**Crowd multiplier:**
When Rule 6 triggers, the compliance score deduction is multiplied by 1.3.
Systemic failure is weighted more severely than individual failures.

**Rationale:**
A site manager does not need 20 individual alerts for a zone-wide compliance failure.
They need to know when a zone has a systemic problem. This rule maps to how safety
managers actually think about risk — systemic failures require systemic responses.

---

## Rule Summary Table

| Rule | Trigger | Severity | Output Type |
|---|---|---|---|
| 1 — No helmet | `no_helmet` conf ≥ 0.50, bbox ≥ 40px | CRITICAL | Alert |
| 1 (elevated) | same, upper 40% of frame | CRITICAL-ELEVATED | Alert |
| 2 — No vest (outdoor) | `no_vest` conf ≥ 0.50, outdoor scene | HIGH | Alert |
| 2 — No vest (indoor) | `no_vest` conf ≥ 0.50, indoor scene | WARNING | Log |
| 3 — Partial compliance | Any PPE conf 0.35–0.65 | WARNING | Review queue |
| 4 — Far-field | Person bbox height < 40px | UNVERIFIABLE | Info flag |
| 5 — Occlusion gap | Person detected, no PPE overlap | WARNING | Review queue |
| 6 — Crowd | ≥4 workers, ≥50% violating | SITE ALERT | Escalated alert |

---

## Rules Configuration (`rules.yaml`)

All thresholds are defined in `rules.yaml` at the project root.
`SafetyChecker` loads them via `inference/constants.py` at import time.
This means a safety engineer can audit and adjust all thresholds
without reading any Python code.

Key thresholds:
```yaml
detection:
  violation_conf_min: 0.50
  partial_compliance_conf_low: 0.35
  partial_compliance_conf_high: 0.65

bbox:
  far_field_height_px: 40
  ppe_person_overlap_iou: 0.10

frame:
  elevation_zone_ratio: 0.60
  visitor_edge_px: 30

crowd:
  min_workers: 4
  violation_ratio_threshold: 0.50
  crowd_violation_conf_min: 0.60
```
