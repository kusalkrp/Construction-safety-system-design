"""Batch test inference on test set images and report score distribution."""
import sys
sys.path.insert(0, ".")

import os
from pathlib import Path
import cv2
from dotenv import load_dotenv
from inference.pipeline import ConstructionSafetyPipeline

load_dotenv()
weights_path = os.getenv("MODEL_WEIGHTS_PATH", "runs/train4/weights/best.pt")
pipeline = ConstructionSafetyPipeline(weights_path=weights_path, conf_threshold=0.30)
test_dir = Path("dataset/merged/images/test")
images = list(test_dir.glob("*.jpg"))[:100]

bands = {}
zero_workers = 0
violations = {}
examples = []

for img_path in images:
    frame_bgr = cv2.imread(str(img_path))
    if frame_bgr is None:
        continue
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = pipeline.analyse(frame_rgb)
    band = result.score_result.band
    bands[band] = bands.get(band, 0) + 1
    if not result.site_report.worker_reports:
        zero_workers += 1
    for r in result.site_report.worker_reports:
        for v in r.violations:
            violations[v] = violations.get(v, 0) + 1
    if result.score_result.band != "COMPLIANT" and len(examples) < 5:
        examples.append((img_path.name, result.score_result.display, result.formatted_report))

print(f"\nScore band distribution ({len(images)} images):")
for band, count in sorted(bands.items()):
    print(f"  {band}: {count}")
print(f"\nImages with 0 workers detected: {zero_workers}")
print(f"\nViolation type counts: {violations}")
if examples:
    print("\n--- Non-compliant examples ---")
    for name, score, report in examples:
        print(f"\n{name} → {score}")
        print(report[:400])
else:
    print("\nNo non-compliant detections found in sample.")
