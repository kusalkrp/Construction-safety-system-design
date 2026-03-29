"""Diagnose zero-worker detections — check if person is detected but below threshold."""
import sys
sys.path.insert(0, ".")

from pathlib import Path
import cv2
from ultralytics import YOLO

model = YOLO("runs/train4/weights/best.pt")
test_imgs = list(Path("dataset/merged/images/test").glob("*.jpg"))

no_person_at_all = 0
person_below_30 = 0
person_above_30 = 0

person_confs = []

for img_path in test_imgs[:200]:
    frame_bgr = cv2.imread(str(img_path))
    if frame_bgr is None:
        continue
    results = model(frame_bgr, conf=0.01, verbose=False)
    found_person = False
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == "person":
                conf = float(box.conf[0])
                person_confs.append(conf)
                found_person = True
                if conf >= 0.30:
                    person_above_30 += 1
                else:
                    person_below_30 += 1
    if not found_person:
        no_person_at_all += 1

print(f"Sampled 200 images:")
print(f"  No person detection at any conf:  {no_person_at_all}")
print(f"  Person detected but conf < 0.30:  {person_below_30}")
print(f"  Person detected with conf >= 0.30: {person_above_30}")

if person_confs:
    person_confs.sort()
    print(f"\nPerson confidence distribution:")
    print(f"  min={min(person_confs):.2f}  max={max(person_confs):.2f}  median={person_confs[len(person_confs)//2]:.2f}")
    buckets = [0]*10
    for c in person_confs:
        buckets[min(int(c*10), 9)] += 1
    for i, cnt in enumerate(buckets):
        print(f"  {i/10:.1f}-{(i+1)/10:.1f}: {'#'*cnt} ({cnt})")
