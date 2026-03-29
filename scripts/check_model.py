"""Check model class names and raw detections on a sample image."""
import sys
sys.path.insert(0, ".")

from pathlib import Path
import cv2
from ultralytics import YOLO

model = YOLO("runs/train4/weights/best.pt")
print("Model class names:", model.names)

# Run raw YOLO on a zero-worker image to see what it actually detects
test_imgs = list(Path("dataset/merged/images/test").glob("*.jpg"))

print("\n--- Raw YOLO detections (conf=0.20) on first 5 images ---")
for img_path in test_imgs[:5]:
    frame_bgr = cv2.imread(str(img_path))
    results = model(frame_bgr, conf=0.20, verbose=False)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            dets.append(f"{model.names[cls_id]}({conf:.2f})")
    print(f"  {img_path.name}: {dets if dets else 'NO DETECTIONS'}")
