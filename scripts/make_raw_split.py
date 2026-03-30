"""Copy only non-augmented images+labels into train_raw split for fine-tuning."""
import shutil
from pathlib import Path

merged = Path("dataset/merged")
src_img = merged / "images" / "train"
src_lbl = merged / "labels" / "train"
dst_img = merged / "images" / "train_raw"
dst_lbl = merged / "labels" / "train_raw"

dst_img.mkdir(parents=True, exist_ok=True)
dst_lbl.mkdir(parents=True, exist_ok=True)

copied = 0
for img in sorted(src_img.glob("*.jpg")):
    if "_aug" in img.name:
        continue
    lbl = src_lbl / (img.stem + ".txt")
    shutil.copy2(img, dst_img / img.name)
    if lbl.exists():
        shutil.copy2(lbl, dst_lbl / lbl.name)
    copied += 1

print(f"Copied {copied} raw images to train_raw/")
