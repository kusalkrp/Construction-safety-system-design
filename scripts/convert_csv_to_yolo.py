"""
Convert kaggle_safety_vests Pascal VOC CSV annotations to YOLO txt format.

The dataset (adilshamim8/safety-vests-detection-dataset) uses CSV annotations:
    filename, width, height, class, xmin, ymin, xmax, ymax

Classes found in audit:
    'Safety Vest'    → project ID 2 (vest_on)
    'NO-Safety Vest' → project ID 3 (no_vest)

Outputs YOLO txt files alongside images, preserving the split structure,
so merge_datasets.py can consume the result directly.

Usage:
    python scripts/convert_csv_to_yolo.py \\
        --source  dataset/raw/kaggle_safety_vests/ \\
        --output  dataset/remapped/kaggle_safety_vests/
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# CSV class name → project class ID
CLASS_MAP: dict[str, int] = {
    "Safety Vest": 2,    # vest_on
    "NO-Safety Vest": 3, # no_vest
}

SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}


def convert_split(csv_path: Path, img_src: Path, out_split: Path) -> None:
    """Convert one split's CSV annotations to YOLO txt files and copy images."""
    img_out = out_split / "images"
    lbl_out = out_split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Build per-image annotation list from CSV
    annotations: dict[str, list[str]] = {}
    skipped_classes: set[str] = set()

    with open(csv_path, encoding="utf-8") as f:
        header = f.readline()  # skip header row
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            filename, width, height, cls_name, xmin, ymin, xmax, ymax = (
                parts[0], int(parts[1]), int(parts[2]),
                parts[3], float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]),
            )
            cls_id = CLASS_MAP.get(cls_name, -1)
            if cls_id == -1:
                skipped_classes.add(cls_name)
                continue

            # Convert to YOLO normalised format
            cx = (xmin + xmax) / 2.0 / width
            cy = (ymin + ymax) / 2.0 / height
            w  = (xmax - xmin) / width
            h  = (ymax - ymin) / height
            yolo_line = f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

            annotations.setdefault(filename, []).append(yolo_line)

    if skipped_classes:
        logger.warning("  Skipped unknown class names: %s", skipped_classes)

    # Copy images and write label files
    n_images = 0
    n_no_label = 0
    cls_counts: dict[int, int] = {}

    for img_file in sorted(img_src.iterdir()):
        if img_file.suffix.lower() not in SUPPORTED_IMAGES:
            continue
        lines = annotations.get(img_file.name, [])
        lbl_file = lbl_out / (img_file.stem + ".txt")
        lbl_file.write_text("\n".join(lines) + ("\n" if lines else ""))
        if not lines:
            n_no_label += 1
        else:
            for line in lines:
                cls_id = int(line.split()[0])
                cls_counts[cls_id] = cls_counts.get(cls_id, 0) + 1
        shutil.copy2(img_file, img_out / img_file.name)
        n_images += 1

    logger.info("  %s: %d images  (%d with no annotations)", out_split.name, n_images, n_no_label)
    logger.info("  Class counts: %s", cls_counts)

    # Warn if critical violation classes are absent
    for cls_id, cls_name in [(2, "vest_on"), (3, "no_vest")]:
        if cls_counts.get(cls_id, 0) == 0:
            logger.warning("  WARNING: %s (id=%d) has 0 annotations in %s.", cls_name, cls_id, out_split.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert kaggle_safety_vests CSV to YOLO format")
    parser.add_argument("--source", required=True, help="Source dataset root (has train/valid/test splits)")
    parser.add_argument("--output", required=True, help="Output directory for YOLO-format dataset")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    split_map = {
        "train": "train",
        "valid": "valid",
        "test":  "test",
    }

    for split_name, out_name in split_map.items():
        # Dataset structure: train/train/, valid/valid/, test/test/
        split_dir = source / split_name / split_name
        csv_path  = split_dir / "_annotations.csv"

        if not split_dir.exists():
            logger.warning("Split dir not found, skipping: %s", split_dir)
            continue
        if not csv_path.exists():
            logger.warning("No _annotations.csv in %s, skipping.", split_dir)
            continue

        logger.info("Converting split: %s", split_name)
        convert_split(csv_path, split_dir, output / out_name)

    logger.info("Done. YOLO-format dataset written to: %s", output)
    logger.info("Next: run scripts/merge_datasets.py including dataset/remapped/kaggle_safety_vests/")


if __name__ == "__main__":
    main()
