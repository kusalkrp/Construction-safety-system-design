"""
Merge custom labelled images into the remapped Roboflow dataset.
Custom images are split 90% train / 10% val (no test — test stays clean Roboflow only).

Usage:
    python scripts/merge_custom_labels.py \
        --custom dataset/custom_labelled/ \
        --data dataset/remapped/data.yaml \
        [--val-split 0.1]
"""

import argparse
import logging
import random
import shutil
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png"}


def load_data_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_image_label_pairs(custom_dir: Path) -> list[tuple[Path, Path | None]]:
    """Return (image_path, label_path_or_None) pairs from custom_labelled/."""
    images_dir = custom_dir / "images"
    labels_dir = custom_dir / "labels"

    if not images_dir.exists():
        # flat structure — images directly in custom_dir
        images_dir = custom_dir
        labels_dir = custom_dir

    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in SUPPORTED_IMAGES:
            continue
        lbl = labels_dir / (img.stem + ".txt")
        pairs.append((img, lbl if lbl.exists() else None))

    return pairs


def copy_pair(img: Path, lbl: Path | None, dest_img_dir: Path, dest_lbl_dir: Path) -> None:
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, dest_img_dir / img.name)
    if lbl:
        shutil.copy2(lbl, dest_lbl_dir / (img.stem + ".txt"))
    else:
        (dest_lbl_dir / (img.stem + ".txt")).write_text("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge custom labelled images into dataset splits")
    parser.add_argument("--custom", required=True, help="Path to custom labelled directory")
    parser.add_argument("--data", default="dataset/remapped/data.yaml", help="Path to data.yaml")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction for val (default 0.1)")
    args = parser.parse_args()

    custom_dir = Path(args.custom)
    data_yaml_path = Path(args.data)

    if not custom_dir.exists():
        raise FileNotFoundError(f"Custom labelled dir not found: {custom_dir}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    config = load_data_yaml(data_yaml_path)
    dataset_root = Path(config["path"])

    pairs = find_image_label_pairs(custom_dir)
    if not pairs:
        raise ValueError(f"No images found in {custom_dir}")

    logger.info("Found %d custom images", len(pairs))

    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_split))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    logger.info("Splitting: %d train / %d val", len(train_pairs), len(val_pairs))

    for img, lbl in train_pairs:
        copy_pair(img, lbl, dataset_root / "train" / "images", dataset_root / "train" / "labels")

    for img, lbl in val_pairs:
        copy_pair(img, lbl, dataset_root / "valid" / "images", dataset_root / "valid" / "labels")

    # Report final counts
    for split in ("train", "valid", "test"):
        split_img_dir = dataset_root / split / "images"
        count = len(list(split_img_dir.glob("*"))) if split_img_dir.exists() else 0
        logger.info("  %s: %d images total", split, count)

    logger.info("Merge complete. Next: run scripts/augment_dataset.py")


if __name__ == "__main__":
    main()
