"""
Merge multiple remapped datasets into one unified dataset.

Each source is prefixed with its directory name to avoid filename collisions.
All pairs are shuffled with a fixed seed then split into train/val/test.

Usage:
    python scripts/merge_datasets.py \\
        --sources \\
            dataset/raw/roboflow_base/ \\
            dataset/remapped/kaggle_ppe_kit/ \\
            dataset/remapped/kaggle_safety_vests/ \\
            dataset/remapped/roboflow_scaffolding/ \\
        --output dataset/merged/ \\
        --val-split 0.15 \\
        --test-split 0.10
"""

import argparse
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_CLASSES = ["helmet_on", "no_helmet", "vest_on", "no_vest", "person", "mask_on"]
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}
RANDOM_SEED = 42


def collect_pairs(source: Path) -> list[tuple[Path, Path | None]]:
    """
    Collect all (image, label) pairs from a dataset directory.
    Handles split (train/valid/test) and flat layouts.
    Returns list of (image_path, label_path_or_None).
    """
    pairs: list[tuple[Path, Path | None]] = []
    split_names = {"train", "valid", "val", "test"}

    subdirs = [d for d in source.iterdir() if d.is_dir() and d.name in split_names]
    search_roots = subdirs if subdirs else [source]

    for root in search_roots:
        img_dir = root / "images"
        lbl_dir = root / "labels"

        if not img_dir.exists():
            # Try flat layout
            img_dir = root
            lbl_dir = root

        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in SUPPORTED_IMAGES:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            pairs.append((img, lbl if lbl.exists() else None))

    return pairs


def count_annotations(label_path: Path | None) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    if label_path is None or not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if parts:
            cls_id = int(parts[0])
            if 0 <= cls_id < len(PROJECT_CLASSES):
                counts[cls_id] += 1
    return counts


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple remapped datasets into one")
    parser.add_argument("--sources", nargs="+", required=True, help="List of remapped dataset directories")
    parser.add_argument("--output", required=True, help="Output merged dataset directory")
    parser.add_argument("--val-split", type=float, default=0.15, help="Val fraction (default 0.15)")
    parser.add_argument("--test-split", type=float, default=0.10, help="Test fraction (default 0.10)")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # -- Collect all pairs from all sources ------------------------------------
    all_pairs: list[tuple[str, Path, Path | None]] = []  # (prefix, img, lbl)

    for src_str in args.sources:
        src = Path(src_str)
        if not src.exists():
            logger.warning("Source not found, skipping: %s", src)
            continue
        prefix = src.name  # e.g. "kaggle_sh17", "roboflow_base"
        pairs = collect_pairs(src)
        logger.info("  %s: %d image-label pairs", prefix, len(pairs))
        for img, lbl in pairs:
            all_pairs.append((prefix, img, lbl))

    if not all_pairs:
        raise ValueError("No image-label pairs found across all sources. Check --sources paths.")

    logger.info("Total pairs collected: %d", len(all_pairs))

    # -- Shuffle and split -----------------------------------------------------
    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)

    n_total = len(all_pairs)
    n_test = max(1, int(n_total * args.test_split))
    n_val  = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val - n_test

    test_pairs  = all_pairs[:n_test]
    val_pairs   = all_pairs[n_test:n_test + n_val]
    train_pairs = all_pairs[n_test + n_val:]

    logger.info("Split: train=%d  val=%d  test=%d", len(train_pairs), len(val_pairs), len(test_pairs))

    # -- Copy files into merged structure --------------------------------------
    split_counts: dict[str, dict] = {}

    for split_name, pairs in [("train", train_pairs), ("valid", val_pairs), ("test", test_pairs)]:
        img_out = output / "images" / split_name
        lbl_out = output / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        cls_counts: dict[int, int] = defaultdict(int)
        seen_names: dict[str, int] = defaultdict(int)

        for prefix, img_path, lbl_path in pairs:
            # Prefix filename to avoid collisions: kaggle_sh17_img0001.jpg
            stem = f"{prefix}_{img_path.stem}"
            # Handle duplicate stems from same source
            seen_names[stem] += 1
            if seen_names[stem] > 1:
                stem = f"{stem}_{seen_names[stem]}"

            dst_img = img_out / (stem + img_path.suffix.lower())
            dst_lbl = lbl_out / (stem + ".txt")

            safe_copy(img_path, dst_img)

            if lbl_path and lbl_path.exists():
                safe_copy(lbl_path, dst_lbl)
                for cls_id, cnt in count_annotations(lbl_path).items():
                    cls_counts[cls_id] += cnt
            else:
                dst_lbl.write_text("")

        split_counts[split_name] = dict(cls_counts)

    # -- Write data.yaml -------------------------------------------------------
    config = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val":   "images/valid",
        "test":  "images/test",
        "nc": len(PROJECT_CLASSES),
        "names": PROJECT_CLASSES,
    }
    with open(output / "data.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # -- Summary ---------------------------------------------------------------
    print()
    print("-- Merge Summary " + "-" * 44)
    print(f"  Total images collected : {n_total:,}")
    print(f"  Train split            : {len(train_pairs):,}")
    print(f"  Val split              : {len(val_pairs):,}")
    print(f"  Test split             : {len(test_pairs):,}")
    print()
    print("  Class distribution (train):")
    train_counts = split_counts.get("train", {})
    for cls_id, cls_name in enumerate(PROJECT_CLASSES):
        cnt = train_counts.get(cls_id, 0)
        cls_type = "!" if cls_name in ("no_helmet", "no_vest") else " "
        print(f"  {cls_type} {cls_name:<15} : {cnt:>6,}")
    print("-" * 60)

    # Warn on insufficient violation class counts
    for cls_id, cls_name, target in [(1, "no_helmet", 2000), (3, "no_vest", 2000)]:
        cnt = train_counts.get(cls_id, 0)
        if cnt < target:
            logger.warning(
                "%s has only %d train annotations (target >= %d). "
                "Check remapping for this class across all sources.",
                cls_name, cnt, target,
            )

    print(f"\nMerged dataset written to: {output}")
    print(f"data.yaml: {output / 'data.yaml'}")
    print("\nNext steps:")
    print("  1. python scripts/validate_dataset.py --data dataset/merged/data.yaml")
    print("  2. python scripts/augment_dataset.py  --data dataset/merged/data.yaml --factor 2")


if __name__ == "__main__":
    main()
