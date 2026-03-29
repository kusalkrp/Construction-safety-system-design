"""
Remap a single dataset's label files to the project 6-class schema.

Reads a mapping YAML (source_class_id → target_class_id, -1 = discard),
walks all label files in the source dataset, rewrites them with remapped
class IDs, and copies corresponding images to the output directory.

Usage:
    python scripts/remap_labels.py \\
        --source   dataset/raw/kaggle_ppe_kit/ \\
        --mapping  dataset/mappings/kaggle_ppe_kit.yaml \\
        --output   dataset/remapped/kaggle_ppe_kit/

Logs per-class annotation counts before and after remapping so you can
verify the mapping is correct before proceeding to merge.
"""

import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_CLASSES = ["helmet_on", "no_helmet", "vest_on", "no_vest", "person", "mask_on"]
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}


def load_mapping(mapping_path: Path) -> dict[int, int]:
    """Load mapping YAML. Returns {source_id: target_id} — -1 means discard."""
    with open(mapping_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("already_remapped"):
        logger.info("Mapping marked already_remapped — identity pass only.")

    raw = cfg.get("mapping", {})
    if not raw:
        raise ValueError(f"No 'mapping' key found in {mapping_path}")

    return {int(k): int(v) for k, v in raw.items()}


def find_dataset_roots(source: Path) -> tuple[Path, Path]:
    """
    Find images/ and labels/ directories — handles flat and nested structures.
    Returns (images_dir, labels_dir).
    """
    # Common layouts:
    # 1. source/images/ + source/labels/
    # 2. source/train/images/ + source/train/labels/  (split already)
    # 3. source/<subdir>/images/ + source/<subdir>/labels/

    candidates = [source] + (list(source.iterdir()) if source.is_dir() else [])
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        img = candidate / "images"
        lbl = candidate / "labels"
        if img.exists() and lbl.exists():
            return img, lbl

    # Try one level deeper
    for sub in source.rglob("images"):
        lbl = sub.parent / "labels"
        if lbl.exists():
            return sub, lbl

    raise FileNotFoundError(
        f"Could not find images/ + labels/ dirs under {source}. "
        "Check the dataset structure and adjust --source path."
    )


def remap_file(
    lbl_src: Path,
    lbl_dst: Path,
    mapping: dict[int, int],
    src_counts: dict[int, int],
    dst_counts: dict[int, int],
) -> int:
    """Remap one label file. Returns number of kept annotations."""
    lines_out = []
    for line in lbl_src.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        src_id = int(parts[0])
        src_counts[src_id] += 1
        dst_id = mapping.get(src_id, -1)
        if dst_id == -1:
            continue
        lines_out.append(f"{dst_id} {' '.join(parts[1:])}")
        dst_counts[dst_id] += 1

    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return len(lines_out)


def remap_dataset(source: Path, mapping: dict[int, int], output: Path) -> None:
    """Remap entire dataset — supports split, nested-split, and flat layouts."""
    SPLIT_NAMES = {"train", "valid", "val", "test"}

    # Layout 1: source/<split>/ directly (e.g. source/train/, source/valid/)
    split_dirs = [d for d in source.iterdir() if d.is_dir() and d.name in SPLIT_NAMES]

    if split_dirs:
        for split_dir in split_dirs:
            split_name = "valid" if split_dir.name in ("valid", "val") else split_dir.name
            try:
                img_src, lbl_src_dir = find_dataset_roots(split_dir)
            except FileNotFoundError:
                logger.warning("Skipping split '%s' — images/labels not found", split_dir.name)
                continue
            _remap_split(img_src, lbl_src_dir, output / split_name, mapping)
        return

    # Layout 2: source/images/<split>/ + source/labels/<split>/
    img_root = source / "images"
    lbl_root = source / "labels"
    if img_root.exists() and lbl_root.exists():
        nested_splits = [d for d in img_root.iterdir() if d.is_dir() and d.name in SPLIT_NAMES]
        if nested_splits:
            for split_dir in nested_splits:
                split_name = "valid" if split_dir.name in ("valid", "val") else split_dir.name
                lbl_split = lbl_root / split_dir.name
                if not lbl_split.exists():
                    logger.warning("Skipping split '%s' — matching labels dir not found", split_dir.name)
                    continue
                _remap_split(split_dir, lbl_split, output / split_name, mapping)
            return

    # Layout 3: flat — all images at same level, treat as train
    try:
        img_src, lbl_src_dir = find_dataset_roots(source)
    except FileNotFoundError:
        raise
    _remap_split(img_src, lbl_src_dir, output / "train", mapping)


def _remap_split(
    img_src: Path,
    lbl_src_dir: Path,
    out_split: Path,
    mapping: dict[int, int],
) -> None:
    img_dst = out_split / "images"
    lbl_dst_dir = out_split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    src_counts: dict[int, int] = defaultdict(int)
    dst_counts: dict[int, int] = defaultdict(int)
    n_images = 0
    n_skipped = 0

    for img_file in sorted(img_src.iterdir()):
        if img_file.suffix.lower() not in SUPPORTED_IMAGES:
            continue

        lbl_file = lbl_src_dir / (img_file.stem + ".txt")
        lbl_out = lbl_dst_dir / (img_file.stem + ".txt")

        kept = 0
        if lbl_file.exists():
            kept = remap_file(lbl_file, lbl_out, mapping, src_counts, dst_counts)
        else:
            lbl_out.write_text("")

        if kept == 0 and lbl_file.exists():
            # All annotations discarded — skip image to avoid empty-label noise
            lbl_out.unlink(missing_ok=True)
            n_skipped += 1
            continue

        shutil.copy2(img_file, img_dst / img_file.name)
        n_images += 1

    split_name = out_split.name
    logger.info("  %s: %d images kept (%d skipped — all annotations discarded)", split_name, n_images, n_skipped)

    # Per-class report
    logger.info("  Source class counts: %s", dict(sorted(src_counts.items())))
    mapped_names = {id_: PROJECT_CLASSES[id_] for id_ in range(len(PROJECT_CLASSES))}
    logger.info("  Output class counts: %s",
                {f"{id_}({mapped_names.get(id_, '?')})": cnt for id_, cnt in sorted(dst_counts.items())})

    # Warn if any expected violation class is zero
    for cls_id, cls_name in [(1, "no_helmet"), (3, "no_vest")]:
        if dst_counts.get(cls_id, 0) == 0:
            logger.warning(
                "  WARNING: %s (id=%d) has 0 annotations in %s. "
                "Check mapping YAML — is the source class ID correct?",
                cls_name, cls_id, split_name,
            )


def write_data_yaml(output: Path) -> None:
    config = {
        "path": str(output.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(PROJECT_CLASSES),
        "names": PROJECT_CLASSES,
    }
    with open(output / "data.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap dataset labels to project 6-class schema")
    parser.add_argument("--source",  required=True, help="Source dataset root directory")
    parser.add_argument("--mapping", required=True, help="Path to mapping YAML file")
    parser.add_argument("--output",  required=True, help="Output directory for remapped dataset")
    args = parser.parse_args()

    source = Path(args.source)
    mapping_path = Path(args.mapping)
    output = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    mapping = load_mapping(mapping_path)
    logger.info("Loaded mapping from %s: %d entries", mapping_path.name, len(mapping))
    logger.info("Remapping: %s", {k: PROJECT_CLASSES[v] if v >= 0 else "DISCARD" for k, v in sorted(mapping.items())})

    output.mkdir(parents=True, exist_ok=True)
    remap_dataset(source, mapping, output)
    write_data_yaml(output)

    logger.info("Done. Remapped dataset written to: %s", output)
    logger.info("Next: verify counts above, then run scripts/merge_datasets.py")


if __name__ == "__main__":
    main()
