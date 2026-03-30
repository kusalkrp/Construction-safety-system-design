"""
Download Construction Site Safety dataset from Roboflow (v12) and remap
the original 10 classes to the project's 6-class schema.

Usage:
    python scripts/download_dataset.py --api-key YOUR_KEY [--output dataset/]
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Class remapping: Roboflow original → project schema ──────────────────────
# Roboflow v12 class indices (0-based, alphabetical by default):
# 0=Hardhat, 1=Mask, 2=NO-Hardhat, 3=NO-Mask, 4=NO-Safety Vest,
# 5=Person, 6=Safety Cone, 7=Safety Vest, 8=machinery, 9=vehicle
ROBOFLOW_TO_PROJECT: dict[int, int | None] = {
    0: 0,    # Hardhat       → helmet_on
    1: 5,    # Mask          → mask_on
    2: 1,    # NO-Hardhat    → no_helmet
    3: None, # NO-Mask       → discard
    4: 3,    # NO-Safety Vest→ no_vest
    5: 4,    # Person        → person
    6: None, # Safety Cone   → discard
    7: 2,    # Safety Vest   → vest_on
    8: None, # machinery     → discard
    9: None, # vehicle       → discard
}

PROJECT_CLASSES = ["helmet_on", "no_helmet", "vest_on", "no_vest", "person", "mask_on"]

ROBOFLOW_WORKSPACE = "roboflow-universe-projects"
ROBOFLOW_PROJECT = "construction-site-safety"
ROBOFLOW_VERSION = 12   # preferred; falls back to latest if not found


def download_from_roboflow(api_key: str, output_dir: Path) -> Path:
    """Download dataset via Roboflow Python SDK."""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError(
            "roboflow package not installed. Run: pip install roboflow"
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)

    # Try preferred version; fall back to latest available
    version = None
    try:
        version = project.version(ROBOFLOW_VERSION)
        logger.info("Using dataset v%d", ROBOFLOW_VERSION)
    except RuntimeError:
        versions = project.versions()
        if not versions:
            raise RuntimeError(f"No versions found for project '{ROBOFLOW_PROJECT}'.")
        latest = versions[-1]
        logger.warning(
            "Version %d not found. Available versions: %s. Using v%s.",
            ROBOFLOW_VERSION,
            [v.version for v in versions],
            latest.version,
        )
        version = latest

    dataset = version.download("yolov8", location=str(output_dir / "roboflow_raw"))
    logger.info("Downloaded to %s", dataset.location)
    return Path(dataset.location)


def remap_label_file(src: Path, dst: Path) -> int:
    """Remap a single YOLO label file. Returns number of kept annotations."""
    kept = 0
    lines_out: list[str] = []

    for line in src.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        original_cls = int(parts[0])
        new_cls = ROBOFLOW_TO_PROJECT.get(original_cls)
        if new_cls is None:
            continue
        lines_out.append(f"{new_cls} {' '.join(parts[1:])}")
        kept += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return kept


def remap_split(raw_root: Path, out_root: Path, split: str) -> dict[str, int]:
    """Remap labels for one split (train/valid/test). Copy images unchanged."""
    img_src = raw_root / split / "images"
    lbl_src = raw_root / split / "labels"
    img_dst = out_root / split / "images"
    lbl_dst = out_root / split / "labels"

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {"images": 0, "annotations": 0, "discarded": 0}

    if not img_src.exists():
        logger.warning("Split '%s' not found at %s — skipping", split, img_src)
        return stats

    for img_file in img_src.iterdir():
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        shutil.copy2(img_file, img_dst / img_file.name)
        stats["images"] += 1

        lbl_file = lbl_src / (img_file.stem + ".txt")
        if lbl_file.exists():
            kept = remap_label_file(lbl_file, lbl_dst / lbl_file.name)
            stats["annotations"] += kept
        else:
            (lbl_dst / (img_file.stem + ".txt")).write_text("")

    logger.info("  %s: %d images, %d annotations kept", split, stats["images"], stats["annotations"])
    return stats


def write_data_yaml(out_root: Path) -> None:
    """Write data.yaml for the remapped dataset."""
    config = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(PROJECT_CLASSES),
        "names": PROJECT_CLASSES,
    }
    yaml_path = out_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info("Wrote %s", yaml_path)


# Name-based remapping — robust across dataset versions regardless of class order.
# Maps Roboflow class NAME → our project class ID (or None to discard).
NAME_TO_PROJECT: dict[str, int | None] = {
    "Hardhat":        0,   # → helmet_on
    "Mask":           5,   # → mask_on
    "NO-Hardhat":     1,   # → no_helmet
    "NO-Mask":        None,
    "NO-Safety Vest": 3,   # → no_vest
    "Person":         4,   # → person
    "Safety Cone":    None,
    "Safety Vest":    2,   # → vest_on
    "machinery":      None,
    "vehicle":        None,
}


def build_index_remapping(raw_root: Path) -> dict[int, int | None]:
    """
    Read the downloaded data.yaml to get actual class order, then build
    index→index remapping. Robust to version differences in class ordering.
    """
    data_yaml = raw_root / "data.yaml"
    if not data_yaml.exists():
        # Some versions place it one level up
        data_yaml = raw_root.parent / "data.yaml"

    if not data_yaml.exists():
        logger.warning("data.yaml not found in downloaded dataset — using hardcoded index order.")
        return ROBOFLOW_TO_PROJECT

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    source_names: list[str] = cfg.get("names", [])
    if not source_names:
        logger.warning("No class names in downloaded data.yaml — using hardcoded index order.")
        return ROBOFLOW_TO_PROJECT

    logger.info("Downloaded dataset classes (%d): %s", len(source_names), source_names)

    remapping: dict[int, int | None] = {}
    for idx, name in enumerate(source_names):
        remapping[idx] = NAME_TO_PROJECT.get(name)
        if name not in NAME_TO_PROJECT:
            logger.warning("Unknown class '%s' (index %d) — will be discarded.", name, idx)

    mapped = {name: remapping[i] for i, name in enumerate(source_names) if remapping.get(i) is not None}
    logger.info("Remapping: %s", mapped)
    return remapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and remap Roboflow dataset")
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"), help="Roboflow API key")
    parser.add_argument("--output", default="dataset", help="Output directory (default: dataset/)")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Roboflow API key required. Set ROBOFLOW_API_KEY in .env or pass --api-key")

    out_root = Path(args.output)
    raw_root = download_from_roboflow(args.api_key, out_root)

    # Build remapping from actual downloaded class names (version-safe)
    global ROBOFLOW_TO_PROJECT
    ROBOFLOW_TO_PROJECT = build_index_remapping(raw_root)

    logger.info("Remapping classes → 6 project classes")
    remapped_root = out_root / "remapped"

    for split in ("train", "valid", "test"):
        remap_split(raw_root, remapped_root, split)

    write_data_yaml(remapped_root)
    logger.info("Done. Remapped dataset at: %s", remapped_root)
    logger.info("Classes: %s", PROJECT_CLASSES)


if __name__ == "__main__":
    main()
