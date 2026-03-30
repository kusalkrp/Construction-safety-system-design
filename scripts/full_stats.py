"""Full dataset statistics — per-source breakdown, aug ratio, split health."""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from collections import defaultdict
import yaml

CLASS_NAMES = ["helmet_on", "no_helmet", "vest_on", "no_vest", "person", "mask_on"]
DATA_YAML = Path("dataset/merged/data.yaml")
MERGED = Path("dataset/merged")
RAW_SOURCES = {
    "roboflow_base":    Path("dataset/remapped/roboflow_base"),
    "kaggle_ppe_kit":   Path("dataset/remapped/kaggle_ppe_kit"),
    "kaggle_safety_vests": Path("dataset/remapped/kaggle_safety_vests"),
    "roboflow_no_vest": Path("dataset/remapped/roboflow_no_vest"),
    "roboflow_scaffolding": Path("dataset/remapped/roboflow_scaffolding"),
    "roboflow_worker":  Path("dataset/remapped/roboflow_worker"),
    "roboflow_no_helmet": Path("dataset/remapped/roboflow_no_helmet"),
}

def count_labels(lbl_dir: Path) -> dict[int, int]:
    counts = defaultdict(int)
    if not lbl_dir.exists():
        return counts
    for f in lbl_dir.rglob("*.txt"):
        for line in f.read_text(errors="ignore").splitlines():
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1
    return counts

def count_images(img_dir: Path) -> int:
    if not img_dir.exists():
        return 0
    return sum(1 for f in img_dir.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})

# ── Per-source breakdown ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("SOURCE BREAKDOWN (remapped, pre-augmentation)")
print("="*70)
print(f"{'Source':<25} {'Images':>7}  " + "  ".join(f"{c[:9]:<9}" for c in CLASS_NAMES))
print("-"*70)
source_totals = defaultdict(int)
for name, path in RAW_SOURCES.items():
    imgs = count_images(path)
    counts = count_labels(path / "train" / "labels")
    for split in ("valid", "test"):
        c = count_labels(path / split / "labels")
        for k, v in c.items():
            counts[k] += v
    row = f"{name:<25} {imgs:>7}  " + "  ".join(f"{counts.get(i,0):<9}" for i in range(6))
    print(row)
    source_totals["images"] += imgs
    for i in range(6):
        source_totals[f"cls_{i}"] += counts.get(i, 0)
print("-"*70)
print(f"{'TOTAL':<25} {source_totals['images']:>7}  " + "  ".join(f"{source_totals[f'cls_{i}']:<9}" for i in range(6)))

# ── Merged split summary ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("MERGED DATASET (post-augmentation)")
print("="*70)
for split in ("train", "valid", "test"):
    img_dir = MERGED / "images" / split
    lbl_dir = MERGED / "labels" / split
    imgs = count_images(img_dir)
    counts = count_labels(lbl_dir)
    total_ann = sum(counts.values())
    aug = sum(1 for f in img_dir.rglob("*.jpg") if "_aug" in f.name) if img_dir.exists() else 0
    raw = imgs - aug
    print(f"\n  {split.upper()}")
    print(f"    Images      : {imgs:,}  (raw={raw:,}  augmented={aug:,})")
    print(f"    Annotations : {total_ann:,}")
    for i, name in enumerate(CLASS_NAMES):
        cnt = counts.get(i, 0)
        pct = cnt / total_ann * 100 if total_ann else 0
        bar = "#" * int(pct / 2)
        print(f"    {name:<15}: {cnt:>6,}  ({pct:5.1f}%)  {bar}")

# ── Overall health ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("DATASET HEALTH")
print("="*70)
train_lbl = count_labels(MERGED / "labels" / "train")
total = sum(train_lbl.values())
no_helmet = train_lbl.get(1, 0)
no_vest   = train_lbl.get(3, 0)
person    = train_lbl.get(4, 0)
violation_ratio = (no_helmet + no_vest) / total * 100 if total else 0

print(f"  Total train annotations : {total:,}")
print(f"  Violation ratio         : {violation_ratio:.1f}%  (target ~31%)")
print(f"  no_helmet / no_vest     : {no_helmet:,} / {no_vest:,}  (imbalance={(abs(no_vest-no_helmet)/max(no_vest,no_helmet)*100):.0f}%)")
print(f"  person annotations      : {person:,}")
print(f"  Val/Train ratio         : {count_images(MERGED/'images'/'valid') / count_images(MERGED/'images'/'train') * 100:.1f}%")
print(f"  Test/Train ratio        : {count_images(MERGED/'images'/'test')  / count_images(MERGED/'images'/'train') * 100:.1f}%")
print()
