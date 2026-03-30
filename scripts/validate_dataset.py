"""
Validate dataset: report per-class annotation counts, split stats,
and flag class imbalance warnings.

Output feeds into docs/DATASET.md.

Usage:
    python scripts/validate_dataset.py --data dataset/remapped/data.yaml [--save-report]
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Targets from DATASET_STRATEGY.md Section 11
CLASS_TARGETS = {
    "helmet_on": 860,
    "no_helmet":  680,
    "vest_on":    770,
    "no_vest":    500,
    "person":     940,
    "mask_on":    120,
}

# no_vest should be within 20% of no_helmet
VIOLATION_BALANCE_TOLERANCE = 0.20


def count_annotations(labels_dir: Path, class_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if not labels_dir.exists():
        return counts
    for lbl_file in labels_dir.glob("*.txt"):
        for line in lbl_file.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id < len(class_names):
                counts[class_names[cls_id]] += 1
    return counts


def print_table(title: str, rows: list[tuple], headers: list[str]) -> str:
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    line = "-" * (sum(col_widths) + len(col_widths) * 3 + 1)
    lines = [f"\n{title}", line]
    header_row = " - ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    lines.append(f" {header_row} ")
    lines.append(line)
    for row in rows:
        data_row = " - ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers)))
        lines.append(f" {data_row} ")
    lines.append(line)
    output = "\n".join(lines)
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset class distribution and splits")
    parser.add_argument("--data", default="dataset/remapped/data.yaml", help="Path to data.yaml")
    parser.add_argument("--save-report", action="store_true", help="Save report to docs/dataset_stats.txt")
    args = parser.parse_args()

    with open(args.data) as f:
        config = yaml.safe_load(f)

    dataset_root = Path(config["path"])
    class_names: list[str] = config["names"]
    splits = ["train", "valid", "test"]

    report_lines: list[str] = []

    # -- Per-split image and annotation counts ---------------------------------
    split_rows = []
    all_annotations: dict[str, int] = defaultdict(int)

    for split in splits:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        n_images = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        counts = count_annotations(lbl_dir, class_names)
        n_annotations = sum(counts.values())
        for cls, cnt in counts.items():
            all_annotations[cls] += cnt
        split_rows.append((split, n_images, n_annotations))

    report_lines.append(print_table(
        "Split Distribution",
        split_rows,
        ["Split", "Images", "Annotations"]
    ))

    # -- Per-class annotation counts (all splits) ------------------------------
    total = sum(all_annotations.values())
    class_rows = []
    for cls in class_names:
        cnt = all_annotations.get(cls, 0)
        pct = f"{cnt / total * 100:.1f}%" if total > 0 else "0.0%"
        target = CLASS_TARGETS.get(cls, "—")
        delta = cnt - target if isinstance(target, int) else "—"
        class_rows.append((cls, cnt, pct, target, delta))

    report_lines.append(print_table(
        "Class Distribution (all splits)",
        class_rows,
        ["Class", "Annotations", "%", "Target", "Delta"]
    ))

    # -- Imbalance warnings ----------------------------------------------------
    warnings: list[str] = []
    no_helmet_cnt = all_annotations.get("no_helmet", 0)
    no_vest_cnt = all_annotations.get("no_vest", 0)

    if no_helmet_cnt > 0:
        ratio = abs(no_helmet_cnt - no_vest_cnt) / no_helmet_cnt
        if ratio > VIOLATION_BALANCE_TOLERANCE:
            warnings.append(
                f"  WARNING: no_vest ({no_vest_cnt}) vs no_helmet ({no_helmet_cnt}) "
                f"imbalance = {ratio:.0%} — target <= {VIOLATION_BALANCE_TOLERANCE:.0%}"
            )

    total_violations = no_helmet_cnt + no_vest_cnt
    violation_ratio = total_violations / total if total > 0 else 0
    if violation_ratio < 0.28:
        warnings.append(
            f"  WARNING: violation class ratio ({violation_ratio:.1%}) is below target 31%. "
            f"Consider adding more no_vest / no_helmet images."
        )

    if warnings:
        print("\nImbalance Warnings:")
        for w in warnings:
            print(w)
        report_lines.extend(["Imbalance Warnings:"] + warnings)
    else:
        print("\nAll class balance checks passed.")
        report_lines.append("All class balance checks passed.")

    print(f"\nTotal annotations: {total}")
    print(f"Violation ratio (no_helmet + no_vest): {violation_ratio:.1%}")

    if args.save_report:
        Path("docs").mkdir(exist_ok=True)
        Path("docs/dataset_stats.txt").write_text("\n".join(report_lines))
        logger.info("Saved report to docs/dataset_stats.txt")


if __name__ == "__main__":
    main()
