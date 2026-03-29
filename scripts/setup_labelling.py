"""
Set up Label Studio for annotating custom scraped images.
Generates:
  - label_studio_data/interface.xml   — Label Studio XML labelling interface
  - label_studio_data/tasks.json      — task import file (one task per scraped image)

Usage:
    python scripts/setup_labelling.py [--images dataset/scraped/] [--host http://localhost:8080]

Label Studio setup (run once):
    pip install label-studio
    label-studio start --port 8080

Then import tasks.json via the Label Studio UI:
    Projects → Create → Import → upload label_studio_data/tasks.json
    Settings → Labelling Interface → paste content of label_studio_data/interface.xml
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_CLASSES = ["helmet_on", "no_helmet", "vest_on", "no_vest", "person", "mask_on"]

# Colour per class for Label Studio bounding box display
CLASS_COLOURS = {
    "helmet_on": "#00AA00",
    "no_helmet": "#FF0000",
    "vest_on":   "#00BB44",
    "no_vest":   "#FF6600",
    "person":    "#4488FF",
    "mask_on":   "#AA00AA",
}

ANNOTATION_POLICY_NOTE = """
ANNOTATION POLICY — read before labelling:

1. BOUNDING BOXES:
   - helmet_on / no_helmet: tight box around HEAD and helmet region
   - vest_on / no_vest: box from SHOULDERS to WAIST (torso region)
   - person: full body — use only when PPE state is unclear
   - mask_on: tight box around FACE / mask region

2. OCCLUSION (50% RULE):
   - Worker > 50% visible → annotate normally
   - Worker 20–50% visible → annotate ONLY if PPE state is clearly determinable
   - Worker < 20% visible → SKIP (do not annotate)

3. MULTIPLE PPE PER WORKER:
   - Draw independent boxes for head AND torso PPE
   - A worker with no helmet but wearing vest gets: 1x no_helmet + 1x vest_on

4. AMBIGUOUS STATE:
   - If PPE state cannot be determined (too small <15px, too blurry) → label as 'person' only
   - DO NOT GUESS

5. FAR-FIELD WORKERS:
   - If full-body bbox height would be < 40px → label as 'person' only (no PPE class)
"""


def build_xml_interface() -> str:
    """Build Label Studio XML labelling interface config."""
    label_items = "\n".join(
        f'      <Label value="{cls}" background="{CLASS_COLOURS[cls]}"/>'
        for cls in PROJECT_CLASSES
    )

    return f"""<View>
  <Header value="Construction Safety Monitor — PPE Annotation"/>
  <Text name="policy" value="{ANNOTATION_POLICY_NOTE.strip()}" style="white-space:pre-wrap;font-size:12px;background:#f5f5f5;padding:8px;margin-bottom:12px;"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image" showInline="true">
{label_items}
  </RectangleLabels>
</View>"""


def build_tasks(images_dir: Path, host: str) -> list[dict]:
    """Build one Label Studio task per image."""
    tasks = []
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted(
        f for f in images_dir.iterdir() if f.suffix.lower() in supported
    )

    for img_path in image_files:
        # Label Studio can serve local files if configured, or use file:// URI
        tasks.append({
            "data": {
                "image": f"{host}/data/local-files/?d={img_path.resolve().as_posix()}",
                "filename": img_path.name,
            }
        })

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Label Studio config for scraped images")
    parser.add_argument("--images", default="dataset/scraped", help="Directory of scraped images")
    parser.add_argument("--host", default="http://localhost:8080", help="Label Studio host URL")
    parser.add_argument("--output", default="label_studio_data", help="Output directory")
    args = parser.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}. Run scrape_images.py first.")

    # Write interface XML
    xml_path = out_dir / "interface.xml"
    xml_path.write_text(build_xml_interface())
    logger.info("Wrote interface: %s", xml_path)

    # Write tasks JSON
    tasks = build_tasks(images_dir, args.host)
    tasks_path = out_dir / "tasks.json"
    with open(tasks_path, "w") as f:
        json.dump(tasks, f, indent=2)
    logger.info("Wrote %d tasks: %s", len(tasks), tasks_path)

    print("\n── Label Studio Setup Instructions ────────────────────────────────")
    print("1. Start Label Studio (if not running):")
    print("     label-studio start --port 8080")
    print("\n2. Create a new project in the UI")
    print("\n3. Import tasks:")
    print(f"     UI → Import → upload {tasks_path}")
    print("\n4. Set labelling interface:")
    print(f"     Settings → Labelling Interface → Code → paste {xml_path}")
    print("\n5. Enable local file serving (one-time):")
    print("     label-studio start --port 8080 --data-dir ./label_studio_data")
    print("     Set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true in env")
    print("\n6. After labelling, export as YOLO format and save to:")
    print("     dataset/custom_labelled/")
    print("────────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
