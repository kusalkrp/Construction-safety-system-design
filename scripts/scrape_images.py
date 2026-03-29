"""
Scrape construction safety images from DuckDuckGo Image Search.
Each collected image is logged to dataset/scraped_manifest.csv with
source URL, query, class hint, and MD5 hash for deduplication.

Usage:
    python scripts/scrape_images.py [--output dataset/scraped/] [--per-query 25]
"""

import argparse
import csv
import hashlib
import logging
import random
import time
from pathlib import Path

import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_HEADER = ["filename", "url", "query", "class_hint", "md5", "width", "height", "kept"]

# Targeted queries per gap (see DATASET_STRATEGY.md Section 5)
SCRAPE_QUERIES: list[dict] = [
    # ── Gap 2: no_vest imbalance ──────────────────────────────────────────────
    {"query": "construction worker without safety vest site", "class_hint": "no_vest"},
    {"query": "builder missing hi-vis jacket construction", "class_hint": "no_vest"},
    {"query": "construction worker no high visibility vest outdoor", "class_hint": "no_vest"},
    {"query": "site worker without reflective vest violation", "class_hint": "no_vest"},

    # ── Gap 1 & 4: no_helmet + elevation scenes ───────────────────────────────
    {"query": "construction worker without helmet violation site", "class_hint": "no_helmet"},
    {"query": "builder missing hard hat scaffolding", "class_hint": "no_helmet"},
    {"query": "construction worker no hard hat elevated platform", "class_hint": "no_helmet"},
    {"query": "worker without helmet scaffolding height", "class_hint": "no_helmet"},

    # ── Gap 1: indoor scenes for helmet_on ────────────────────────────────────
    {"query": "construction worker wearing hard hat indoor warehouse", "class_hint": "helmet_on"},
    {"query": "builder helmet PPE indoor construction floor", "class_hint": "helmet_on"},
    {"query": "worker hard hat scaffold elevated site", "class_hint": "helmet_on"},

    # ── Gap 1: indoor scenes for vest_on ─────────────────────────────────────
    {"query": "construction worker high visibility vest warehouse indoor", "class_hint": "vest_on"},
    {"query": "builder hi-vis vest indoor construction site", "class_hint": "vest_on"},
    {"query": "worker reflective vest covered construction floor", "class_hint": "vest_on"},

    # ── Gap 4: mixed/crowd scaffolding scenes ────────────────────────────────
    {"query": "construction site workers crowd safety PPE", "class_hint": "mixed_scene"},
    {"query": "scaffolding workers varying PPE compliance", "class_hint": "mixed_scene"},
    {"query": "construction site multiple workers scaffolding outdoor", "class_hint": "mixed_scene"},
]

MIN_DIMENSION_PX = 100
MAX_FILE_SIZE_MB = 5


def compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def is_valid_image(data: bytes) -> tuple[bool, int, int]:
    """Check image is valid, not too small. Returns (valid, width, height)."""
    try:
        img = Image.open(BytesIO(data))
        w, h = img.size
        if w < MIN_DIMENSION_PX or h < MIN_DIMENSION_PX:
            return False, w, h
        return True, w, h
    except Exception:
        return False, 0, 0


DDG_RETRY_ATTEMPTS = 4
DDG_RETRY_BASE_DELAY = 10  # seconds — doubles each attempt


def _fetch_ddg_results(query: str, max_results: int) -> list[dict]:
    """
    Fetch DuckDuckGo image results with exponential backoff on rate limit.
    Returns empty list if all retries exhausted.
    """
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import RatelimitException

    for attempt in range(DDG_RETRY_ATTEMPTS):
        try:
            with DDGS() as ddgs:
                return list(ddgs.images(query, max_results=max_results))
        except RatelimitException:
            wait = DDG_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(1, 4)
            if attempt < DDG_RETRY_ATTEMPTS - 1:
                logger.warning(
                    "DuckDuckGo rate limit hit (attempt %d/%d). Waiting %.0fs...",
                    attempt + 1, DDG_RETRY_ATTEMPTS, wait,
                )
                time.sleep(wait)
            else:
                logger.error("DuckDuckGo rate limit — all retries exhausted for query: '%s'", query)
        except Exception as exc:
            logger.warning("DDG search error for '%s': %s", query, exc)
            break
    return []


def scrape_query(query: str, class_hint: str, out_dir: Path, max_results: int,
                 seen_hashes: set[str], manifest_writer: "csv.DictWriter") -> int:
    """Scrape one query. Returns number of images saved."""
    results = _fetch_ddg_results(query, max_results=max_results * 3)
    if not results:
        return 0

    saved = 0
    for result in results:
        if saved >= max_results:
            break

        url = result.get("image", "")
        if not url:
            continue

        try:
            resp = requests.get(url, timeout=8, stream=True)
            if resp.status_code != 200:
                continue
            if int(resp.headers.get("content-length", 0)) > MAX_FILE_SIZE_MB * 1024 * 1024:
                continue

            data = resp.content
            md5 = compute_md5(data)
            if md5 in seen_hashes:
                continue

            valid, w, h = is_valid_image(data)
            if not valid:
                continue

            content_type = resp.headers.get("content-type", "image/jpeg")
            ext = ".jpg" if "jpeg" in content_type else ".png" if "png" in content_type else ".jpg"

            filename = f"{class_hint}_{md5[:8]}{ext}"
            (out_dir / filename).write_bytes(data)

            seen_hashes.add(md5)
            manifest_writer.writerow({
                "filename": filename,
                "url": url,
                "query": query,
                "class_hint": class_hint,
                "md5": md5,
                "width": w,
                "height": h,
                "kept": "yes",
            })
            saved += 1
            time.sleep(0.5)

        except Exception as exc:
            logger.debug("Failed %s: %s", url, exc)
            continue

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape construction safety images")
    parser.add_argument("--output", default="dataset/scraped", help="Output directory")
    parser.add_argument("--per-query", type=int, default=25, help="Max images per query")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("dataset/scraped_manifest.csv")
    seen_hashes: set[str] = set()

    # Load existing manifest to avoid re-downloading
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                seen_hashes.add(row["md5"])
        logger.info("Loaded %d existing hashes from manifest", len(seen_hashes))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if manifest_path.exists() else "w"

    with open(manifest_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        if mode == "w":
            writer.writeheader()

        total = 0
        for i, entry in enumerate(tqdm(SCRAPE_QUERIES, desc="Queries")):
            n = scrape_query(
                query=entry["query"],
                class_hint=entry["class_hint"],
                out_dir=out_dir,
                max_results=args.per_query,
                seen_hashes=seen_hashes,
                manifest_writer=writer,
            )
            logger.info("  '%s' → %d images", entry["query"][:50], n)
            total += n
            f.flush()
            # Pause between queries to avoid rate limiting
            if i < len(SCRAPE_QUERIES) - 1:
                pause = random.uniform(4, 8)
                time.sleep(pause)

    logger.info("Scraped %d images total. Manifest: %s", total, manifest_path)
    logger.info("Next: review images in %s, then run scripts/setup_labelling.py", out_dir)


if __name__ == "__main__":
    main()
