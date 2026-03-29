"""
Offline augmentation pipeline — applied to the training split ONLY.
Each augmentation maps to a named real-world construction site condition.

Usage:
    python scripts/augment_dataset.py --data dataset/remapped/data.yaml --factor 2

--factor N produces N additional augmented copies per training image (doubles at 2).
Augmented images are written back into the train split alongside originals.
"""

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42

# Augmentation probabilities — each maps to a real-world condition
AUG_PROBS = {
    "shadow":     0.40,   # diagonal shadow overlay → scaffolding / overhead obstruction
    "fog":        0.20,   # haze simulation → overcast outdoor morning / dusty site
    "blur":       0.30,   # Gaussian blur → low-quality CCTV / cheap IP camera
    "brightness": 0.80,   # brightness/contrast shift → indoor fluorescent vs. direct sunlight
    "noise":      0.30,   # Gaussian noise → surveillance camera grain / low-light sensor
    "flip":       0.50,   # horizontal flip → mirrored site layouts / opposite-side cameras
}


def apply_shadow(img: np.ndarray) -> np.ndarray:
    """Diagonal shadow overlay simulating scaffolding / overhead obstruction."""
    h, w = img.shape[:2]
    shadow = img.copy()
    x1, x2 = sorted(random.sample(range(w), 2))
    y1, y2 = 0, h
    pts = np.array([[x1, y1], [x2, y2], [w, y2], [w, y1]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    alpha = random.uniform(0.35, 0.65)
    shadow[mask == 1] = (shadow[mask == 1] * (1 - alpha)).astype(np.uint8)
    return shadow


def apply_fog(img: np.ndarray) -> np.ndarray:
    """Fog / haze overlay simulating overcast outdoor or dusty construction site."""
    fog_level = random.uniform(0.25, 0.55)
    fog_layer = np.ones_like(img, dtype=np.float32) * 255
    return cv2.addWeighted(img.astype(np.float32), 1 - fog_level, fog_layer, fog_level, 0).astype(np.uint8)


def apply_blur(img: np.ndarray) -> np.ndarray:
    """Gaussian blur simulating low-quality CCTV or cheap IP camera lens."""
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_brightness(img: np.ndarray) -> np.ndarray:
    """Brightness / contrast shift simulating indoor vs. outdoor lighting."""
    alpha = random.uniform(0.55, 1.55)  # contrast
    beta = random.randint(-40, 60)      # brightness
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def apply_noise(img: np.ndarray) -> np.ndarray:
    """Gaussian noise simulating surveillance camera grain."""
    sigma = random.uniform(8, 28)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_flip(img: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Horizontal flip — also flips YOLO bounding box x-coordinates."""
    flipped_img = cv2.flip(img, 1)
    flipped_labels = []
    for line in labels:
        parts = line.strip().split()
        if not parts:
            flipped_labels.append(line)
            continue
        cls = parts[0]
        x_c = 1.0 - float(parts[1])
        rest = parts[2:]
        flipped_labels.append(f"{cls} {x_c:.6f} {' '.join(rest)}")
    return flipped_img, flipped_labels


def augment_image(img: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Apply a random combination of augmentations to one image."""
    aug_img = img.copy()
    aug_labels = labels[:]

    if random.random() < AUG_PROBS["flip"]:
        aug_img, aug_labels = apply_flip(aug_img, aug_labels)
    if random.random() < AUG_PROBS["brightness"]:
        aug_img = apply_brightness(aug_img)
    if random.random() < AUG_PROBS["shadow"]:
        aug_img = apply_shadow(aug_img)
    if random.random() < AUG_PROBS["fog"]:
        aug_img = apply_fog(aug_img)
    if random.random() < AUG_PROBS["blur"]:
        aug_img = apply_blur(aug_img)
    if random.random() < AUG_PROBS["noise"]:
        aug_img = apply_noise(aug_img)

    return aug_img, aug_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline augmentation for training split")
    parser.add_argument("--data", default="dataset/remapped/data.yaml", help="Path to data.yaml")
    parser.add_argument("--factor", type=int, default=2, help="Augmented copies per image (default 2)")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    with open(args.data) as f:
        config = yaml.safe_load(f)

    dataset_root = Path(config["path"])
    img_dir = dataset_root / "images" / "train"
    lbl_dir = dataset_root / "labels" / "train"

    image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    logger.info("Found %d training images. Generating ×%d augmented copies...", len(image_files), args.factor)

    generated = 0
    for img_path in tqdm(image_files, desc="Augmenting"):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read %s — skipping", img_path)
            continue

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        labels: list[str] = lbl_path.read_text().splitlines() if lbl_path.exists() else []

        for i in range(args.factor):
            aug_img, aug_labels = augment_image(img, labels)

            out_name = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(img_dir / f"{out_name}.jpg"), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (lbl_dir / f"{out_name}.txt").write_text("\n".join(aug_labels) + "\n")
            generated += 1

    logger.info("Generated %d augmented images. Training set now: %d images",
                generated, len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png"))))
    logger.info("Next: run scripts/validate_dataset.py --data %s", args.data)


if __name__ == "__main__":
    main()
