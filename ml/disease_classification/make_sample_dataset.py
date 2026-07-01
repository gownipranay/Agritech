"""Generate a tiny synthetic image dataset to smoke-test the training pipeline.

IMPORTANT: These are procedurally-drawn placeholder images (colored leaf
shapes with synthetic lesion blobs), NOT real plant photos. They exist only
to prove dataset.py / model.py / train.py / eval.py run correctly end to end
before real data is plugged in. Do not use metrics from this run as real
model performance — see model_card.md and data/sources.md.

Usage:
    python make_sample_dataset.py --out ../data/sample_images --per-class 40
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from classes import CLASSES, HEALTHY_CLASSES

CROP_BASE_COLOR = {
    "tomato": (46, 125, 50),
    "potato": (56, 142, 60),
    "corn": (85, 139, 47),
}

LESION_COLOR = {
    "early_blight": (139, 90, 43),
    "late_blight": (74, 60, 40),
    "leaf_mold": (196, 178, 94),
    "common_rust": (170, 74, 44),
}


def draw_leaf(size=224):
    img = Image.new("RGB", (size, size), (235, 235, 225))
    draw = ImageDraw.Draw(img)
    return img, draw


def synthesize_image(class_name: str, size: int = 224, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    crop, condition = class_name.split("_", 1)
    base_color = CROP_BASE_COLOR[crop]
    jitter = tuple(max(0, min(255, c + rng.randint(-15, 15))) for c in base_color)

    img, draw = draw_leaf(size)
    margin = size // 8
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=jitter,
        outline=(20, 60, 20),
    )

    if class_name not in HEALTHY_CLASSES:
        lesion_color = LESION_COLOR.get(condition, (120, 80, 40))
        n_spots = rng.randint(6, 14)
        for _ in range(n_spots):
            cx = rng.randint(margin, size - margin)
            cy = rng.randint(margin, size - margin)
            r = rng.randint(4, 14)
            spot_color = tuple(max(0, min(255, c + rng.randint(-10, 10))) for c in lesion_color)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=spot_color)

    # Simulate real phone-photo noise/blur variability.
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="../data/sample_images")
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    args = parser.parse_args()

    out_root = Path(args.out)
    for split in ("train", "val"):
        for class_name in CLASSES:
            (out_root / split / class_name).mkdir(parents=True, exist_ok=True)

    n_val = max(1, int(args.per_class * args.val_fraction))
    n_train = args.per_class - n_val

    for class_name in CLASSES:
        for i in range(n_train):
            img = synthesize_image(class_name, seed=hash((class_name, "train", i)) & 0xFFFF)
            img.save(out_root / "train" / class_name / f"{i:04d}.jpg", quality=85)
        for i in range(n_val):
            img = synthesize_image(class_name, seed=hash((class_name, "val", i)) & 0xFFFF)
            img.save(out_root / "val" / class_name / f"{i:04d}.jpg", quality=85)

    print(f"Wrote {n_train} train + {n_val} val images per class to {out_root.resolve()}")
    print("Reminder: this is synthetic smoke-test data, not real plant photos.")


if __name__ == "__main__":
    main()
