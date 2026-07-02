"""Restructure the raw Kaggle PlantVillage download into a train/val
ImageFolder layout, with a per-class cap so CPU training stays tractable.

The real dataset is heavily imbalanced (152 to 3209 images/class). Rather
than hiding that, we keep the natural imbalance below the cap and record
per-class counts in the eval report / model card, since a farmer-facing
system deployed on rare classes (few real photos available) needs that
visibility.

Usage:
    python prepare_plantvillage.py --raw ../../data/plantvillage_raw/PlantVillage/PlantVillage \
        --out ../../data/plantvillage --train-cap 250 --val-cap 50
"""

import argparse
import random
import shutil
from pathlib import Path

from classes import CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--train-cap", type=int, default=250)
    parser.add_argument("--val-cap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    raw_root = Path(args.raw)
    out_root = Path(args.out)

    counts = {}
    for class_name in CLASSES:
        src_dir = raw_root / class_name
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {src_dir}")

        images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(images)

        n_val = min(args.val_cap, max(1, int(len(images) * 0.2)))
        n_train = min(args.train_cap, len(images) - n_val)

        val_images = images[:n_val]
        train_images = images[n_val : n_val + n_train]

        for split, split_images in (("train", train_images), ("val", val_images)):
            dest_dir = out_root / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, dest_dir / img_path.name)

        counts[class_name] = {"available": len(images), "train": n_train, "val": n_val}
        print(f"{class_name}: available={len(images)} train={n_train} val={n_val}")

    print(f"\nWrote dataset to {out_root.resolve()}")


if __name__ == "__main__":
    main()
