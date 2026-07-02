"""Merge the maize / rice / cotton Kaggle datasets into the existing
data/plantvillage train/val ImageFolder, under canonical class-folder names
that double as treatment-lookup keys (see data/knowledge_base/treatments.json).

Each source dataset has its own folder layout and its own disease vocabulary,
so we map explicitly rather than guessing. Groundnut is intentionally excluded:
its Kaggle dataset ships as object-detection (JPG + Pascal-VOC XML) with no
per-disease class folders, so it stays advisory-only.

Usage:
    python prepare_extra_crops.py --raw ../../data/crop_raw --out ../../data/plantvillage \
        --train-cap 250 --val-cap 50
"""

import argparse
import random
import shutil
from pathlib import Path

# source-folder (relative to --raw) -> canonical class name
MAPPING = {
    "maize/data/Blight": "corn_northern_leaf_blight",
    "maize/data/Common_Rust": "corn_common_rust",
    "maize/data/Gray_Leaf_Spot": "corn_gray_leaf_spot",
    "maize/data/Healthy": "corn_healthy",
    "rice/Bacterialblight": "paddy_bacterial_leaf_blight",
    "rice/Blast": "paddy_blast",
    "rice/Brownspot": "paddy_brown_spot",
    "rice/Tungro": "paddy_tungro",
    "cotton/cotton/bacterial_blight": "cotton_bacterial_blight",
    "cotton/cotton/curl_virus": "cotton_leaf_curl_virus",
    "cotton/cotton/fussarium_wilt": "cotton_fusarium_wilt",
    "cotton/cotton/healthy": "cotton_healthy",
}

# Chilli is a Roboflow export: images sit flat in train/valid/test with the
# class encoded as the filename prefix (e.g. "Curly-1-_jpg.rf.<hash>.jpg"),
# not in class subfolders. Map prefix -> canonical class. Only leaf curl and
# healthy are present in this dataset.
CHILLI_DIR = "chilli"
CHILLI_PREFIX_MAPPING = {
    "Curly": "chilli_leaf_curl_virus",
    "Healthy": "chilli_healthy",
}

IMG_SUFFIXES = (".jpg", ".jpeg", ".png")


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

    for src_rel, class_name in MAPPING.items():
        src_dir = raw_root / src_rel
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Expected source folder not found: {src_dir}")

        images = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)
        rng.shuffle(images)

        n_val = min(args.val_cap, max(1, int(len(images) * 0.2)))
        n_train = min(args.train_cap, len(images) - n_val)

        val_images = images[:n_val]
        train_images = images[n_val : n_val + n_train]

        for split, split_images in (("train", train_images), ("val", val_images)):
            dest_dir = out_root / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for i, img_path in enumerate(split_images):
                # rename to avoid collisions across source subfolders
                dest = dest_dir / f"{class_name}_{i:04d}{img_path.suffix.lower()}"
                shutil.copy2(img_path, dest)

        print(f"{class_name}: available={len(images)} train={n_train} val={n_val}")

    # Chilli: gather flat images from all splits, group by filename prefix.
    chilli_root = raw_root / CHILLI_DIR
    if chilli_root.is_dir():
        by_class = {v: [] for v in CHILLI_PREFIX_MAPPING.values()}
        for img in chilli_root.rglob("*"):
            if img.suffix.lower() not in IMG_SUFFIXES:
                continue
            prefix = img.name.split("-")[0].split("_")[0]
            cls = CHILLI_PREFIX_MAPPING.get(prefix)
            if cls:
                by_class[cls].append(img)

        for class_name, images in by_class.items():
            rng.shuffle(images)
            n_val = min(args.val_cap, max(1, int(len(images) * 0.2)))
            n_train = min(args.train_cap, len(images) - n_val)
            val_images = images[:n_val]
            train_images = images[n_val : n_val + n_train]
            for split, split_images in (("train", train_images), ("val", val_images)):
                dest_dir = out_root / split / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for i, img_path in enumerate(split_images):
                    dest = dest_dir / f"{class_name}_{i:04d}{img_path.suffix.lower()}"
                    shutil.copy2(img_path, dest)
            print(f"{class_name}: available={len(images)} train={n_train} val={n_val}")

    print(f"\nMerged extra crops into {out_root.resolve()}")


if __name__ == "__main__":
    main()
