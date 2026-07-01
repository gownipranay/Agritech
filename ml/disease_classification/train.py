"""Train the disease classifier.

Usage:
    python train.py --data ../data/sample_images --epochs 5 --out ../checkpoints/disease_model.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from classes import CLASSES
from dataset import load_image_folder_dataset
from model import build_model


def train(data_dir: str, epochs: int, batch_size: int, lr: float, out_path: str, pretrained: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = load_image_folder_dataset(f"{data_dir}/train", train=True)
    val_ds = load_image_folder_dataset(f"{data_dir}/val", train=False)

    assert train_ds.classes == CLASSES, (
        f"Dataset class order {train_ds.classes} does not match classes.py {CLASSES}. "
        "ImageFolder sorts alphabetically — keep classes.py in the same alphabetical order."
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=len(CLASSES), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_acc = 0.0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_acc = evaluate_accuracy(model, val_loader, device)
        print(
            f"epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "class_to_idx": train_ds.class_to_idx,
                    "val_acc": val_acc,
                },
                out_path,
            )

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps({"classes": CLASSES, "best_val_acc": best_val_acc}, indent=2))
    print(f"Saved best checkpoint (val_acc={best_val_acc:.3f}) to {out_path}")


@torch.no_grad()
def evaluate_accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/sample_images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="../checkpoints/disease_model.pt")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch instead of ImageNet-pretrained weights (faster on tiny smoke-test data, no download).",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_path=args.out,
        pretrained=not args.no_pretrained,
    )
