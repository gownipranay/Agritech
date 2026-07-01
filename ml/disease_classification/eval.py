"""Evaluate a trained checkpoint and report accuracy, per-class metrics, and
false negative rate (FNR) separately from overall accuracy.

FNR here = P(model predicts healthy | true label is a disease), i.e. how
often a real problem gets missed entirely. This is tracked separately from
accuracy because a missed disease is worse than a false alarm for a farmer —
a false alarm costs a wasted consultation, a false negative can cost a crop.

Usage:
    python eval.py --data ../data/sample_images/val --checkpoint ../checkpoints/disease_model.pt
"""

import argparse
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from classes import CLASSES, HEALTHY_CLASSES, IDX_TO_CLASS
from dataset import load_image_folder_dataset
from model import build_model


@torch.no_grad()
def run_eval(data_dir: str, checkpoint_path: str, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes=len(ckpt["classes"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = load_image_folder_dataset(data_dir, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    healthy_idx = {i for i, c in IDX_TO_CLASS.items() if c in HEALTHY_CLASSES}

    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    disease_total = 0
    disease_missed_as_healthy = 0
    correct_total, n_total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)

        for true_idx, pred_idx in zip(labels.tolist(), preds.tolist()):
            per_class_total[true_idx] += 1
            n_total += 1
            if true_idx == pred_idx:
                per_class_correct[true_idx] += 1
                correct_total += 1
            if true_idx not in healthy_idx:
                disease_total += 1
                if pred_idx in healthy_idx:
                    disease_missed_as_healthy += 1

    accuracy = correct_total / n_total if n_total else 0.0
    false_negative_rate = disease_missed_as_healthy / disease_total if disease_total else 0.0

    report = {
        "overall_accuracy": round(accuracy, 4),
        "false_negative_rate": round(false_negative_rate, 4),
        "false_negative_rate_definition": (
            "Fraction of true-disease samples predicted as a healthy class "
            "(a missed diagnosis, tracked separately from accuracy)."
        ),
        "n_samples": n_total,
        "per_class": {
            IDX_TO_CLASS[idx]: {
                "n": per_class_total[idx],
                "accuracy": round(per_class_correct[idx] / per_class_total[idx], 4)
                if per_class_total[idx]
                else None,
            }
            for idx in sorted(per_class_total)
        },
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/sample_images/val")
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/disease_model.pt")
    args = parser.parse_args()

    report = run_eval(args.data, args.checkpoint)
    print(json.dumps(report, indent=2))
