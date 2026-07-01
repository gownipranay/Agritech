"""Single-image inference with a confidence threshold.

Below CONFIDENCE_THRESHOLD, the caller should show "uncertain — please
consult local expert" instead of forcing a guess (see backend router).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import EVAL_TRANSFORM
from model import build_model

CONFIDENCE_THRESHOLD = 0.60


class DiseaseClassifier:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.classes = ckpt["classes"]
        self.model = build_model(num_classes=len(self.classes), pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        image = image.convert("RGB")
        tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        top_prob, top_idx = torch.max(probs, dim=0)
        confidence = top_prob.item()
        predicted_class = self.classes[top_idx.item()]

        is_confident = confidence >= CONFIDENCE_THRESHOLD

        return {
            "predicted_class": predicted_class if is_confident else None,
            "confidence": round(confidence, 4),
            "is_confident": is_confident,
            "message": (
                None
                if is_confident
                else "uncertain — please consult local expert (Krishi Vigyan Kendra or agri-input dealer)"
            ),
            "top_5": [
                {"class": self.classes[i], "confidence": round(p.item(), 4)}
                for p, i in zip(*torch.topk(probs, k=min(5, len(self.classes))))
            ],
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="../checkpoints/disease_model.pt")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    clf = DiseaseClassifier(args.checkpoint)
    result = clf.predict(Image.open(Path(args.image)))
    print(result)
