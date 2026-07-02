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
    def predict(self, image: Image.Image, allowed_prefixes: list[str] | None = None) -> dict:
        """Classify a leaf image.

        allowed_prefixes: if given (e.g. ["Tomato"] or ["corn"]), the model is
        restricted to classes whose name starts with one of these prefixes and
        the probabilities are renormalized within that set. This is used to
        scope a prediction to the crop the user selected, so a tomato photo can
        never be labelled "cotton" — a real problem when users upload
        out-of-distribution field photos to a lab-trained model.
        """
        image = image.convert("RGB")
        tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        if allowed_prefixes:
            allowed_idx = [
                i for i, c in enumerate(self.classes)
                if any(c.startswith(p) for p in allowed_prefixes)
            ]
        else:
            allowed_idx = list(range(len(self.classes)))

        # Renormalize over just the allowed classes so confidence is meaningful
        # relative to the selected crop.
        allowed_probs = probs[allowed_idx]
        allowed_probs = allowed_probs / allowed_probs.sum()

        best_local = int(torch.argmax(allowed_probs).item())
        top_idx = allowed_idx[best_local]
        confidence = allowed_probs[best_local].item()
        predicted_class = self.classes[top_idx]

        is_confident = confidence >= CONFIDENCE_THRESHOLD

        k = min(5, len(allowed_idx))
        top_probs, top_local = torch.topk(allowed_probs, k=k)
        top_5 = [
            {"class": self.classes[allowed_idx[li]], "confidence": round(p.item(), 4)}
            for p, li in zip(top_probs, top_local)
        ]

        return {
            "predicted_class": predicted_class if is_confident else None,
            "confidence": round(confidence, 4),
            "is_confident": is_confident,
            "message": (
                None
                if is_confident
                else "uncertain — please consult local expert (Krishi Vigyan Kendra or agri-input dealer)"
            ),
            "top_5": top_5,
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
