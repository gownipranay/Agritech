"""MobileNetV3-Small classifier head for crop disease classification.

Chosen over EfficientNet-Lite for smaller size and faster CPU inference,
important for eventual on-device / low-end-phone deployment.
"""

import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model
