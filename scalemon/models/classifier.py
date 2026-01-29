"""
Implements application or behavior classification models
used within the ScaleMon pipeline.
"""

import torch
import torch.nn as nn
from torchvision import models

class ResNet18Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 2,
        pth_path: str | None = None,
    ):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            num_classes
        )

        if pth_path is not None:
            state_dict = torch.load(pth_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        return self.backbone(x)
