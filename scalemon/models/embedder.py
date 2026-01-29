""" 
Encodes input I/O image into compact one-dimensional representation
for downstream anomaly detection. 
"""


import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import timm

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, output_dim=512, use_pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        backbone = models.resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1]) 

        for p in self.backbone.parameters():
            p.requires_grad = False

        assert 512 % output_dim == 0, \
            f"output_dim must divide 512. Got {output_dim}."
        self.group_factor = 512 // output_dim

        self.output_dim = output_dim

    def forward(self, x):

        out = torch.zeros(x.size(0), 3, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        out[:, 1, :, :] = x[:, 0, :, :]
        out[:, 2, :, :] = x[:, 1, :, :]
        x = out
        
        feat = self.backbone(x)

        feat = feat.view(feat.size(0), 512)

        feat = feat.view(feat.size(0), self.output_dim, self.group_factor)

        feat = feat.mean(dim=-1)

        return feat

    def load_checkpoint(self, ckpt_path, device="cpu"):
        state = torch.load(ckpt_path, map_location=device)
        self.load_state_dict(state, strict=False)
        print(f"[+] Loaded checkpoint from {ckpt_path}")