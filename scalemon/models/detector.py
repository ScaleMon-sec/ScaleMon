"""
Model-agnostic anomaly detection interface supporting multiple learning-based detectors.
"""

import torch
import torch.nn as nn
import numpy as np
import numpy as np
import joblib


class DeepSVDD3(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        h1 = in_dim // 2
        h2 = h1 // 2
        latent_dim = int(2 * np.sqrt(in_dim))

        self.net = nn.Sequential(
            nn.Linear(in_dim, h1, bias=False),
            nn.ReLU(),
            nn.Linear(h1, h2, bias=False),
            nn.ReLU(),
            nn.Linear(h2, latent_dim, bias=False)
        )

        self.register_buffer("c", torch.zeros(latent_dim))

    def set_c(self, c):
        self.c.copy_(c)

    def forward(self, x):
        z = self.net(x)

        if self.training:
            return z
        else:
            score = torch.sum((z - self.c) ** 2, dim=1)
            return score

class DeepSVDD4(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        h1 = in_dim // 2
        h2 = h1 // 2
        latent_dim = int(2 * np.sqrt(in_dim))

        self.net = nn.Sequential(
            nn.Linear(in_dim, h1, bias=False),
            nn.ReLU(),
            nn.Linear(h1, h1, bias=False),
            nn.ReLU(),
            nn.Linear(h1, h2, bias=False),
            nn.ReLU(),
            nn.Linear(h2, latent_dim, bias=False)
        )

        self.register_buffer("c", torch.zeros(latent_dim))

    def set_c(self, c):
        self.c.copy_(c)

    def forward(self, x):
        z = self.net(x)

        if self.training:
            return z
        else:
            score = torch.sum((z - self.c) ** 2, dim=1)
            return score

class AutoEncoder2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        h1 = in_dim // 2
        h2 = h1 // 2
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
        

class AutoEncoder3(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            h1 = in_dim // 2
            h2 = h1 // 2
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, h1), nn.ReLU(),
                nn.Linear(h1, h1), nn.ReLU(),
                nn.Linear(h1, h2), nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(h2, h1), nn.ReLU(),
                nn.Linear(h1, h1), nn.ReLU(),
                nn.Linear(h1, in_dim)
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

class SklearnWrapper:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = joblib.load(model_path)

    def _to_numpy(self, x: torch.Tensor):
        return x.detach().cpu().numpy()

    def _to_tensor(self, x: np.ndarray):
        return torch.from_numpy(x).float().to(self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class IsolationForestWrapper(SklearnWrapper):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np = self._to_numpy(x)
        scores = -self.model.score_samples(x_np)
        return self._to_tensor(scores)

class OneClassSVMWrapper(SklearnWrapper):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np = self._to_numpy(x)
        scores = -self.model.decision_function(x_np)
        return self._to_tensor(scores)

class LOFWrapper(SklearnWrapper):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np = self._to_numpy(x)
        scores = -self.model.decision_function(x_np)
        return self._to_tensor(scores)

class KNNWrapper(SklearnWrapper):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np = self._to_numpy(x)
        dist, _ = self.model.kneighbors(x_np)
        scores = dist.mean(axis=1)
        return self._to_tensor(scores)

class GMMWrapper(SklearnWrapper):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_np = self._to_numpy(x)
        scores = -self.model.score_samples(x_np)
        return self._to_tensor(scores)