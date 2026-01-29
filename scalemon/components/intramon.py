"""
Implements intra-file monitoring.
Detects fine-grained anomalies within individual files based on access behavior patterns.
"""

import torch
import numpy as np
import torch.nn.functional as F
from scalemon.components.intra_fg import Intra_FG

class IntraMon:
    def __init__(
        self,
        classifier,
        conf_threshold,
        embedder,
        detector,
        threshold,
        device="cuda",
        alpha=0.95,
    ):

        self.intra_fg = Intra_FG()

        self.classifier = classifier.to(device).eval()
        self.embedder = embedder.to(device).eval()

        self.detector = {
            k: v.to(device).eval() for k, v in detector.items()
        }

        self.conf_threshold = conf_threshold
        self.threshold = threshold
        self.device = device
        self.alpha = alpha

    @torch.no_grad()
    def phase1(self, app_id, io_images):
        x = torch.from_numpy(np.stack(io_images)).float().to(self.device)

        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)

        app_probs = probs[:, int(app_id) - 1]
        return app_probs < self.conf_threshold  

    @torch.no_grad()
    def phase2(self, app_id, io_images):

        x = torch.from_numpy(np.stack(io_images)).float().to(self.device)

        features = self.embedder(x)
        scores = self.detector[app_id](features)

        return scores > self.threshold[app_id][self.alpha] 

    def __call__(self, app_id, df_intra):
        io_images, intra_files = self.intra_fg(df_intra)
        x = torch.from_numpy(
            np.stack(io_images)
        ).float().to(self.device)
        
        phase1_mask = self.phase1(app_id, io_images)

        pass_idx = (~phase1_mask).nonzero(as_tuple=True)[0]

        final_mask = phase1_mask.clone()

        if pass_idx.numel() > 0:
            filtered_images = [io_images[i] for i in pass_idx.tolist()]
            phase2_mask = self.phase2(app_id, filtered_images)
            final_mask[pass_idx] = phase2_mask

        anomaly_indices = final_mask.nonzero(as_tuple=True)[0].tolist()
        intra_anomaly_files = [intra_files[idx] for idx in anomaly_indices]

        return intra_anomaly_files