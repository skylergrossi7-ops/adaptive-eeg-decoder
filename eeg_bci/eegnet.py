"""EEGNet-8,2 for 250 Hz Graz hand imagery, after the shared offline bandpass."""

import torch
from torch import nn


class EEGNet(nn.Module):
    """Accept (trials, 22 electrodes, 750 samples) in microvolts. Return two logits."""

    def __init__(self):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 125), padding=(0, 62), bias=False),
            nn.BatchNorm2d(8),
        )
        # Each temporal filter learns two independent mixtures of the 22 electrodes.
        self.spatial = nn.Conv2d(8, 16, (22, 1), groups=8, bias=False)
        self.pool_spatial = nn.Sequential(
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.5)
        )
        self.separable = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, (1, 16), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(16 * (750 // 4 // 8), 2)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (22, 750):
            raise ValueError("Expected (trials, 22 electrodes, 750 samples).")
        temporal = self.temporal(x.unsqueeze(1))
        spatial = self.pool_spatial(self.spatial(temporal))
        return self.classifier(self.separable(spatial).flatten(1))

    @torch.no_grad()
    def constrain_weights(self):
        """Project each spatial filter and classifier row after the optimizer step."""
        for weight, limit in ((self.spatial.weight, 1.0), (self.classifier.weight, 0.25)):
            rows = weight.flatten(1)
            norms = rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
            rows.mul_((limit / norms).clamp_max(1.0))
