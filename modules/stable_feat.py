import torch
import torch.nn as nn
import torch.nn.functional as F


class StableFeature(nn.Module):
    def __init__(self, feat_type="none", window=3):
        super().__init__()
        self.feat_type = str(feat_type).lower()
        self.window = max(1, int(window))

    def _moving_average(self, x):
        # x: [B, L, C]
        if self.window <= 1:
            return x
        k = self.window
        left = k // 2
        right = k // 2
        if k % 2 == 0:
            left = max(0, left - 1)
        x = x.permute(0, 2, 1)
        x = F.pad(x, (left, right), mode="replicate")
        x = F.avg_pool1d(x, kernel_size=k, stride=1, padding=0)
        return x.permute(0, 2, 1)

    def forward(self, x):
        # x: [B, L, C]
        if self.feat_type == "none":
            return x
        if self.feat_type == "detrend":
            trend = self._moving_average(x)
            return x - trend
        if self.feat_type == "diff":
            diff = x[:, 1:, :] - x[:, :-1, :]
            pad = torch.zeros_like(x[:, :1, :])
            return torch.cat([pad, diff], dim=1)
        raise ValueError(f"Unsupported stable_feat_type: {self.feat_type}")
