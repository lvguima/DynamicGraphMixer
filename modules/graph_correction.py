import torch
import torch.nn as nn

from .gcn import ResidualGCN
from .head import ForecastHead


class GraphCorrectionHead(nn.Module):
    def __init__(
        self,
        d_model,
        num_vars,
        num_tokens,
        pred_len,
        gcn_layers=1,
        gcn_norm="sym",
        gcn_g_mode="scalar",
        dropout=0.0,
    ):
        super().__init__()
        self.gcn = ResidualGCN(
            d_model=d_model,
            num_vars=num_vars,
            layers=gcn_layers,
            norm=gcn_norm,
            dropout=dropout,
            g_init=0.0,
            g_mode=gcn_g_mode,
        )
        self.head = ForecastHead(
            d_model=d_model,
            num_tokens=num_tokens,
            pred_len=pred_len,
        )

    def forward(self, h_mix, adj):
        # h_mix: [B, C, T, D], adj: [B, C, C] or [C, C]
        h_corr = self.gcn(h_mix, adj)
        return self.head(h_corr)
