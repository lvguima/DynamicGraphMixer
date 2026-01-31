import torch
import torch.nn as nn


class ResidualGCN(nn.Module):
    def __init__(self, d_model, num_vars, layers=1, norm="sym", dropout=0.0, g_init=0.0, g_mode="scalar"):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(max(1, int(layers)))])
        self.norm = str(norm).lower()
        if self.norm not in ("sym", "row"):
            raise ValueError("gcn_norm must be 'sym' or 'row'.")
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        g_mode = str(g_mode).lower()
        if g_mode == "scalar":
            self.g_scale = nn.Parameter(torch.tensor(float(g_init)).view(1, 1, 1, 1))
        elif g_mode == "per_var":
            self.g_scale = nn.Parameter(torch.full((1, int(num_vars), 1, 1), float(g_init)))
        else:
            raise ValueError("gcn_g_mode must be 'scalar' or 'per_var'.")

    def _normalize_adj(self, adj):
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        bsz, n_vars, _ = adj.shape
        eye = torch.eye(n_vars, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj_hat = adj + eye
        if self.norm == "sym":
            deg = adj_hat.sum(-1).clamp_min(1e-12)
            deg_inv_sqrt = deg.pow(-0.5)
            return deg_inv_sqrt.unsqueeze(-1) * adj_hat * deg_inv_sqrt.unsqueeze(-2)
        return adj_hat / adj_hat.sum(-1, keepdim=True).clamp_min(1e-12)

    def forward(self, x, adj):
        # x: [B, C, T, D]
        if adj.device != x.device or adj.dtype != x.dtype:
            adj = adj.to(device=x.device, dtype=x.dtype)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.shape[0], -1, -1)
        adj_norm = self._normalize_adj(adj)

        out = x
        for layer in self.layers:
            out = torch.einsum("bij,bjtd->bitd", adj_norm, out)
            out = layer(out)
            out = self.activation(out)
        out = self.dropout(out)
        return x + out * self.g_scale
