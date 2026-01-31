import math
import torch
import torch.nn as nn

from .mixer import GraphMixer


class _SegmentGATLayer(nn.Module):
    def __init__(self, d_model, heads, topk, bias_base, dropout):
        super().__init__()
        self.heads = int(heads)
        self.topk = int(topk)
        self.bias_base = float(bias_base)
        if d_model % self.heads != 0:
            raise ValueError("d_model must be divisible by gat_heads.")
        self.head_dim = d_model // self.heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _apply_topk(self, scores):
        if self.topk <= 0 or self.topk >= scores.shape[-1]:
            return scores
        k = min(self.topk, scores.shape[-1])
        idx = torch.topk(scores, k, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        return scores.masked_fill(~mask, float("-inf"))

    def forward(self, h_val, z_map, base_adj=None, residual_scale=None):
        # h_val: [B, C, T, D], z_map: [B, C, D]
        bsz, n_vars, _, d_model = h_val.shape
        q = self.q_proj(z_map).view(bsz, n_vars, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(z_map).view(bsz, n_vars, self.heads, self.head_dim).permute(0, 2, 1, 3)
        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if base_adj is not None and self.bias_base != 0.0:
            bias = torch.log(base_adj.clamp_min(1e-12))
            scores = scores + self.bias_base * bias.unsqueeze(1)
        raw_attn = torch.softmax(scores, dim=-1)
        scores = self._apply_topk(scores)
        attn = torch.softmax(scores, dim=-1)

        v = (
            self.v_proj(h_val)
            .contiguous()
            .view(bsz, n_vars, -1, self.heads, self.head_dim)
            .permute(0, 3, 1, 2, 4)
        )
        mixed = torch.einsum("bhij,bhjtd->bhitd", attn, v)
        mixed = mixed.permute(0, 2, 3, 1, 4).contiguous().view(bsz, n_vars, -1, d_model)
        mixed = self.out_proj(mixed)
        if residual_scale is not None:
            mixed = mixed * residual_scale
        mixed = self.dropout(mixed)
        out = h_val + mixed
        return out, attn.mean(dim=1), raw_attn.mean(dim=1)


class _TokenAttnLayer(nn.Module):
    def __init__(self, d_model, heads, topk, dropout):
        super().__init__()
        self.heads = int(heads)
        self.topk = int(topk)
        if d_model % self.heads != 0:
            raise ValueError("d_model must be divisible by attn_heads.")
        self.head_dim = d_model // self.heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _apply_topk(self, scores):
        if self.topk <= 0 or self.topk >= scores.shape[-1]:
            return scores
        k = min(self.topk, scores.shape[-1])
        idx = torch.topk(scores, k, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        return scores.masked_fill(~mask, float("-inf"))

    def forward(self, h_val, h_map=None, residual_scale=None):
        # h_val: [B, C, T, D]
        if h_map is None:
            h_map = h_val
        bsz, n_vars, num_tokens, d_model = h_val.shape
        q = self.q_proj(h_map).permute(0, 2, 1, 3).contiguous().view(bsz * num_tokens, n_vars, d_model)
        k = self.k_proj(h_map).permute(0, 2, 1, 3).contiguous().view(bsz * num_tokens, n_vars, d_model)
        v = self.v_proj(h_val).permute(0, 2, 1, 3).contiguous().view(bsz * num_tokens, n_vars, d_model)

        q = q.view(bsz * num_tokens, n_vars, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bsz * num_tokens, n_vars, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(bsz * num_tokens, n_vars, self.heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        scores = self._apply_topk(scores)
        attn = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("bhij,bhjd->bhid", attn, v)
        mixed = mixed.permute(0, 2, 1, 3).contiguous().view(bsz * num_tokens, n_vars, d_model)
        mixed = self.out_proj(mixed)
        mixed = mixed.view(bsz, num_tokens, n_vars, d_model).permute(0, 2, 1, 3)
        if residual_scale is not None:
            mixed = mixed * residual_scale
        mixed = self.dropout(mixed)
        out = h_val + mixed
        return out


class GraphMixerV7(nn.Module):
    def __init__(self, configs, num_vars, num_tokens):
        super().__init__()
        self.graph_mixer_type = str(getattr(configs, "graph_mixer_type", "baseline")).lower()
        if self.graph_mixer_type not in ("baseline", "gat_seg", "attn_token", "gcn_norm"):
            raise ValueError(f"Unsupported graph_mixer_type: {self.graph_mixer_type}")

        dropout = float(getattr(configs, "dropout", 0.0))
        self.baseline = GraphMixer(
            dropout=dropout,
            gate_mode=getattr(configs, "gate_mode", "none"),
            num_vars=num_vars,
            num_tokens=num_tokens,
            gate_init=float(getattr(configs, "gate_init", 0.0)),
        )
        self.gate = self.baseline.gate if self.graph_mixer_type == "baseline" else None

        self.residual_scale = nn.Parameter(
            torch.full((1, int(num_vars), 1, 1), float(getattr(configs, "residual_scale_init", 0.0)))
        )
        self.warmup_epochs = int(getattr(configs, "warmup_epochs", 0))
        self.current_epoch = 0

        d_model = int(getattr(configs, "d_model", 128))

        self.gat_layers = None
        if self.graph_mixer_type == "gat_seg":
            gat_heads = int(getattr(configs, "gat_heads", 4))
            gat_topk = int(getattr(configs, "gat_topk", 6))
            gat_bias_base = float(getattr(configs, "gat_bias_base", 0.0))
            gat_layers = int(getattr(configs, "gat_layers", 1))
            self.gat_layers = nn.ModuleList(
                [_SegmentGATLayer(d_model, gat_heads, gat_topk, gat_bias_base, dropout) for _ in range(gat_layers)]
            )

        self.attn_layers = None
        if self.graph_mixer_type == "attn_token":
            attn_heads = int(getattr(configs, "attn_heads", 4))
            attn_topk = int(getattr(configs, "attn_topk", 0))
            attn_layers = int(getattr(configs, "attn_layers", 1))
            self.attn_layers = nn.ModuleList(
                [_TokenAttnLayer(d_model, attn_heads, attn_topk, dropout) for _ in range(attn_layers)]
            )

        self.gcn_layers = None
        if self.graph_mixer_type == "gcn_norm":
            gcn_layers = int(getattr(configs, "gcn_layers", 1))
            self.gcn_norm = str(getattr(configs, "gcn_norm", "row")).lower()
            self.gcn_layers = nn.ModuleList(
                [nn.Linear(d_model, d_model) for _ in range(max(1, gcn_layers))]
            )
            self.gcn_activation = nn.ReLU()
            self.gcn_dropout = nn.Dropout(dropout)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _residual_scale(self):
        scale = self.residual_scale.clamp(0.0, 1.0)
        if self.training and self.warmup_epochs > 0 and self.current_epoch <= self.warmup_epochs:
            return torch.zeros_like(scale)
        return scale

    def _normalize_adj(self, adj):
        bsz, n_vars, _ = adj.shape
        eye = torch.eye(n_vars, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(bsz, -1, -1)
        adj_hat = adj + eye
        if self.gcn_norm == "sym":
            deg = adj_hat.sum(-1).clamp_min(1e-12)
            deg_inv_sqrt = deg.pow(-0.5)
            return deg_inv_sqrt.unsqueeze(-1) * adj_hat * deg_inv_sqrt.unsqueeze(-2)
        return adj_hat / adj_hat.sum(-1, keepdim=True).clamp_min(1e-12)

    def _forward_gcn(self, h_val, adj):
        adj_norm = self._normalize_adj(adj)
        out = h_val
        for layer in self.gcn_layers:
            out = torch.einsum("bij,bjtd->bitd", adj_norm, out)
            out = layer(out)
            out = self.gcn_activation(out)
        out = self.gcn_dropout(out)
        scale = self._residual_scale()
        return h_val + out * scale

    def _forward_gat(self, h_val, h_map, base_adj=None):
        if h_map is None:
            h_map = h_val
        z_map = h_map.mean(dim=2)
        out = h_val
        raw_adj = None
        adj = None
        scale = self._residual_scale()
        for layer in self.gat_layers:
            out, adj, raw_adj = layer(out, z_map, base_adj=base_adj, residual_scale=scale)
        return out, adj, raw_adj

    def _forward_attn(self, h_val, h_map):
        out = h_val
        scale = self._residual_scale()
        for layer in self.attn_layers:
            out = layer(out, h_map=h_map, residual_scale=scale)
        return out

    def forward(self, h_val, h_map=None, adj=None, base_adj=None, token_offset=0):
        if self.graph_mixer_type == "baseline":
            out = self.baseline(adj, h_val, token_offset=token_offset)
            return out, None, None
        if self.graph_mixer_type == "gat_seg":
            return self._forward_gat(h_val, h_map, base_adj=base_adj)
        if self.graph_mixer_type == "attn_token":
            return self._forward_attn(h_val, h_map), None, None
        if self.graph_mixer_type == "gcn_norm":
            if adj is None:
                raise ValueError("gcn_norm requires adj.")
            return self._forward_gcn(h_val, adj), None, None
        raise ValueError(f"Unsupported graph_mixer_type: {self.graph_mixer_type}")
