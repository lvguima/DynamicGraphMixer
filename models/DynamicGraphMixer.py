import torch
import torch.nn as nn

from layers.TemporalEncoder import TemporalEncoderTCN, TemporalEncoderTransformer
from layers.DynamicGraph import LowRankGraphGenerator, LinearGraphMixing


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.graph_scale = max(1, int(getattr(configs, "graph_scale", 1)))
        self.graph_rank = int(getattr(configs, "graph_rank", 8))
        self.graph_smooth_lambda = float(getattr(configs, "graph_smooth_lambda", 0.0))

        self.use_patch = bool(getattr(configs, "use_patch", False))
        self.patch_len = int(getattr(configs, "patch_len", 16))
        self.patch_stride = int(getattr(configs, "patch_stride", self.patch_len))

        if self.patch_len <= 0:
            self.patch_len = self.seq_len
        self.patch_len = min(self.patch_len, self.seq_len)
        if self.patch_stride <= 0:
            self.patch_stride = self.patch_len
        self.patch_stride = min(self.patch_stride, self.patch_len)

        tcn_kernel = int(getattr(configs, "tcn_kernel", 3))
        tcn_dilation = int(getattr(configs, "tcn_dilation", 2))
        temporal_encoder = getattr(configs, "temporal_encoder", "tcn").lower()
        if temporal_encoder == "transformer":
            self.temporal_encoder = TemporalEncoderTransformer(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                num_layers=configs.e_layers,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
                activation=getattr(configs, "activation", "gelu"),
                attn_factor=getattr(configs, "factor", 1),
            )
        elif temporal_encoder == "tcn":
            self.temporal_encoder = TemporalEncoderTCN(
                d_model=configs.d_model,
                num_layers=configs.e_layers,
                kernel_size=tcn_kernel,
                dilation_base=tcn_dilation,
                dropout=configs.dropout,
            )
        else:
            raise ValueError(f"Unsupported temporal_encoder: {temporal_encoder}")
        self.graph_generator = LowRankGraphGenerator(
            in_dim=configs.d_model,
            rank=self.graph_rank,
            dropout=configs.dropout,
        )
        self.graph_mixer = LinearGraphMixing(dropout=configs.dropout)

        if self.use_patch:
            self.num_tokens = (self.seq_len - self.patch_len) // self.patch_stride + 1
        else:
            self.num_tokens = self.seq_len
        self.head = nn.Linear(configs.d_model * self.num_tokens, self.pred_len)
        self.graph_reg_loss = None
        self.graph_log = bool(getattr(configs, "graph_log", False))
        self.last_graph_adjs = None

    def _mix_segments(self, h_time):
        # h_time: [B, C, N, D]
        bsz, n_vars, num_tokens, dim = h_time.shape
        scale = min(self.graph_scale, num_tokens)

        segments = []
        reg = None
        prev_adj = None
        log_adjs = [] if self.graph_log else None
        for start in range(0, num_tokens, scale):
            end = min(start + scale, num_tokens)
            h_seg = h_time[:, :, start:end, :]
            z_t = h_seg.mean(dim=2)
            adj, _, _ = self.graph_generator(z_t)
            if self.graph_smooth_lambda > 0 and prev_adj is not None:
                reg_term = torch.mean(torch.abs(adj - prev_adj))
                reg = reg_term if reg is None else reg + reg_term
            prev_adj = adj
            if log_adjs is not None:
                log_adjs.append(adj.detach().cpu())
            h_seg = self.graph_mixer(adj, h_seg)
            segments.append(h_seg)
        h_mix = torch.cat(segments, dim=2)
        if reg is None:
            reg = h_time.new_tensor(0.0)
        if log_adjs is not None:
            self.last_graph_adjs = log_adjs
        else:
            self.last_graph_adjs = None
        return h_mix, reg

    def _apply_patch_pooling(self, h_time):
        # h_time: [B, C, L, D]
        if not self.use_patch:
            return h_time
        bsz, n_vars, seq_len, dim = h_time.shape
        patch_len = min(self.patch_len, seq_len)
        stride = min(self.patch_stride, patch_len)

        h = h_time.reshape(bsz * n_vars, seq_len, dim)
        patches = h.unfold(dimension=1, size=patch_len, step=stride)
        h_pooled = patches.mean(dim=2)
        return h_pooled.reshape(bsz, n_vars, -1, dim)

    def forecast(self, x_enc):
        h_time = self.temporal_encoder(x_enc)
        h_time = self._apply_patch_pooling(h_time)
        h_mix, reg = self._mix_segments(h_time)
        self.graph_reg_loss = reg

        if self.c_out < h_mix.shape[1]:
            h_mix = h_mix[:, -self.c_out:, :, :]

        bsz, out_vars, num_tokens, dim = h_mix.shape
        h_flat = h_mix.reshape(bsz, out_vars, num_tokens * dim)
        out = self.head(h_flat)
        return out.permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)
        raise ValueError("DynamicGraphMixer only supports forecast tasks for now.")
