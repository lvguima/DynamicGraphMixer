import torch
import torch.nn as nn

from modules.tokenizer import PatchTokenizer
from modules.temporal import TemporalEncoderWrapper
from modules.graph_learner import LowRankGraphLearner
from modules.mixer import GraphMixer
from modules.head import ForecastHead


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
        patch_len = int(getattr(configs, "patch_len", 16))
        patch_stride = int(getattr(configs, "patch_stride", patch_len))
        self.tokenizer = PatchTokenizer(
            seq_len=self.seq_len,
            use_patch=self.use_patch,
            patch_len=patch_len,
            patch_stride=patch_stride,
        )
        self.patch_len = self.tokenizer.patch_len
        self.patch_stride = self.tokenizer.patch_stride
        self.num_tokens = self.tokenizer.num_tokens

        self.temporal_encoder = TemporalEncoderWrapper(configs)
        self.graph_learner = LowRankGraphLearner(
            in_dim=configs.d_model,
            rank=self.graph_rank,
            dropout=configs.dropout,
        )
        self.graph_generator = self.graph_learner
        self.graph_mixer = GraphMixer(dropout=configs.dropout)
        self.head = ForecastHead(
            d_model=configs.d_model,
            num_tokens=self.num_tokens,
            pred_len=self.pred_len,
        )
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
            adj, _, _ = self.graph_learner(z_t)
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
        return self.tokenizer(h_time)

    def forecast(self, x_enc):
        h_time = self.temporal_encoder(x_enc)
        h_time = self._apply_patch_pooling(h_time)
        h_mix, reg = self._mix_segments(h_time)
        self.graph_reg_loss = reg

        if self.c_out < h_mix.shape[1]:
            h_mix = h_mix[:, -self.c_out:, :, :]

        return self.head(h_mix)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)
        raise ValueError("DynamicGraphMixer only supports forecast tasks for now.")
