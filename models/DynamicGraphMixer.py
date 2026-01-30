import torch
import torch.nn as nn

from modules.temporal import TemporalEncoderWrapper
from modules.graph_learner import LowRankGraphLearner
from modules.graph_map import GraphMapNormalizer
from modules.mixer import GraphMixer
from modules.head import ForecastHead


class TrendHead(nn.Module):
    def __init__(self, seq_len, pred_len, num_vars, share=True):
        super().__init__()
        self.share = bool(share)
        self.num_vars = int(num_vars)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        if self.share:
            self.proj = nn.Linear(self.seq_len, self.pred_len)
        else:
            self.proj = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.num_vars)]
            )

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1).contiguous()
        if self.share:
            out = self.proj(x)
        else:
            outs = []
            for idx in range(self.num_vars):
                outs.append(self.proj[idx](x[:, idx, :]))
            out = torch.stack(outs, dim=1)
        return out.permute(0, 2, 1)


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
        self.graph_map_norm = str(getattr(configs, "graph_map_norm", "none")).lower()
        self.graph_map_window = int(getattr(configs, "graph_map_window", 16))
        self.graph_map_alpha = float(getattr(configs, "graph_map_alpha", 0.3))
        self.graph_map_detach = bool(getattr(configs, "graph_map_detach", False))
        self.decomp_mode = str(getattr(configs, "decomp_mode", "none")).lower()
        self.decomp_alpha = float(getattr(configs, "decomp_alpha", 0.3))
        self.trend_head_mode = str(getattr(configs, "trend_head", "none")).lower()
        self.trend_head_share = bool(int(getattr(configs, "trend_head_share", 1)))
        self.trend_only = bool(getattr(configs, "trend_only", False))
        self.graph_base_mode = str(getattr(configs, "graph_base_mode", "none")).lower()
        self.graph_base_l1 = float(getattr(configs, "graph_base_l1", 0.0))
        self.adj_sparsify = str(getattr(configs, "adj_sparsify", "none")).lower()
        self.gate_mode = str(getattr(configs, "gate_mode", "none")).lower()
        self.gate_init = float(getattr(configs, "gate_init", -4.0))
        # Removed features: stable_stream graph source, patch/token pooling, and graph smoothness.
        # Keep backward compatibility by validating deprecated args.
        graph_source = str(getattr(configs, "graph_source", "content_mean")).lower()
        if graph_source != "content_mean":
            raise ValueError("graph_source=stable_stream is no longer supported (content_mean only).")
        if bool(getattr(configs, "use_patch", False)):
            raise ValueError("Patch/Token Pooling is no longer supported (use_patch must be false).")
        graph_smooth = float(getattr(configs, "graph_smooth_lambda", 0.0))
        if abs(graph_smooth) > 0:
            raise ValueError("graph_smooth_lambda is deprecated and must be 0.")
        adj_topk = int(getattr(configs, "adj_topk", 6))
        if self.adj_sparsify == "topk" and adj_topk != 6:
            raise ValueError("adj_topk is deprecated and fixed to 6.")
        base_alpha_init = float(getattr(configs, "graph_base_alpha_init", -8.0))
        if self.graph_base_mode == "mix" and base_alpha_init != -8.0:
            raise ValueError("graph_base_alpha_init is deprecated and fixed to -8.")
        self.adj_topk = 6
        self.num_tokens = self.seq_len

        self.temporal_encoder = TemporalEncoderWrapper(configs)
        self.graph_map = GraphMapNormalizer(
            mode=self.graph_map_norm,
            window=self.graph_map_window,
            alpha=self.graph_map_alpha,
        )
        self.trend_head = None
        if self.decomp_mode not in ("none", "ema"):
            raise ValueError(f"Unsupported decomp_mode: {self.decomp_mode}")
        if self.trend_head_mode not in ("none", "linear"):
            raise ValueError(f"Unsupported trend_head: {self.trend_head_mode}")
        if self.decomp_mode == "ema" and self.trend_head_mode == "linear":
            self.trend_head = TrendHead(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                num_vars=self.enc_in,
                share=self.trend_head_share,
            )
        if self.trend_only:
            if self.decomp_mode != "ema" or self.trend_head is None:
                raise ValueError("trend_only requires decomp_mode=ema and trend_head=linear.")
        self.graph_learner = LowRankGraphLearner(
            in_dim=configs.d_model,
            rank=self.graph_rank,
            dropout=configs.dropout,
        )
        self.graph_base_logits = None
        self.graph_base_alpha = None
        if self.graph_base_mode == "mix":
            self.graph_base_logits = nn.Parameter(torch.zeros(self.enc_in, self.enc_in))
            self.graph_base_alpha = nn.Parameter(
                torch.tensor(-8.0, dtype=torch.float)
            )
        elif self.graph_base_mode != "none":
            raise ValueError(f"Unsupported graph_base_mode: {self.graph_base_mode}")
        self.graph_generator = self.graph_learner
        self.graph_mixer = GraphMixer(
            dropout=configs.dropout,
            gate_mode=self.gate_mode,
            num_vars=self.enc_in,
            num_tokens=self.num_tokens,
            gate_init=self.gate_init,
        )
        self.head = ForecastHead(
            d_model=configs.d_model,
            num_tokens=self.num_tokens,
            pred_len=self.pred_len,
        )
        self.graph_reg_loss = None
        self.graph_base_reg_loss = None
        self.graph_log = bool(getattr(configs, "graph_log", False))
        self.last_graph_adjs = None
        self.last_graph_raw_adjs = None
        self.last_graph_base_adj = None
        self.last_graph_map_mean_abs = None
        self.last_graph_map_std_mean = None
        self.last_decomp_energy = None

    def _sparsify_adj(self, adj):
        if self.adj_sparsify != "topk":
            return adj
        k = max(0, int(self.adj_topk))
        if k <= 0:
            return adj
        bsz, n_vars, _ = adj.shape
        k = min(k, n_vars)
        vals, idx = torch.topk(adj, k, dim=-1)
        mask = torch.zeros_like(adj)
        mask.scatter_(-1, idx, 1.0)
        masked = adj * mask
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return masked / denom

    def _mix_segments(self, h_time, h_graph=None):
        # h_time: [B, C, N, D], h_graph provides map features for adjacency
        if h_graph is None:
            h_graph = h_time
        bsz, n_vars, num_tokens, dim = h_time.shape
        scale = min(self.graph_scale, num_tokens)

        base_adj = None
        base_reg = None
        alpha = None
        if self.graph_base_mode == "mix":
            base_adj_single = torch.softmax(self.graph_base_logits, dim=-1)
            base_adj = base_adj_single.unsqueeze(0).expand(bsz, -1, -1)
            alpha = torch.sigmoid(self.graph_base_alpha)
            if self.graph_base_l1 > 0:
                base_reg = base_adj_single.abs().mean()

        segments = []
        prev_adj = None
        log_adjs = [] if self.graph_log else None
        log_raw_adjs = [] if self.graph_log else None
        if self.graph_log and base_adj is not None:
            self.last_graph_base_adj = base_adj_single.detach().cpu()
        elif self.graph_log:
            self.last_graph_base_adj = None
        for start in range(0, num_tokens, scale):
            end = min(start + scale, num_tokens)
            h_seg = h_time[:, :, start:end, :]
            h_graph_seg = h_graph[:, :, start:end, :]
            z_t = h_graph_seg.mean(dim=2)
            adj, _, _ = self.graph_learner(z_t)
            if base_adj is not None:
                adj = (1.0 - alpha) * adj + alpha * base_adj
            if log_raw_adjs is not None:
                log_raw_adjs.append(adj.detach().cpu())
            adj = self._sparsify_adj(adj)
            prev_adj = adj
            if log_adjs is not None:
                log_adjs.append(adj.detach().cpu())
            h_seg = self.graph_mixer(adj, h_seg, token_offset=start)
            segments.append(h_seg)
        h_mix = torch.cat(segments, dim=2)
        if base_reg is None:
            base_reg = h_time.new_tensor(0.0)
        self.graph_base_reg_loss = base_reg
        if log_adjs is not None:
            self.last_graph_adjs = log_adjs
            self.last_graph_raw_adjs = log_raw_adjs
        else:
            self.last_graph_adjs = None
            self.last_graph_raw_adjs = None
            self.last_graph_base_adj = None
        return h_mix, None

    def _ema_decompose(self, x_enc):
        # x_enc: [B, L, C]
        alpha = max(0.0, min(1.0, float(self.decomp_alpha)))
        bsz, seq_len, n_vars = x_enc.shape
        if seq_len == 0:
            return x_enc, x_enc
        trend = []
        prev = x_enc[:, 0, :]
        trend.append(prev)
        for idx in range(1, seq_len):
            prev = alpha * x_enc[:, idx, :] + (1.0 - alpha) * prev
            trend.append(prev)
        trend = torch.stack(trend, dim=1)
        season = x_enc - trend
        return trend, season

    def _forecast_graph(self, x_enc):
        h_time = self.temporal_encoder(x_enc)
        h_graph = h_time
        h_map = self.graph_map(h_graph)
        if self.graph_map_detach and h_map is not None:
            h_map = h_map.detach()
        if self.graph_log and h_map is not None:
            self.last_graph_map_mean_abs = h_map.detach().abs().mean().cpu()
            self.last_graph_map_std_mean = h_map.detach().std(dim=-1).mean().cpu()
        else:
            self.last_graph_map_mean_abs = None
            self.last_graph_map_std_mean = None
        h_mix, _ = self._mix_segments(h_time, h_map)

        if self.c_out < h_mix.shape[1]:
            h_mix = h_mix[:, -self.c_out:, :, :]

        return self.head(h_mix)

    def _forecast_trend(self, x_trend):
        if self.trend_head is None:
            return x_trend.new_zeros((x_trend.shape[0], self.pred_len, self.c_out))
        out = self.trend_head(x_trend)
        if self.c_out < out.shape[2]:
            out = out[:, :, -self.c_out:]
        return out

    def forecast(self, x_enc):
        if self.decomp_mode == "ema":
            x_trend, x_season = self._ema_decompose(x_enc)
            if self.graph_log:
                trend_energy = x_trend.detach().pow(2).mean().cpu()
                season_energy = x_season.detach().pow(2).mean().cpu()
                denom = trend_energy + season_energy + 1e-12
                ratio = trend_energy / denom
                self.last_decomp_energy = (trend_energy, season_energy, ratio)
            else:
                self.last_decomp_energy = None
            if self.trend_only:
                return self._forecast_trend(x_trend)
            season_out = self._forecast_graph(x_season)
            trend_out = self._forecast_trend(x_trend)
            return season_out + trend_out

        self.last_decomp_energy = None
        return self._forecast_graph(x_enc)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)
        raise ValueError("DynamicGraphMixer only supports forecast tasks for now.")
