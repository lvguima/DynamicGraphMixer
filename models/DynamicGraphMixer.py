import os
import numpy as np
import torch
import torch.nn as nn

from modules.temporal import TemporalEncoderWrapper
from modules.graph_learner import LowRankGraphLearner
from modules.graph_map import GraphMapNormalizer
from modules.graph_mixer_v7 import GraphMixerV7
from modules.head import ForecastHead
from modules.gcn import ResidualGCN
from modules.graph_correction import GraphCorrectionHead


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
        self.base_graph_type = str(getattr(configs, "base_graph_type", "learned")).lower()
        self.graph_base_l1 = float(getattr(configs, "graph_base_l1", 0.0))
        self.adj_sparsify = str(getattr(configs, "adj_sparsify", "none")).lower()
        self.graph_mixer_type = str(getattr(configs, "graph_mixer_type", "baseline")).lower()
        self.gate_mode = str(getattr(configs, "gate_mode", "none")).lower()
        self.gate_init = float(getattr(configs, "gate_init", -4.0))
        self.routing_mode = str(getattr(configs, "routing_mode", "none")).lower()
        self.routing_conf_metric = str(getattr(configs, "routing_conf_metric", "overlap_topk")).lower()
        self.routing_gamma = float(getattr(configs, "routing_gamma", 2.0))
        self.routing_warmup_epochs = int(getattr(configs, "routing_warmup_epochs", 0))
        self.routing_l1_scale = float(getattr(configs, "routing_l1_scale", 2.0))
        self.season_gcn_enable = bool(getattr(configs, "season_gcn", False))
        self.season_gcn_layers = int(getattr(configs, "season_gcn_layers", 1))
        self.season_gcn_norm = str(getattr(configs, "season_gcn_norm", "sym")).lower()
        self.season_gcn_g_init = float(getattr(configs, "season_gcn_g_init", 0.0))
        self.season_gcn_g_mode = str(getattr(configs, "season_gcn_g_mode", "scalar")).lower()
        self.season_gcn_adj_source = str(getattr(configs, "season_gcn_adj_source", "prior")).lower()
        self.graph_correction_enable = bool(getattr(configs, "graph_correction", False))
        self.graph_correction_on = str(getattr(configs, "graph_correction_on", "season")).lower()
        self.graph_correction_beta_mode = str(getattr(configs, "graph_correction_beta_mode", "scalar")).lower()
        self.graph_correction_beta_init = float(getattr(configs, "graph_correction_beta_init", 0.0))
        self.graph_correction_gcn_layers = int(getattr(configs, "graph_correction_gcn_layers", 1))
        self.graph_correction_gcn_norm = str(getattr(configs, "graph_correction_gcn_norm", "sym")).lower()
        self.graph_correction_gcn_g_mode = str(getattr(configs, "graph_correction_gcn_g_mode", "scalar")).lower()
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
        base_adj_fixed = None
        if self.graph_base_mode == "mix":
            self.graph_base_alpha = nn.Parameter(torch.tensor(-8.0, dtype=torch.float))
            if self.base_graph_type == "learned":
                self.graph_base_logits = nn.Parameter(torch.zeros(self.enc_in, self.enc_in))
            elif self.base_graph_type == "identity":
                base_adj_fixed = torch.eye(self.enc_in, dtype=torch.float)
            elif self.base_graph_type == "prior":
                base_adj_fixed = self._load_prior_graph(configs)
            else:
                raise ValueError(f"Unsupported base_graph_type: {self.base_graph_type}")
        elif self.graph_base_mode != "none":
            raise ValueError(f"Unsupported graph_base_mode: {self.graph_base_mode}")
        if base_adj_fixed is not None:
            self.register_buffer("graph_base_fixed", base_adj_fixed)
        else:
            self.graph_base_fixed = None
        self.graph_generator = self.graph_learner
        self.graph_mixer = GraphMixerV7(
            configs=configs,
            num_vars=self.enc_in,
            num_tokens=self.num_tokens,
        )
        self.season_gcn = None
        self.season_gcn_adj = None
        if self.season_gcn_enable:
            self.season_gcn = ResidualGCN(
                d_model=configs.d_model,
                num_vars=self.enc_in,
                layers=self.season_gcn_layers,
                norm=self.season_gcn_norm,
                dropout=configs.dropout,
                g_init=self.season_gcn_g_init,
                g_mode=self.season_gcn_g_mode,
            )
            if self.season_gcn_adj_source == "prior":
                if self.graph_base_fixed is not None:
                    self.season_gcn_adj = self.graph_base_fixed
                else:
                    prior_adj = self._load_prior_graph(configs)
                    self.register_buffer("season_gcn_adj", prior_adj)
                    self.season_gcn_adj = self.season_gcn_adj
            elif self.season_gcn_adj_source == "identity":
                eye = torch.eye(self.enc_in, dtype=torch.float)
                self.register_buffer("season_gcn_adj", eye)
                self.season_gcn_adj = self.season_gcn_adj
            else:
                raise ValueError(f"Unsupported season_gcn_adj_source: {self.season_gcn_adj_source}")
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
        self.last_routing_conf = None
        self.last_routing_alpha = None
        self.last_mix_adj = None
        self.last_correction_beta = None
        self.last_correction_delta_norm = None
        self.current_epoch = 0

        self.routing_w = None
        self.routing_b = None
        if self.routing_mode not in ("none", "deterministic", "affine_learned"):
            raise ValueError(f"Unsupported routing_mode: {self.routing_mode}")
        if self.routing_conf_metric not in ("overlap_topk", "l1_distance"):
            raise ValueError(f"Unsupported routing_conf_metric: {self.routing_conf_metric}")
        if self.routing_mode == "affine_learned":
            w_init = float(getattr(configs, "routing_w_init", 2.0))
            b_init = float(getattr(configs, "routing_b_init", 0.0))
            self.routing_w = nn.Parameter(torch.tensor(w_init, dtype=torch.float))
            self.routing_b = nn.Parameter(torch.tensor(b_init, dtype=torch.float))

        self.graph_correction = None
        self.graph_correction_beta = None
        if self.graph_correction_enable:
            if self.graph_correction_on not in ("season", "full"):
                raise ValueError("graph_correction_on must be 'season' or 'full'.")
            self.graph_correction = GraphCorrectionHead(
                d_model=configs.d_model,
                num_vars=self.enc_in,
                num_tokens=self.num_tokens,
                pred_len=self.pred_len,
                gcn_layers=self.graph_correction_gcn_layers,
                gcn_norm=self.graph_correction_gcn_norm,
                gcn_g_mode=self.graph_correction_gcn_g_mode,
                dropout=configs.dropout,
            )
            if self.graph_correction_beta_mode == "scalar":
                self.graph_correction_beta = nn.Parameter(torch.tensor(self.graph_correction_beta_init))
            elif self.graph_correction_beta_mode == "per_var":
                self.graph_correction_beta = nn.Parameter(
                    torch.full((1, 1, self.c_out), float(self.graph_correction_beta_init))
                )
            else:
                raise ValueError("graph_correction_beta_mode must be 'scalar' or 'per_var'.")

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

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

    def _resolve_prior_path(self, configs):
        path = str(getattr(configs, "prior_graph_path", "")).strip()
        if path:
            return path
        root = str(getattr(configs, "prior_graph_dir", "./prior_graphs")).strip()
        method = str(getattr(configs, "prior_graph_method", "pearson_abs")).strip()
        topk = int(getattr(configs, "prior_graph_topk", 8))
        data_path = str(getattr(configs, "data_path", "")).strip()
        stem = os.path.splitext(os.path.basename(data_path))[0] or str(getattr(configs, "data", "dataset"))
        topk = max(0, min(topk, self.enc_in - 1))
        name = f"{stem}_{method}_topk{topk}.npy"
        return os.path.join(root, name)

    def _load_prior_graph(self, configs):
        path = self._resolve_prior_path(configs)
        if not os.path.exists(path):
            raise FileNotFoundError(f"prior graph not found: {path}")
        arr = np.load(path)
        if arr.shape != (self.enc_in, self.enc_in):
            raise ValueError(f"prior graph shape mismatch: {arr.shape} vs {(self.enc_in, self.enc_in)}")
        return torch.tensor(arr, dtype=torch.float)

    def _routing_confidence(self, dyn_adj, base_adj):
        if base_adj is None:
            return None
        if self.routing_conf_metric == "overlap_topk":
            k = max(1, int(self.adj_topk))
            scores_dyn = dyn_adj.clone()
            scores_base = base_adj.clone()
            scores_dyn.diagonal(dim1=-2, dim2=-1).fill_(-float("inf"))
            scores_base.diagonal(dim1=-2, dim2=-1).fill_(-float("inf"))
            k = min(k, scores_dyn.shape[-1])
            dyn_idx = torch.topk(scores_dyn, k, dim=-1).indices
            base_idx = torch.topk(scores_base, k, dim=-1).indices
            matches = dyn_idx.unsqueeze(-1) == base_idx.unsqueeze(-2)
            overlap = matches.any(-1).float().mean(-1)
            return overlap.mean(-1).clamp(0.0, 1.0)
        if self.routing_conf_metric == "l1_distance":
            l1 = torch.abs(dyn_adj - base_adj).mean(dim=(-1, -2))
            conf = 1.0 - (l1 / max(1e-6, float(self.routing_l1_scale)))
            return conf.clamp(0.0, 1.0)
        raise ValueError(f"Unsupported routing_conf_metric: {self.routing_conf_metric}")

    def _routing_alpha(self, conf):
        if conf is None:
            return None
        if self.routing_mode == "deterministic":
            alpha = (1.0 - conf).clamp(0.0, 1.0) ** max(1e-6, float(self.routing_gamma))
            return alpha
        if self.routing_mode == "affine_learned":
            return torch.sigmoid(self.routing_w * (1.0 - conf) + self.routing_b)
        return None

    def _mix_segments(self, h_time, h_graph=None):
        # h_time: [B, C, N, D], h_graph provides map features for adjacency
        if h_graph is None:
            h_graph = h_time
        bsz, n_vars, num_tokens, dim = h_time.shape
        scale = min(self.graph_scale, num_tokens)

        base_adj = None
        base_reg = None
        base_alpha = None
        if self.graph_base_mode == "mix":
            base_alpha = torch.sigmoid(self.graph_base_alpha)
            if self.graph_base_fixed is not None:
                base_adj_single = self.graph_base_fixed
                base_adj = base_adj_single.unsqueeze(0).expand(bsz, -1, -1)
            else:
                base_adj_single = torch.softmax(self.graph_base_logits, dim=-1)
                base_adj = base_adj_single.unsqueeze(0).expand(bsz, -1, -1)
                if self.graph_base_l1 > 0:
                    base_reg = base_adj_single.abs().mean()

        segments = []
        prev_adj = None
        log_adjs = [] if self.graph_log else None
        log_raw_adjs = [] if self.graph_log else None
        adj_sum = None
        adj_count = 0
        if self.graph_log and base_adj is not None:
            self.last_graph_base_adj = base_adj_single.detach().cpu()
        elif self.graph_log:
            self.last_graph_base_adj = None
        conf_list = [] if self.graph_log else None
        routing_alpha_list = [] if self.graph_log else None

        for start in range(0, num_tokens, scale):
            end = min(start + scale, num_tokens)
            h_seg = h_time[:, :, start:end, :]
            h_graph_seg = h_graph[:, :, start:end, :]

            log_adj = None
            log_raw = None

            if self.graph_mixer_type in ("baseline", "gcn_norm"):
                z_t = h_graph_seg.mean(dim=2)
                dyn_adj, _, _ = self.graph_learner(z_t)
                adj = dyn_adj
                if base_adj is not None:
                    if self.routing_mode != "none":
                        conf = self._routing_confidence(dyn_adj, base_adj)
                        alpha = self._routing_alpha(conf)
                        if self.training and self.routing_warmup_epochs > 0:
                            if self.current_epoch <= self.routing_warmup_epochs:
                                alpha = torch.ones_like(alpha)
                        if alpha is not None:
                            alpha_view = alpha.view(-1, 1, 1)
                            adj = (1.0 - alpha_view) * dyn_adj + alpha_view * base_adj
                            if conf_list is not None:
                                conf_list.append(conf.detach().cpu())
                            if routing_alpha_list is not None:
                                routing_alpha_list.append(alpha.detach().cpu())
                    elif base_alpha is not None:
                        adj = (1.0 - base_alpha) * dyn_adj + base_alpha * base_adj
                log_raw = adj
                adj = self._sparsify_adj(adj)
                if self.graph_correction_enable and adj is not None:
                    adj_sum = adj if adj_sum is None else (adj_sum + adj)
                    adj_count += 1
                prev_adj = adj
                log_adj = adj
                h_seg, _, _ = self.graph_mixer(
                    h_seg,
                    h_graph_seg,
                    adj=adj,
                    base_adj=base_adj,
                    token_offset=start,
                )
            elif self.graph_mixer_type == "gat_seg":
                h_seg, log_adj, log_raw = self.graph_mixer(
                    h_seg,
                    h_graph_seg,
                    base_adj=base_adj,
                    token_offset=start,
                )
            elif self.graph_mixer_type == "attn_token":
                h_seg, _, _ = self.graph_mixer(
                    h_seg,
                    h_graph_seg,
                    token_offset=start,
                )
            else:
                raise ValueError(f"Unsupported graph_mixer_type: {self.graph_mixer_type}")

            if log_raw_adjs is not None and log_raw is not None:
                log_raw_adjs.append(log_raw.detach().cpu())
            if log_adjs is not None and log_adj is not None:
                log_adjs.append(log_adj.detach().cpu())
            segments.append(h_seg)
        h_mix = torch.cat(segments, dim=2)
        if base_reg is None:
            base_reg = h_time.new_tensor(0.0)
        self.graph_base_reg_loss = base_reg
        if self.graph_correction_enable:
            self.last_mix_adj = adj_sum / max(1, adj_count) if adj_sum is not None else None
        else:
            self.last_mix_adj = None
        if log_adjs is not None:
            self.last_graph_adjs = log_adjs
            self.last_graph_raw_adjs = log_raw_adjs
            if conf_list:
                self.last_routing_conf = torch.stack(conf_list, dim=0).mean(dim=0)
            else:
                self.last_routing_conf = None
            if routing_alpha_list:
                self.last_routing_alpha = torch.stack(routing_alpha_list, dim=0).mean(dim=0)
            else:
                self.last_routing_alpha = None
        else:
            self.last_graph_adjs = None
            self.last_graph_raw_adjs = None
            self.last_graph_base_adj = None
            self.last_routing_conf = None
            self.last_routing_alpha = None
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

    def _forecast_graph(self, x_enc, use_season_gcn=False, return_latent=False):
        h_time = self.temporal_encoder(x_enc)
        if use_season_gcn and self.season_gcn is not None:
            if self.season_gcn_adj is None:
                raise ValueError("season_gcn is enabled but season_gcn_adj is not set.")
            h_time = self.season_gcn(h_time, self.season_gcn_adj)
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

        out = self.head(h_mix)
        if return_latent:
            return out, h_mix
        return out

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
            season_out, season_latent = self._forecast_graph(x_season, use_season_gcn=True, return_latent=True)
            trend_out = self._forecast_trend(x_trend)
            base_out = season_out + trend_out
            if self.graph_correction_enable and self.graph_correction is not None:
                if self.graph_correction_on == "season":
                    corr_input = season_latent
                else:
                    _, full_latent = self._forecast_graph(x_enc, use_season_gcn=False, return_latent=True)
                    corr_input = full_latent
                corr_adj = self.last_mix_adj
                if corr_adj is not None:
                    delta_y = self.graph_correction(corr_input, corr_adj)
                    beta = self.graph_correction_beta
                    if beta is None:
                        beta = base_out.new_tensor(0.0)
                    if beta.dim() == 0:
                        beta_view = beta.view(1, 1, 1)
                    else:
                        beta_view = beta
                    self.last_correction_beta = beta.detach().cpu()
                    self.last_correction_delta_norm = delta_y.detach().abs().mean().cpu()
                    return base_out + beta_view * delta_y
            self.last_correction_beta = None
            self.last_correction_delta_norm = None
            return base_out

        self.last_decomp_energy = None
        out = self._forecast_graph(x_enc, use_season_gcn=False)
        self.last_correction_beta = None
        self.last_correction_delta_norm = None
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)
        raise ValueError("DynamicGraphMixer only supports forecast tasks for now.")
