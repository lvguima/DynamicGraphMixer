# DynamicGraphMixer Model Design (B5 Baseline, v3)

This document describes the DynamicGraphMixer architecture **as used in the B5 setting**
(Dual-Stream + SMGP + graph propagation with dynamic graph learning, base-graph mixing,
and gate). It mirrors the implementation in `models/DynamicGraphMixer.py` and related
modules and can be used directly as the model-structure reference in the paper.

**Note:** The following experimental design options have been removed from the codebase:
temporal transformer encoder, patch/token pooling, `graph_source=stable_stream`,
`adj_topk` tuning, `graph_base_alpha_init` tuning, and `graph_smooth_lambda`.

## 1. Scope and Assumptions

- Task: multivariate long-term forecasting (the model only supports forecast tasks).
- Input shape: `x_enc` is [B, L, C] where B=batch size, L=seq_len, C=variables.
- Output shape: prediction is [B, pred_len, C_out].
- All shapes below follow this convention unless stated.

## 2. High-Level Architecture (B5)

Under B5, DynamicGraphMixer combines:

1) Temporal encoding (TCN) for each variable.
2) Dynamic graph construction per segment.
3) Graph mixing with a learnable gate.
4) Forecast head for season branch.
5) Dual-stream trend branch (EMA decomposition + trend head).

In B5, the final output is:

    y_hat = ForecastGraph(season) + TrendHead(trend)

This dual-stream sum is always used in B5.

## 3. Temporal Encoder (TCN)

`TemporalEncoderWrapper` uses **TCN only** in the current codebase:

- **TCN** (`temporal_encoder=tcn`):
  - Causal Conv1D with exponential dilation (dilation_base^i).
  - Input reshaped to [B*C, 1, L], output reshaped to [B, C, L, D].
Key args: `d_model`, `e_layers`, `d_ff`, `dropout`, `tcn_kernel`, `tcn_dilation`.

## 4. Dual-Stream Decomposition (B5)

In B5, `decomp_mode=ema`, so the input is decomposed:

- trend_t = alpha * x_t + (1 - alpha) * trend_(t-1)
- season_t = x_t - trend_t

The season branch is fed into graph forecasting. The trend branch is fed into
`TrendHead`, a linear projection from seq_len to pred_len. In B5, the trend head is
**shared across variables** (`trend_head_share=1`).

Key args: `decomp_mode`, `decomp_alpha`, `trend_head`, `trend_head_share`.

## 5. Graph Token Source and SMGP (B5)

Graph construction uses **content_mean only** (stable-stream options are removed).

Graph map normalization (SMGP) is applied before graph learning. In B5 it is:

- `graph_map_norm=ma_detrend`, `graph_map_window=16`
- `graph_map_detach` is off by default (gradients flow through the map branch).

Key args: `graph_map_norm`, `graph_map_window`, `graph_map_alpha`, `graph_map_detach`.

## 6. Dynamic Graph Construction (B5)

Graph is constructed per segment:

1) Tokens are split into segments of size `graph_scale` (min with num_tokens).
2) For each segment, token features are averaged:
   - `z_t = mean(h_graph_seg over tokens)`
   - shape of z_t is [B, C, D]
3) Low-rank graph learner produces adjacency:
   - `A = softmax(U V^T / sqrt(rank))`

Base-graph mixing (enabled in B5):

- If `graph_base_mode=mix`, a learned base adjacency `A_base` is mixed in:
  - `A = (1 - alpha) * A_dyn + alpha * A_base`
  - `alpha = sigmoid(graph_base_alpha)`

Sparsification (enabled in B5):

- If `adj_sparsify=topk`, keep top-k per row and renormalize.
- In the current code, `adj_topk` is **fixed to 6** (no tuning).

Key args: `graph_rank`, `graph_scale`, `graph_base_mode`,
`graph_base_alpha_init`, `graph_base_l1`, `adj_sparsify`, `adj_topk`.

## 7. Graph Mixing with Gate (B5)

For each segment, mixing is:

    mixed = A @ x
    mixed = gate * mixed (if gate enabled)
    x_out = x + dropout(mixed)

Gate is stored as a logit; actual gate is `sigmoid(gate_param)`.
In B5 we use **per-variable gates** (`gate_mode=per_var`) with a weak initial
propagation strength (`gate_init=-6`).

Key args: `gate_mode`, `gate_init`.

## 8. Forecast Head

The season branch uses `ForecastHead`:

- Flatten tokens: [B, C, N, D] -> [B, C, N*D]
- Linear projection to `pred_len`.
- Output [B, pred_len, C_out].

If `c_out < enc_in`, output uses last `c_out` variables.

## 9. Regularization and Loss

Training loss is:

    L = MSE(pred, target)
        + graph_base_l1 * mean(|A_base|)

Only base adjacency L1 regularization is used in the current code.

Key args: `graph_base_l1`.

## 10. Logging and Metrics

If `graph_log_interval > 0`, graph statistics are logged to:

    graph_logs/<graph_log_exp_id or setting>/stats.csv

Logged statistics include:

- adjacency entropy/overlap/diff and related stats
- gate and base alpha stats
- SMGP map stats (mean abs, std)
- decomposition energy (trend/season/ratio)

After testing, MSE/MAE are appended to the same `stats.csv`.

Key args: `graph_log_interval`, `graph_log_topk`,
`graph_log_num_segments`, `graph_log_dir`, `graph_log_exp_id`.

## 11. B5 Configuration Summary (Paper-Ready)

This is the B5 configuration used in v3 ablations:

- Temporal encoder: **TCN** (kernel=3, dilation=2), `e_layers=2`, `d_model=128`, `d_ff=256`
- Dual-Stream: `decomp_mode=ema`, `decomp_alpha=0.1`,
  `trend_head=linear`, `trend_head_share=1`
- SMGP: `graph_map_norm=ma_detrend`, `graph_map_window=16`
- Dynamic graph: `graph_rank=8`, `graph_scale=8`, `adj_sparsify=topk`, `adj_topk=6` (fixed)
- Base graph mixing: `graph_base_mode=mix`, `graph_base_alpha_init=-8` (fixed), `graph_base_l1=0.001`
- Graph mixing gate: `gate_mode=per_var`, `gate_init=-6`

These values define the **B5 baseline** in the paper and should be listed as the
default configuration before ablation.
