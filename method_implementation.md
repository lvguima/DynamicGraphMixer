---
title: Dynamic Graph Mixer Forecaster - Implementation Notes
status: draft
scope: code-level design
---

# Dynamic Graph Mixer Forecaster - Implementation Notes

This document translates the method design into a code-level plan that fits the
current repository. It focuses on module boundaries, file layout, interfaces,
shapes, and configuration knobs so we can implement and test variants step by step.

## 0. Assumptions and scope

* Primary target tasks: long_term_forecast and short_term_forecast.
* Input pipeline follows the default forecasting setup:
  * Raw CSV is read by the dataset loader.
  * If `--use_norm=1` (default), a `StandardScaler` is fit on the train split and
    applied to all splits; `x_enc`/`x_dec` fed into the model are normalized.
  * If `--use_norm=0`, raw values are used.
  * `--inverse` only affects evaluation/visualization; the model itself never
    denormalizes outputs.
* We avoid online learning and keep all graph generation inside forward.
* All shapes are in batch-first format.

## 1. Proposed file layout

* `models/DynamicGraphMixer.py` (new)
  * Model entry point, task routing, forecast head.
* `layers/DynamicGraph.py` (new)
  * GraphGenerator variants and GraphMixing operators.
* `layers/TemporalEncoder.py` (new)
  * Temporal encoder variants (TCN, Transformer, SSM).
* `models/__init__.py` (update)
  * Register `DynamicGraphMixer`.
* `exp/exp_basic.py` (update)
  * Add model to registry.
* `run.py` (update)
  * Add CLI args for graph and encoder options.

## 2. Shape conventions

* Input time series: `x_enc` shape `[B, L, C]`.
* Temporal tokens per variable:
  * If `use_patch` is true: `N = floor((L - patch_len) / stride) + 1` (no padding).
  * Else: `N = L` (point-level tokens).
* Temporal encoder output:
  * `H_time` shape `[B, C, N, d]`.
* Graphs:
  * `A_t` shape `[B, C, C]` per coarse segment t.
* Mixed features:
  * `H_mix` shape `[B, C, N, d]` after graph mixing.
* Forecast output:
  * `Y_hat` shape `[B, pred_len, C]`.

## 3. Configuration surface (new args)

### 3.1 Temporal encoder

* `--temporal_encoder`: `tcn | transformer | ssm`
* `--temporal_dim`: embedding dim d
* `--temporal_layers`: number of encoder blocks
* `--temporal_kernel`: kernel size for TCN
* `--temporal_dilation`: dilation base for TCN
* `--temporal_dropout`: dropout in encoder
* `--use_patch`: bool
* `--patch_len`, `--patch_stride`

### 3.2 Graph generator

* `--graph_mode`: `low_rank | topk | dict | prior_residual`
* `--graph_rank`: low-rank r
* `--graph_topk`: top-k neighbors
* `--graph_dict_size`: number of graph prototypes
* `--graph_prior`: `none | identity | file`
* `--graph_prior_path`: path to prior adjacency (optional)
* `--graph_normalize`: `row_softmax | row_sum | sym`
* `--graph_directed`: bool
* `--graph_input`: `pool_time | stats | exog | concat`

### 3.3 Graph scale and regularization

* `--graph_scale`: coarse segment length in tokens (P_long)
* `--graph_pooling`: `avg | max | last`
* `--graph_smooth_lambda`: smoothness weight
* `--graph_smooth_target`: `A | U` (apply regularizer to adjacency or factors)

### 3.4 Graph mixing

* `--graph_mix`: `linear | attention | diffusion`
* `--graph_attn_heads`: for attention mixing
* `--graph_diffusion_steps`: for diffusion
* `--graph_mix_dropout`

### 3.5 Output head

* `--head_type`: `linear | seq2seq`

## 4. Module designs

### 4.1 TemporalEncoder (layers/TemporalEncoder.py)

Common interface:

* `forward(x_enc) -> H_time`
  * `x_enc` shape `[B, L, C]`
  * Output `H_time` shape `[B, C, N, d]`

Variants:

1) TCN
   * Reshape to `[B * C, 1, L]` and apply dilated 1D conv stacks.
   * Project to `d`, then reshape back to `[B, C, N, d]`.
2) Transformer
   * Tokenize per variable (patch or point).
   * Reshape to `[B * C, N, d]` and apply a shared TransformerEncoder.
3) SSM (mamba_ssm.Mamba)
   * Same reshaping as Transformer.
   * Uses the `mamba_ssm` implementation with residual + RMSNorm blocks.
   * Requires `mamba_ssm` to be installed when `--temporal_encoder ssm` is used.

Notes:
* Use shared weights across variables to reduce parameters.
* If `use_patch`:
  * SSM/Transformer use `PatchEmbedding` (learned projection + positional).
  * TCN applies non-learned patch pooling after the encoder.

### 4.2 DynamicGraph Generator (layers/DynamicGraph.py)

Common interface:

* `forward(z_t, stats=None, exog=None) -> A_t`
  * `z_t` shape `[B, C, d_z]`
  * Output `A_t` shape `[B, C, C]`

Input feature `z_t` options:
* `pool_time`: pooled `H_time` per coarse segment.
* `stats`: per-variable stats from `x_enc` (mean, std, diff).
* `exog`: optional external features.
* `concat`: concatenate multiple sources before projection.

Variants:

1) Low-rank (recommended first)
   * `U_t = f_u(z_t)`, `V_t = f_v(z_t)` -> shapes `[B, C, r]`.
   * `A_t = row_softmax(U_t @ V_t^T + B_prior)`.
2) Top-k sparse
   * `S = f(z_t) @ f(z_t)^T`, keep top-k per row.
   * Apply masked softmax for rows.
3) Dictionary mix
   * Learn `K` prototype graphs `A_k`.
   * Gating: `pi_t = softmax(g(z_t))`, then `A_t = sum_k pi_tk * A_k`.
   * Prototypes can be full or low-rank.
4) Prior + residual
   * `A_t = A_prior + delta_t`, `delta_t` low-rank or sparse.

Normalization:
* `row_softmax` for stochastic adjacency.
* `sym` optional for undirected graphs.

### 4.3 GraphMixing (layers/DynamicGraph.py)

Common interface:

* `forward(A_t, H_seg) -> H_mix_seg`
  * `A_t` shape `[B, C, C]`
  * `H_seg` shape `[B, C, n_seg, d]`
  * Output `H_mix_seg` shape `[B, C, n_seg, d]`

Variants:

1) Linear mixing
   * `H_mix = A_t @ H_seg` (per time token).
   * Add residual: `H_seg + H_mix`.
2) Graph attention
   * Use `A_t` as bias/mask in attention scores.
   * Keep head count small; add dropout.
3) Diffusion
   * Apply k-step propagation: `H_{k+1} = A_t @ H_k`.

### 4.4 Coarse scale graph schedule

Given `graph_scale = P_long` in token units:

* Split `H_time` into segments of length `P_long`.
* For each segment:
  * Pool per variable to `z_t` (avg or max over tokens).
  * Generate `A_t`.
  * Mix all tokens in the segment with `A_t`.
* Broadcast is implicit by applying the same `A_t` to all tokens in the segment.

### 4.5 Anti-spurious regularization

Option A: smooth adjacency
* `L_smooth = sum_t ||A_t - A_{t-1}||_1`

Option B: smooth factors
* `L_smooth = sum_t ||U_t - U_{t-1}||_2`

Return `graph_reg_loss` and add to training loss in Exp classes.

### 4.6 Forecast head (models/DynamicGraphMixer.py)

Option 1: linear head (initial target)
* Flatten `H_mix` over tokens and feature dim.
* Per variable head: `Linear(d * N, pred_len)`.
* Reshape to `[B, pred_len, C]`.

Option 2: seq2seq head
* Not required for first iteration.

## 5. Model forward path (forecast)

1) `H_time = TemporalEncoder(x_enc)` -> `[B, C, N, d]`
2) Segment tokens by `graph_scale`.
3) For each segment:
   * `z_t = pool(H_time[:, :, seg, :])`
   * `A_t = GraphGenerator(z_t, stats, exog)`
   * `H_seg = GraphMixing(A_t, H_time[:, :, seg, :])`
4) Concatenate mixed segments -> `H_mix`.
5) Forecast head -> `Y_hat`.
6) Store `self.graph_reg_loss` if enabled.

## 6. Integration in training

Minimum change (recommended):

* Model sets `self.graph_reg_loss` (float tensor or 0).
* `exp/exp_long_term_forecasting.py` adds:
  * `loss = mse + graph_reg_lambda * model.graph_reg_loss`
  * Only if attribute exists.

## 7. Minimal first implementation (v0)

* Temporal encoder: TCN (shared across variables).
* Graph generator: low-rank.
* Graph mixing: linear with residual.
* Coarse scale: enabled with `graph_scale >= 1`.
* Head: linear.
* No exogenous inputs, no dictionary, no attention mixing.

This gives a stable baseline to compare against existing mixers.

## 8. Debug hooks

* `--return_graph`: if true, cache last `A_t` for inspection.
* `--graph_stats`: log mean degree, entropy, and sparsity.
* Optional `visualize_graph.py` script later for analysis.
