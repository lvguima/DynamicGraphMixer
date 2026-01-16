---
title: Dynamic Graph Mixer Forecaster - Implementation Notes
status: baseline
scope: code-level design + experiment plan
---

# Dynamic Graph Mixer Forecaster - Implementation Notes

This document reflects the current code implementation as the baseline.
Future experiments will extend this baseline with alternative components.
The goal is to keep module boundaries stable while exploring variants.

## 0. Scope and inputs (current behavior)

* Primary target tasks: long_term_forecast and short_term_forecast.
* Input pipeline follows the default forecasting setup:
  * Raw CSV is read by the dataset loader.
  * If `--use_norm=1` (default), a `StandardScaler` is fit on the train split and
    applied to all splits; `x_enc`/`x_dec` fed into the model are normalized.
  * If `--use_norm=0`, raw values are used.
  * `--inverse` only affects evaluation/visualization; the model itself never
    denormalizes outputs.
* All shapes are batch-first.
* `--use_patch` changes tokenization:
  * TCN: non-learned patch pooling after the encoder.
  * Transformer/SSM: learned patch embedding before the encoder.

## 1. Current file layout (implemented)

* `models/DynamicGraphMixer.py`
  * Model entry point, task routing, forecast head.
* `layers/DynamicGraph.py`
  * Low-rank graph generator and linear mixing.
* `layers/TemporalEncoder.py`
  * TCN, Transformer, and SSM temporal encoders.
* `models/__init__.py`, `exp/exp_basic.py`
  * Model registration.
* `exp/exp_long_term_forecasting.py`, `exp/exp_short_term_forecasting.py`
  * Training integration with graph regularization.
* `run.py`
  * CLI args for DynamicGraphMixer.

## 2. Shape conventions (current)

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

## 3. Implemented configuration surface (current)

### 3.1 Input and patching

* `--use_norm` (0 or 1)
* `--use_patch`
* `--patch_len`, `--patch_stride`

### 3.2 Temporal encoder

* `--temporal_encoder`: `tcn | transformer | ssm`
* Shared encoder params:
  * `--d_model`, `--e_layers`, `--d_ff`, `--dropout`
* Transformer-specific:
  * `--n_heads`, `--activation`, `--factor`
* TCN-specific:
  * `--tcn_kernel`, `--tcn_dilation`
* SSM-specific:
  * `--ssm_state_dim`, `--d_conv`, `--expand`
  * SSM requires `mamba_ssm` when `--temporal_encoder ssm` is used.

### 3.3 Graph and regularization

* `--graph_scale`: coarse segment length (token units).
* `--graph_rank`: low-rank size.
* `--graph_smooth_lambda`: smoothness weight for adjacency differences.

## 4. Baseline modules (current code)

### 4.1 TemporalEncoder

* TCN: shared across variables; outputs `[B, C, L, d]`.
* Transformer: uses `EncoderLayer + FullAttention` from existing layers.
* SSM: uses `mamba_ssm.Mamba` blocks with residual + RMSNorm.

### 4.2 Patch handling

* If `use_patch`:
  * Transformer/SSM use `PatchEmbedding` (learned projection + positional).
  * TCN applies non-learned patch pooling after the TCN output.
* If not `use_patch`: point-level tokens are used.

### 4.3 Dynamic graph generator

* Low-rank generator:
  * `U_t = f_u(z_t)`, `V_t = f_v(z_t)`
  * `A_t = softmax(U_t @ V_t^T)` (row-wise)

### 4.4 Graph mixing

* Linear residual mixing:
  * `H_mix = H_seg + A_t @ H_seg`

### 4.5 Coarse graph schedule

* Split tokens by `graph_scale`.
* For each segment:
  * Pool tokens to `z_t`.
  * Generate `A_t`.
  * Mix segment tokens with `A_t`.

### 4.6 Regularization

* Smooth adjacency:
  * `L_smooth = sum_t ||A_t - A_{t-1}||_1`
* Added to loss in long/short term forecasting exp code.

### 4.7 Forecast head

* Linear head:
  * Flatten `[B, C, N, d]` -> `[B, C, N*d]`
  * Per-variable linear to `pred_len`.

## 5. Forward path (forecast)

1) `H_time = TemporalEncoder(x_enc)` -> `[B, C, N, d]`
2) If TCN and `use_patch`, apply patch pooling.
3) Segment tokens by `graph_scale`.
4) For each segment:
   * `z_t = pool(H_time[:, :, seg, :])`
   * `A_t = GraphGenerator(z_t)`
   * `H_seg = GraphMixing(A_t, H_time[:, :, seg, :])`
5) Concatenate mixed segments -> `H_mix`.
6) Forecast head -> `Y_hat`.
7) Store `graph_reg_loss` if enabled.

## 6. Training integration (current)

* Model exposes `graph_reg_loss`.
* `exp_long_term_forecasting.py` and `exp_short_term_forecasting.py` add:
  * `loss = mse + graph_smooth_lambda * graph_reg_loss`

## 7. Baseline configurations (current)

* Baseline A (fast):
  * `temporal_encoder=tcn`, low-rank graph, linear mixing, linear head.
* Baseline B (learned patch):
  * `temporal_encoder=transformer`, `use_patch=1`, low-rank graph, linear mixing.
* SSM baseline:
  * `temporal_encoder=ssm` requires `mamba_ssm` and is optional.

## 8. Planned experiments (detailed roadmap)

We will extend the current baseline by swapping one component at a time.
Each phase should keep all other parts fixed unless noted.

### Phase 1: Temporal encoder and patching

* Compare `{tcn, transformer, ssm}` under matched `d_model`, `e_layers`, `d_ff`.
* Patch ablations:
  * `patch_len` in `{8, 16, 32}`
  * `patch_stride` in `{4, 8, 16}`
* Measure:
  * MSE/MAE on ETTm1 and ETTm2
  * Training speed (sec/iter) and peak GPU memory

### Phase 2: Graph generator variants

Planned additions:
* `graph_mode`: `topk`, `dict`, `prior_residual`
* `graph_input`: `pool_time`, `stats`, `concat`
* `graph_normalize`: `row_softmax`, `row_sum`, `sym`
* `graph_directed`: toggle directed vs symmetric

Evaluation:
* Hold temporal encoder fixed (best from Phase 1).
* Sweep `graph_rank`/`graph_topk` with constant `graph_scale`.

### Phase 3: Graph mixing variants

Planned additions:
* Attention mixing with `A_t` as bias/mask.
* Diffusion mixing with configurable steps.

Evaluation:
* Compare accuracy vs throughput.
* Check stability with larger `graph_scale`.

### Phase 4: Graph scale and regularization

Planned additions:
* `graph_pooling`: `avg | max | last`
* `graph_smooth_target`: `A | U` (adjacency or factors)

Evaluation:
* Sweep `graph_scale` in `{4, 8, 16, 32}`
* Test smoothness weight in `{0, 0.01, 0.05, 0.1}`

### Phase 5: Output head variants

Planned additions:
* `head_type=seq2seq`
* Compare to linear head under fixed encoder/mixer.

### Phase 6: Reporting and diagnostics

Planned additions:
* `--return_graph` and `--graph_stats` for logging
* Record graph sparsity, entropy, and mean degree per epoch
* Add a small script to visualize adjacency snapshots
