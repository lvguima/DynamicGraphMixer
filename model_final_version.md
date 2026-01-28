# Final Model Structure (Fixed) — DynamicGraphMixer v3.1 (B5)

> This is the **single fixed** final architecture recommended based on the latest B0–B5 ablation summary (no-traffic).  
> It corresponds to **B5: Dual-Stream + SMGP + Graph propagation**.

## 1. Task & Notation

- Input multivariate sequence:  \(X \in \mathbb{R}^{B\times L\times C}\)
- Output forecast:  \(\hat{Y}\in\mathbb{R}^{B\times H\times C_{out}}\)

Common setting used in ablations:
- \(L=96\), \(H=96\), multivariate→multivariate.

## 2. Overall Pipeline

1. **(Optional in code, fixed here) Normalization**: apply the existing `use_norm=1` preprocessing.
2. **Dual-Stream EMA decomposition** of the raw input into trend & seasonal residual.
3. **Trend stream**: shared linear head (DLinear-style) predicts future trend.
4. **Season stream**: TemporalEncoder (TCN) + SMGP + Dynamic Graph propagation + ForecastHead predicts residual.
5. **Sum**: final prediction is the sum of trend and seasonal predictions.

## 3. Dual-Stream Decomposition (Fixed)

EMA trend (per variable, along time axis):
$$
x^{\text{trend}}_{t} = \alpha x_{t} + (1-\alpha)x^{\text{trend}}_{t-1}, \quad \alpha=0.1
$$

Seasonal / residual component:
$$
x^{\text{season}} = x - x^{\text{trend}}
$$

## 4. Trend Stream (Fixed)

- `trend_head = linear`
- `trend_head_share = 1` (shared across variables)

For each variable \(c\):
$$
\hat{y}^{\text{trend}}_{:,c} = W_{\text{trend}}\, x^{\text{trend}}_{:,c} + b_{\text{trend}}, 
\quad W_{\text{trend}}\in\mathbb{R}^{H\times L}
$$

## 5. Season Stream (Fixed)

### 5.1 Temporal encoder (TCN)

- `temporal_encoder = tcn`
- kernel=3, dilation_base=2
- d_model=128, e_layers=2, d_ff=256

Produces token-wise hidden states:
$$
H \in \mathbb{R}^{B\times C\times N\times d},\quad N=L\ (\text{no patch})
$$

### 5.2 SMGP (Stationary-Map Graph Propagation)

Map–Value decoupling:
- \(H_{val} = H\) (used for propagation)
- \(H_{map} = \text{Stationarize}(H)\) (used only to learn graph)

Fixed stationarization in B5:
- `graph_map_norm = ma_detrend`, `graph_map_window = 16`

$$
H_{map} = H - \text{MA}_{w=16}(H)
$$

### 5.3 Dynamic graph learner (Low-rank)

Per segment (segment length `graph_scale=8`):
- average tokens inside segment to obtain node features
$$
Z_k = \text{Mean}_{n\in seg(k)}(H_{map,:,n,:})\in\mathbb{R}^{B\times C\times d}
$$

Low-rank adjacency (rank=8), row-softmax:
$$
U=Z_kW_u,\ V=Z_kW_v,\
A_k=\text{Softmax}_{row}\left(\frac{1}{\sqrt{r}}UV^\top\right)
$$

### 5.4 Base-graph mixing + sparsification (Fixed)

- base graph mixing **enabled** (`graph_base_mode=mix`)
- adjacency sparsification: `adj_sparsify=topk`, `adj_topk=6`

### 5.5 Graph propagation (GraphMixer)

Residual propagation:
$$
H_{out} = H_{val} + g \odot (A_k \cdot H_{val})
$$

- gate: `gate_mode=per_var`, `gate_init=-6` (weak but non-zero propagation)

### 5.6 Forecast head

Flatten token dimension and project to \(H\):
$$
\hat{Y}^{\text{season}} = \text{Head}(H_{out}) \in \mathbb{R}^{B\times H\times C_{out}}
$$

## 6. Final Output (Fixed)

$$
\hat{Y} = \hat{Y}^{\text{trend}} + \hat{Y}^{\text{season}}
$$

## 7. Fixed Hyperparameter Summary (B5)

- `decomp_mode=ema`, `decomp_alpha=0.1`
- `trend_head=linear`, `trend_head_share=1`
- `graph_map_norm=ma_detrend`, `graph_map_window=16`
- `gate_mode=per_var`, `gate_init=-6`
- `graph_rank=8`, `graph_scale=8`, `adj_topk=6`
- temporal encoder: `tcn(kernel=3, dilation=2)`

