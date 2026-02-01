# Visualization Plan (8 Figures)

This document summarizes the 8 visualization types we prepared, what each figure means, which current experiments can support it, and how strong the evidence is **with current available results** (ETTm1 pl96, flotation pl4, grinding pl30 from `0201_graphsweep_*`).

## Current Data Availability (as of now)

Completed (usable for plots):
- ETTm1: `pred_len=96` (graph sweep)
- flotation: `pred_len=4` (graph sweep)
- grinding: `pred_len=30` (graph sweep)

Not finished yet:
- Other horizons (ETTm1 192/336, flotation 2/6, grinding 60/90)
- Baseline models across horizons (running)

General note:
- For our method, `graph_logs/<exp_id>/stats.csv` provides epoch/step-level metrics.
- With `--graph_log_artifacts`, we also have artifacts under `graph_logs/<exp_id>/epochXXX_stepYYYYY/A_mix/`:
  - `adj_mean.npy`, `base_adj.npy`, `raw_adj_mean.npy`, `adj_segments_mean.npy`, etc.
  - `per_var_conf.npy`, `per_var_raw_entropy.npy`, `per_var_overlap.npy`, ...
- `results/<setting>/pred.npy` + `true.npy` supports forecast-behavior plots.

## Figure 1: Prior vs A_mix Heatmaps (and Difference)

**Purpose**
- Show that **Prior-as-Base** provides a structural backbone, while the learned dynamic graph (A_mix) makes bounded corrections.

**What to plot**
- Heatmap of `A_mix` (e.g., `adj_mean.npy` or one segment from `adj_segments_mean.npy`)
- Heatmap of `Prior` (`base_adj.npy`)
- Heatmap of `A_mix - Prior` (difference)
- (Optional) `Raw A` (`raw_adj_mean.npy`) to show effect of sparsify/normalize

**Supported by current results**
- Yes. Works well on ETTm1 pl96 and grinding pl30.

**Evidence strength (current)**
- **Strong** for the claim: “A_mix stays close to prior and applies small corrections.”
- Not suitable to claim: “the dynamic graph rewrites the prior drastically.”

**Typical narrative**
- A_mix shares the same dominant edges as the prior; differences are localized and bounded.

## Figure 2: Top-k Neighbor Graph (Network Diagram)

**Purpose**
- Provide an interpretable “edge list” view of the strongest dependencies.
- Show stability across (a) prior vs A_mix, and (b) segment-specific vs mean graph.

**What to plot**
- `top_edges20` directed graph for:
  - A_mix mean
  - Prior mean
  - A_mix seg0 (or seg1)

**Supported by current results**
- Yes. grinding pl30 already produces clean, readable graphs.

**Evidence strength (current)**
- **Medium**: good for interpretability and stability, but alone does not prove performance improvement.

**Typical narrative**
- The strongest dependency skeleton is stable and aligned with the prior; segment-level variation is limited on industrial data.

## Figure 3: Per-variable Confidence Distribution (Violin/Box)

**Purpose**
- Show **heterogeneity** across variables (some nodes are more confident / more uncertain).
- Show training-time stabilization via “first vs last” distribution change.

**What to plot**
- Violin plots of `per_var_conf.npy` (or `per_var_raw_conf.npy`) at:
  - first logged artifact
  - last logged artifact

**Supported by current results**
- Yes. grinding pl30 provides a clear distribution and change.

**Evidence strength (current)**
- **Strong** as a mechanism evidence of stabilization / regularization.

## Figure 4: Horizon-segment Error (S1/S2/S3)

**Purpose**
- Show error growth from short → mid → long horizons and whether degradation saturates.

**What to plot**
- Segment-averaged MAE/MSE over the prediction horizon:
  - S1: first third
  - S2: middle third
  - S3: last third

**Supported by current results**
- Yes, as a single-model descriptive plot (ETTm1 pl96 already).

**Evidence strength (current)**
- **Medium** without baselines/ablations.
- Becomes **Strong** when plotted as “Baseline vs Ours” grouped bars for each segment.

## Figure 5: Error vs Mechanism Metric (Scatter)

**Purpose**
- Demonstrate correlation between a mechanism metric and predictive error.

**What to plot**
- Scatter of `MSE` (y) vs a metric like `dyn_vs_prior_l1` or `conf_mean` (x).

**Supported by current results**
- Technically yes, but only a few points.

**Evidence strength (current)**
- **Weak** with only 3–4 cross-dataset points (scale mismatch and low sample size).
  - Recommended upgrade: within-dataset scatter (multiple settings) or normalized error.

## Figure 6: Per-variable Error Heatmap

**Purpose**
- Show which variables are harder and whether improvements target difficult variables.

**What to plot**
- Per-variable MAE/MSE heatmap from `pred.npy/true.npy`.

**Supported by current results**
- Yes, as a single-method descriptive plot (ETTm1 pl96 already).

**Evidence strength (current)**
- **Medium** without baselines.
- Becomes **Strong** when adding a delta heatmap: `MAE_baseline - MAE_ours`.

## Figure 7: Case Study Prediction Curves (GT vs Pred)

**Purpose**
- Visualize qualitative behavior: trend tracking, turning points, amplitude under/over-shoot.

**What to plot**
- A few selected (sample, variable) curves:
  - `GT` vs `Pred`
  - (Recommended) `GT` vs `Baseline Pred` vs `Ours Pred`

**Supported by current results**
- Yes for single method (ETTm1 pl96 already).

**Evidence strength (current)**
- **Medium** without baseline overlay; best used as qualitative illustration.

## Figure 8: Dual-Stream Energy Stackplot (E_trend vs E_season)

**Purpose**
- Show Dual-Stream decomposition allocation during training (trend vs season energy).

**What to plot**
- Stackplot over epochs: `E_trend` and `E_season` from `stats.csv`.

**Supported by current results**
- Yes (ETTm1 pl96 / grinding pl30 / flotation pl4 can be used).

**Evidence strength (current)**
- **Medium–Strong** depending on how much variation exists; stronger when the dataset shows clear energy reallocation over epochs.

## Summary: What We Can Reliably Claim Now

With the currently completed three runs:
- **Strong mechanism evidence**:
  - per-variable confidence/entropy distribution shifts (first vs last)
  - alignment between A_mix and prior (prior-as-base)
- **Good interpretability evidence**:
  - top-edges network diagrams (prior vs A_mix; mean vs segment)
- **Descriptive forecast behavior**:
  - horizon-segment errors
  - per-variable error heatmaps
  - case prediction curves

After baselines finish (and after other horizons finish), we should prioritize “paired” comparisons:
- Horizon-segment error: Baseline vs Ours
- Per-variable error: Baseline vs Ours and delta heatmaps
- Case study: overlay Baseline vs Ours on same sample/variable

