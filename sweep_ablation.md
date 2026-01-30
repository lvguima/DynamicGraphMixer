# Sweep Ablation: Graph-Structure Hyperparameters

This report summarizes the graph-structure hyperparameter sweep results based on `graph_logs/*/stats.csv`.

## Sweep Setup

- Base configuration: B5 (Dual-Stream + SMGP + graph propagation with base mix + gate).
- One-factor-at-a-time (OFAT) sweep around the base values.
- Base graph settings: graph_rank=8, graph_scale=8, adj_topk=6, graph_base_alpha_init=-8, graph_smooth_lambda=0.0
- Sweep values:
  - graph_rank: 4, 16
  - graph_scale: 4, 16
  - adj_topk: 4, 8
  - graph_base_alpha_init: -10, -6
  - graph_smooth_lambda: 0.05, 0.1

## ETTh1

- Base (GBASE): MSE=0.385156, MAE=0.399463

| Parameter | Value | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---:|---:|---:|---:|---:|
| graph_rank | 4 | 0.384568 | 0.399685 | -0.000588 | +0.000222 |
| graph_rank | 16 | 0.385801 | 0.400297 | +0.000645 | +0.000834 |
| graph_scale | 4 | 0.385045 | 0.399452 | -0.000112 | -0.000011 |
| graph_scale | 16 | 0.385223 | 0.399434 | +0.000066 | -0.000029 |
| adj_topk | 4 | 0.385205 | 0.399516 | +0.000049 | +0.000053 |
| adj_topk | 8 | 0.385176 | 0.399456 | +0.000019 | -0.000007 |
| graph_base_alpha_init | -10 | 0.385185 | 0.399487 | +0.000029 | +0.000025 |
| graph_base_alpha_init | -6 | 0.385173 | 0.399469 | +0.000017 | +0.000006 |
| graph_smooth_lambda | 0.05 | 0.385196 | 0.399488 | +0.000039 | +0.000025 |
| graph_smooth_lambda | 0.1 | 0.385123 | 0.399439 | -0.000034 | -0.000023 |

## ETTh2

- Base (GBASE): MSE=0.327903, MAE=0.382421

| Parameter | Value | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---:|---:|---:|---:|---:|
| graph_rank | 4 | 0.333148 | 0.387142 | +0.005245 | +0.004722 |
| graph_rank | 16 | 0.334958 | 0.387403 | +0.007056 | +0.004983 |
| graph_scale | 4 | 0.327863 | 0.382429 | -0.000039 | +0.000008 |
| graph_scale | 16 | 0.327525 | 0.382149 | -0.000378 | -0.000271 |
| adj_topk | 4 | 0.327911 | 0.382440 | +0.000008 | +0.000019 |
| adj_topk | 8 | 0.327997 | 0.382505 | +0.000094 | +0.000085 |
| graph_base_alpha_init | -10 | 0.327894 | 0.382396 | -0.000008 | -0.000025 |
| graph_base_alpha_init | -6 | 0.327947 | 0.382439 | +0.000045 | +0.000019 |
| graph_smooth_lambda | 0.05 | 0.327844 | 0.382422 | -0.000059 | +0.000001 |
| graph_smooth_lambda | 0.1 | 0.327895 | 0.382407 | -0.000008 | -0.000014 |

## ETTm1

- Base (GBASE): MSE=0.317662, MAE=0.357933

| Parameter | Value | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---:|---:|---:|---:|---:|
| graph_rank | 4 | 0.318886 | 0.358371 | +0.001225 | +0.000438 |
| graph_rank | 16 | 0.318736 | 0.359076 | +0.001074 | +0.001144 |
| graph_scale | 4 | 0.316655 | 0.356732 | -0.001006 | -0.001201 |
| graph_scale | 16 | 0.317371 | 0.357479 | -0.000291 | -0.000454 |
| adj_topk | 4 | 0.317185 | 0.357522 | -0.000477 | -0.000411 |
| adj_topk | 8 | 0.316997 | 0.357430 | -0.000665 | -0.000502 |
| graph_base_alpha_init | -10 | 0.317097 | 0.357498 | -0.000565 | -0.000434 |
| graph_base_alpha_init | -6 | 0.317316 | 0.357740 | -0.000346 | -0.000193 |
| graph_smooth_lambda | 0.05 | 0.317478 | 0.357895 | -0.000183 | -0.000038 |
| graph_smooth_lambda | 0.1 | 0.317095 | 0.357425 | -0.000566 | -0.000507 |

## ETTm2

- Base (GBASE): MSE=0.187905, MAE=0.282247

| Parameter | Value | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---:|---:|---:|---:|---:|
| graph_rank | 4 | 0.189250 | 0.284497 | +0.001345 | +0.002250 |
| graph_rank | 16 | 0.186239 | 0.280505 | -0.001667 | -0.001742 |
| graph_scale | 4 | 0.188066 | 0.282276 | +0.000161 | +0.000029 |
| graph_scale | 16 | 0.186854 | 0.281258 | -0.001052 | -0.000989 |
| adj_topk | 4 | 0.187679 | 0.282005 | -0.000226 | -0.000243 |
| adj_topk | 8 | 0.187730 | 0.282117 | -0.000176 | -0.000131 |
| graph_base_alpha_init | -10 | 0.187768 | 0.282101 | -0.000137 | -0.000147 |
| graph_base_alpha_init | -6 | 0.187667 | 0.282117 | -0.000238 | -0.000131 |
| graph_smooth_lambda | 0.05 | 0.187803 | 0.282105 | -0.000102 | -0.000142 |
| graph_smooth_lambda | 0.1 | 0.187882 | 0.282230 | -0.000023 | -0.000018 |

## flotation

- Base (GBASE): MSE=0.786082, MAE=0.622778

| Parameter | Value | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---:|---:|---:|---:|---:|
| graph_rank | 4 | 0.787303 | 0.623409 | +0.001221 | +0.000632 |
| graph_rank | 16 | 0.791967 | 0.624372 | +0.005885 | +0.001594 |
| graph_scale | 4 | 0.786214 | 0.622754 | +0.000132 | -0.000024 |
| graph_scale | 16 | 0.785741 | 0.622585 | -0.000342 | -0.000193 |
| adj_topk | 4 | 0.786140 | 0.622786 | +0.000058 | +0.000008 |
| adj_topk | 8 | 0.786149 | 0.622795 | +0.000067 | +0.000018 |
| graph_base_alpha_init | -10 | 0.786071 | 0.622763 | -0.000012 | -0.000015 |
| graph_base_alpha_init | -6 | 0.786051 | 0.622763 | -0.000031 | -0.000015 |
| graph_smooth_lambda | 0.05 | 0.786281 | 0.622845 | +0.000199 | +0.000067 |
| graph_smooth_lambda | 0.1 | 0.786296 | 0.622854 | +0.000214 | +0.000076 |

## grinding

- Base (GBASE): MSE=4.239915, MAE=0.698634

| Parameter | Value | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---:|---:|---:|---:|---:|
| graph_rank | 4 | 4.322254 | 0.719141 | +0.082339 | +0.020507 |
| graph_rank | 16 | 4.245402 | 0.699162 | +0.005486 | +0.000528 |
| graph_scale | 4 | 4.241079 | 0.698965 | +0.001163 | +0.000331 |
| graph_scale | 16 | 4.236457 | 0.698208 | -0.003458 | -0.000425 |
| adj_topk | 4 | 4.239775 | 0.698667 | -0.000140 | +0.000033 |
| adj_topk | 8 | 4.239473 | 0.698567 | -0.000443 | -0.000067 |
| graph_base_alpha_init | -10 | 4.239541 | 0.698629 | -0.000374 | -0.000004 |
| graph_base_alpha_init | -6 | 4.239414 | 0.698564 | -0.000502 | -0.000070 |
| graph_smooth_lambda | 0.05 | 4.240859 | 0.698838 | +0.000943 | +0.000204 |
| graph_smooth_lambda | 0.1 | 4.240347 | 0.698727 | +0.000432 | +0.000093 |

## weather

- Base (GBASE): MSE=0.172699, MAE=0.229743

| Parameter | Value | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---:|---:|---:|---:|---:|
| graph_rank | 4 | 0.172454 | 0.230285 | -0.000245 | +0.000542 |
| graph_rank | 16 | 0.173285 | 0.231764 | +0.000586 | +0.002021 |
| graph_scale | 4 | 0.172925 | 0.230271 | +0.000226 | +0.000528 |
| graph_scale | 16 | 0.172935 | 0.230802 | +0.000237 | +0.001059 |
| adj_topk | 4 | 0.172600 | 0.229611 | -0.000098 | -0.000132 |
| adj_topk | 8 | 0.172695 | 0.229779 | -0.000003 | +0.000036 |
| graph_base_alpha_init | -10 | 0.172638 | 0.229632 | -0.000061 | -0.000111 |
| graph_base_alpha_init | -6 | 0.172654 | 0.229678 | -0.000045 | -0.000065 |
| graph_smooth_lambda | 0.05 | 0.172681 | 0.229666 | -0.000018 | -0.000076 |
| graph_smooth_lambda | 0.1 | 0.172638 | 0.229629 | -0.000060 | -0.000114 |

## Best MSE Per Dataset

| Dataset | Best Setting | MSE | MAE | dMSE vs Base | dMAE vs Base |
|---|---|---:|---:|---:|---:|
| ETTh1 | graph_rank=4 | 0.384568 | 0.399685 | -0.000588 | +0.000222 |
| ETTh2 | graph_scale=16 | 0.327525 | 0.382149 | -0.000378 | -0.000271 |
| ETTm1 | graph_scale=4 | 0.316655 | 0.356732 | -0.001006 | -0.001201 |
| ETTm2 | graph_rank=16 | 0.186239 | 0.280505 | -0.001667 | -0.001742 |
| flotation | graph_scale=16 | 0.785741 | 0.622585 | -0.000342 | -0.000193 |
| grinding | graph_scale=16 | 4.236457 | 0.698208 | -0.003458 | -0.000425 |
| weather | graph_rank=4 | 0.172454 | 0.230285 | -0.000245 | +0.000542 |

## Best Graph Settings Per Dataset (OFAT Best)

These are the **best single-parameter changes** from the sweep. All unspecified values
remain at the base settings: `graph_rank=8`, `graph_scale=8`, `adj_topk=6`,
`graph_base_alpha_init=-8`, `graph_smooth_lambda=0.0`.

| Dataset | graph_rank | graph_scale | adj_topk | graph_base_alpha_init | graph_smooth_lambda |
|---|---:|---:|---:|---:|---:|
| ETTh1 | **4** | 8 | 6 | -8 | 0.0 |
| ETTh2 | 8 | **16** | 6 | -8 | 0.0 |
| ETTm1 | 8 | **4** | 6 | -8 | 0.0 |
| ETTm2 | **16** | 8 | 6 | -8 | 0.0 |
| flotation | 8 | **16** | 6 | -8 | 0.0 |
| grinding | 8 | **16** | 6 | -8 | 0.0 |
| weather | **4** | 8 | 6 | -8 | 0.0 |

Note: because this is OFAT, these are not guaranteed to be globally optimal when
parameters are changed jointly.

## Summary Analysis

Key observations across datasets (based on MSE/MAE deltas vs base):

- **Overall sensitivity is low**: most changes are within ~1e-3 MSE, indicating the B5 base is already near a local optimum for graph settings.
- **graph_scale** shows the most consistent upside: `graph_scale=16` is best on 3/7 datasets (ETTh2, flotation, grinding) and has the best average MSE delta overall (slightly negative). `graph_scale=4` is best on ETTm1 but otherwise near-neutral.
- **graph_rank=4 is risky**: it hurts ETTh2/ETTm1 and is much worse on grinding (+0.082 MSE). It only wins ETTh1 and weather with very small margins.
- **graph_rank=16 helps ETTm2** (largest single improvement in this sweep: -0.001667 MSE) but is slightly worse elsewhere.
- **adj_topk, graph_base_alpha_init, graph_smooth_lambda** are **very low impact** under current settings; differences are tiny and inconsistent.

If you want a single, low-risk tweak across datasets, `graph_scale=16` is the most plausible candidate. Otherwise, keeping the base settings is reasonable.
