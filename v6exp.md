# Step0 Results (B5 Baseline)

Source: `graph_logs/*_BASE/stats.csv` (last row).
Metrics: `mse`, `mae`, `gate_mean`, `entropy_mean` (A_entropy), `topk_overlap` (A_overlap),
`l1_adj_diff` (adj_diff), `alpha_mean` (base_alpha), `map_mean_abs`, `map_std_mean`,
`E_trend`, `E_season`, `E_ratio`.
Note: `flotation_BASE/stats.csv` only contains `mse/mae`, other fields are N/A.

## ETTm1 (BASE)

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.317224 | 0.357522 | 0.002445 | 1.790191 | 0.859253 | 0.038090 | 0.000335 | 0.082575 | 0.105627 | 0.916860 | 0.126411 | 0.878832 |

## weather (BASE)

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.172845 | 0.230119 | 0.002500 | 1.791757 | 0.359294 | 0.058360 | 0.000336 | 0.050809 | 0.073401 | 0.754258 | 0.106719 | 0.876049 |

## flotation (BASE)

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.786068 | 0.622770 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## grinding (BASE)

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.249201 | 0.699097 | 0.002481 | 1.787538 | 0.515270 | 0.071912 | 0.000335 | 0.046374 | 0.066474 | 1.099435 | 0.074838 | 0.936269 |

# Step1 Results (graph_scale / gate_init)

Source: `graph_logs/*_S1_*/stats.csv` (last row).
Note: `flotation_*` only logged `mse/mae`, other fields are N/A.

## F1_S16_Gm6 (graph_scale=16, gate_init=-6)

### ETTm1

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.316530 | 0.356723 | 0.002453 | 1.791759 | 0.855357 | 0.035337 | 0.000336 | 0.082435 | 0.105929 | 0.916860 | 0.126411 | 0.878832 |

### weather

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.172772 | 0.229822 | 0.002495 | 1.791759 | 0.348958 | 0.059210 | 0.000335 | 0.052594 | 0.075654 | 0.764476 | 0.127317 | 0.857235 |

### flotation

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.785766 | 0.622600 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### grinding

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.244832 | 0.698420 | 0.002478 | 1.791699 | 0.515625 | 0.071319 | 0.000335 | 0.046258 | 0.066451 | 1.099435 | 0.074838 | 0.936269 |

## F2_S16_Gm2 (graph_scale=16, gate_init=-2)

### ETTm1

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.317481 | 0.358480 | 0.114252 | 1.791759 | 0.859018 | 0.034154 | 0.000384 | 0.082606 | 0.106411 | 0.916860 | 0.126411 | 0.878832 |

### weather

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.172712 | 0.230852 | 0.114162 | 1.791760 | 0.403661 | 0.054418 | 0.000354 | 0.051540 | 0.075022 | 0.754258 | 0.106719 | 0.876049 |

### flotation

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.777083 | 0.619054 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### grinding

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.215553 | 0.697153 | 0.118124 | 1.791717 | 0.522396 | 0.070666 | 0.000341 | 0.047238 | 0.068159 | 1.099435 | 0.074838 | 0.936269 |

## F1b_S4_Gm6 (graph_scale=4, gate_init=-6) [ETTm1 only]

### ETTm1

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.317395 | 0.357455 | 0.002489 | 1.736746 | 0.842236 | 0.060406 | 0.000325 | 0.082393 | 0.105717 | 0.916860 | 0.126411 | 0.878832 |
