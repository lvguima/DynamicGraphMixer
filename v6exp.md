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
