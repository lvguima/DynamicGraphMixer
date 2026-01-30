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

# Step2 Results (warmup / confidence gate / routing)

Source: `graph_logs/*_S2_*/stats.csv` (last row).
Note: `flotation_*` only logged `mse/mae`, other fields are N/A.

## C1_WARMUP3 (graph_scale=16, gate_init=-6, warmup=3)

### ETTm1

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.316854 | 0.357232 | 0.002484 | 1.791759 | 0.858572 | 0.034220 | 0.000335 | 0.082738 | 0.106304 | 0.916860 | 0.126411 | 0.878832 |

### weather

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.172729 | 0.229455 | 0.002476 | 1.791760 | 0.337708 | 0.060174 | 0.000335 | 0.052608 | 0.075957 | 0.764476 | 0.127317 | 0.857235 |

### flotation

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.785733 | 0.622602 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### grinding

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.246447 | 0.698840 | 0.002475 | 1.791753 | 0.519896 | 0.072066 | 0.000335 | 0.046157 | 0.066317 | 1.099435 | 0.074838 | 0.936269 |

## C2_CONF_GATE (conf_per_var + affine, warmup=3)

### ETTm1

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.317319 | 0.357582 | 0.018039 | 1.791759 | 0.859643 | 0.033938 | 0.000336 | 0.082316 | 0.105800 | 0.916860 | 0.126411 | 0.878832 |

### weather

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.173031 | 0.231278 | 0.017985 | 1.791760 | 0.340982 | 0.059936 | 0.000336 | 0.050948 | 0.073919 | 0.754258 | 0.106719 | 0.876049 |

### flotation

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.784539 | 0.622052 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### grinding

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.243555 | 0.699016 | 0.017997 | 1.791753 | 0.519375 | 0.072031 | 0.000335 | 0.046145 | 0.066272 | 1.099435 | 0.074838 | 0.936269 |

## C3_CONF_GATE_ROUTE (conf_per_var + routing, warmup=3)

### ETTm1

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.316909 | 0.357374 | 0.018111 | 1.791688 | 0.937679 | 0.017511 | 0.527527 | 0.082461 | 0.105962 | 0.916860 | 0.126411 | 0.878832 |

### weather

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.172886 | 0.231011 | 0.018219 | 1.791692 | 0.842024 | 0.018383 | 0.501548 | 0.050974 | 0.073918 | 0.754258 | 0.106719 | 0.876049 |

### flotation

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.784256 | 0.621921 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### grinding

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.242198 | 0.698710 | 0.018040 | 1.791752 | 0.936927 | 0.013072 | 0.504151 | 0.046145 | 0.066269 | 1.099435 | 0.074838 | 0.936269 |

## C4_CONF_ROUTE_L2 (routing + base_reg l2_to_identity)

### ETTm1

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.317112 | 0.357362 | 0.018145 | 1.789999 | 0.936071 | 0.014643 | 0.493920 | 0.082153 | 0.105575 | 0.916860 | 0.126411 | 0.878832 |

### weather

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.172523 | 0.229049 | 0.018172 | 1.790935 | 0.852381 | 0.011343 | 0.501050 | 0.052521 | 0.075595 | 0.764476 | 0.127317 | 0.857235 |

### flotation

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.784448 | 0.621991 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### grinding

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.243124 | 0.698810 | 0.018032 | 1.791526 | 0.926875 | 0.010037 | 0.494308 | 0.046225 | 0.066428 | 1.099435 | 0.074838 | 0.936269 |

# Step3 Results (trend graph propagation)

Source: `graph_logs/*_S3_*/stats.csv` (last row).
Note: `flotation_*` only logged `mse/mae`, other fields are N/A.

## D0_BASE_S16_Gm6 (baseline, no trend graph)

### ETTm1

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.317161 | 0.357476 | 0.002453 | 1.791759 | 0.856964 | 0.035272 | 0.000336 | 0.082472 | 0.105959 | 0.916860 | 0.126411 | 0.878832 |

### weather

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.173183 | 0.231471 | 0.002496 | 1.791759 | 0.350863 | 0.058926 | 0.000335 | 0.050858 | 0.073454 | 0.754258 | 0.106719 | 0.876049 |

### flotation

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.785771 | 0.622599 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### grinding

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.244329 | 0.698435 | 0.002478 | 1.791700 | 0.515417 | 0.071302 | 0.000335 | 0.046247 | 0.066432 | 1.099435 | 0.074838 | 0.936269 |

## D1_TREND_GRAPH (trend graph on)

### ETTm1

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.318358 | 0.357971 | 0.002556 | 1.791749 | 0.864732 | 0.032775 | 0.000316 | 0.081683 | 0.104624 | 0.916860 | 0.126411 | 0.878832 |

### weather

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.173176 | 0.231238 | 0.002551 | 1.791760 | 0.344375 | 0.059726 | 0.000329 | 0.050877 | 0.073316 | 0.754258 | 0.106719 | 0.876049 |

### flotation

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.785935 | 0.622577 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### grinding

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.280162 | 0.724803 | 0.002523 | 1.791697 | 0.518958 | 0.071886 | 0.000329 | 0.046357 | 0.066455 | 1.099435 | 0.074838 | 0.936269 |

## D2_TREND_SMGP (trend graph + SMGP)

### ETTm1

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.318202 | 0.357875 | 0.002522 | 1.791093 | 0.867411 | 0.033738 | 0.000336 | 0.081520 | 0.104396 | 0.916860 | 0.126411 | 0.878832 |

### weather

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.173224 | 0.231313 | 0.002574 | 1.791760 | 0.349851 | 0.059263 | 0.000336 | 0.050844 | 0.073427 | 0.754258 | 0.106719 | 0.876049 |

### flotation

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.785818 | 0.622533 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### grinding

| mse | mae | gate_mean | entropy_mean | topk_overlap | l1_adj_diff | alpha_mean | map_mean_abs | map_std_mean | E_trend | E_season | E_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.246465 | 0.702030 | 0.002490 | 1.791658 | 0.516354 | 0.071518 | 0.000335 | 0.046319 | 0.066364 | 1.099435 | 0.074838 | 0.936269 |
