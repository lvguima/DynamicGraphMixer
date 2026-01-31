# v8 Experiments

## Step1 Prior Graph (topk=6)
Graph params (shared across methods):
| Param | Value |
| --- | --- |
| graph_rank | 8 |
| graph_scale | 16 |
| adj_sparsify / adj_topk | topk / 6 |
| graph_base_mode | mix |
| base_graph_type | prior |
| prior_graph_method | pearson_abs / spearman_abs / mi |
| prior_graph_topk | 6 |
| graph_base_alpha_init | -8 |
| graph_base_l1 | 0.001 |
| gate_mode / gate_init | per_var / -6 |
| graph_map_norm / graph_map_window | ma_detrend / 16 |
| graph_mixer_type | baseline |

| Method | ETTm1 (MSE/MAE) | weather (MSE/MAE) | flotation (MSE/MAE) | grinding (MSE/MAE) |
| --- | --- | --- | --- | --- |
| pearson_abs | 0.316674/0.357238 | 0.172743/0.229681 | 0.785710/0.622562 | 4.244828/0.698469 |
| spearman_abs | 0.317190/0.357456 | 0.173056/0.231420 | 0.785732/0.622572 | 4.245224/0.698573 |
| mi | 0.316670/0.357124 | 0.172732/0.229718 | 0.785694/0.622555 | 4.245790/0.698566 |

Graph stats (from `stats.csv` last row; NA means not logged in that run):

### Prior Sparsity
| Method | ETTm1 | weather | flotation | grinding |
| --- | --- | --- | --- | --- |
| pearson_abs | 0.857143 | 0.285714 | NA | 0.500000 |
| spearman_abs | 0.857143 | 0.285714 | NA | 0.500000 |
| mi | 0.857143 | 0.285714 | NA | 0.500000 |

### Dyn/Prior Overlap
| Method | ETTm1 | weather | flotation | grinding |
| --- | --- | --- | --- | --- |
| pearson_abs | 0.914286 | 0.942857 | NA | 0.883333 |
| spearman_abs | 0.942857 | 0.904762 | NA | 0.916667 |
| mi | 0.885714 | 0.980952 | NA | 0.900000 |

### Dyn/Prior L1
| Method | ETTm1 | weather | flotation | grinding |
| --- | --- | --- | --- | --- |
| pearson_abs | 0.086004 | 0.016158 | NA | 0.031538 |
| spearman_abs | 0.082701 | 0.010947 | NA | 0.032326 |
| mi | 0.055634 | 0.029904 | NA | 0.026152 |

### Base Alpha Mean
| Method | ETTm1 | weather | flotation | grinding |
| --- | --- | --- | --- | --- |
| pearson_abs | 0.000351 | 0.000354 | NA | 0.000341 |
| spearman_abs | 0.000350 | 0.000352 | NA | 0.000340 |
| mi | 0.000349 | 0.000366 | NA | 0.000334 |

### Gate Mean
| Method | ETTm1 | weather | flotation | grinding |
| --- | --- | --- | --- | --- |
| pearson_abs | 0.002459 | 0.002452 | NA | 0.002490 |
| spearman_abs | 0.002458 | 0.002456 | NA | 0.002490 |
| mi | 0.002460 | 0.002461 | NA | 0.002478 |

### Map Mean Abs
| Method | ETTm1 | weather | flotation | grinding |
| --- | --- | --- | --- | --- |
| pearson_abs | 0.082290 | 0.052608 | NA | 0.046252 |
| spearman_abs | 0.082323 | 0.050822 | NA | 0.046217 |
| mi | 0.082413 | 0.052741 | NA | 0.046238 |

### Graph Stats Summary (PS/Ovl/L1/A/G/Map)
| Method | ETTm1 | weather | flotation | grinding |
| --- | --- | --- | --- | --- |
| pearson_abs | 0.8571/0.9143/0.0860/0.0004/0.0025/0.0823 | 0.2857/0.9429/0.0162/0.0004/0.0025/0.0526 | NA/NA/NA/NA/NA/NA | 0.5000/0.8833/0.0315/0.0003/0.0025/0.0463 |
| spearman_abs | 0.8571/0.9429/0.0827/0.0004/0.0025/0.0823 | 0.2857/0.9048/0.0109/0.0004/0.0025/0.0508 | NA/NA/NA/NA/NA/NA | 0.5000/0.9167/0.0323/0.0003/0.0025/0.0462 |
| mi | 0.8571/0.8857/0.0556/0.0003/0.0025/0.0824 | 0.2857/0.9810/0.0299/0.0004/0.0025/0.0527 | NA/NA/NA/NA/NA/NA | 0.5000/0.9000/0.0262/0.0003/0.0025/0.0462 |
