import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

from data_provider.data_loader import Dataset_ETT_minute, Dataset_Custom

try:
    from sklearn.feature_selection import mutual_info_regression
except Exception:
    mutual_info_regression = None


DATASET_CONFIGS = {
    "ETTm1": {"data": "ETTm1", "data_path": "ETTm1.csv", "freq": "t", "cls": Dataset_ETT_minute},
    "ETTm2": {"data": "ETTm2", "data_path": "ETTm2.csv", "freq": "t", "cls": Dataset_ETT_minute},
    "ETTh1": {"data": "ETTh1", "data_path": "ETTh1.csv", "freq": "h", "cls": Dataset_Custom},
    "ETTh2": {"data": "ETTh2", "data_path": "ETTh2.csv", "freq": "h", "cls": Dataset_Custom},
    "weather": {"data": "custom", "data_path": "weather.csv", "freq": "t", "cls": Dataset_Custom},
    "flotation": {"data": "custom", "data_path": "flotation.csv", "freq": "t", "cls": Dataset_Custom},
    "grinding": {"data": "custom", "data_path": "grinding.csv", "freq": "t", "cls": Dataset_Custom},
}


def _spearman_corrcoef(x: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float32)
    return np.corrcoef(ranks, rowvar=False)


def _mutual_info_matrix(x: np.ndarray, random_state: int = 0) -> np.ndarray:
    if mutual_info_regression is None:
        raise ImportError("scikit-learn is required for mutual information.")
    n_vars = x.shape[1]
    mi = np.zeros((n_vars, n_vars), dtype=np.float32)
    for j in range(n_vars):
        y = x[:, j]
        x_other = np.delete(x, j, axis=1)
        scores = mutual_info_regression(
            x_other,
            y,
            discrete_features=False,
            random_state=random_state,
        )
        idx = 0
        for i in range(n_vars):
            if i == j:
                continue
            mi[j, i] = scores[idx]
            idx += 1
    mi = (mi + mi.T) * 0.5
    return mi


def _topk_mask(weights: np.ndarray, topk: int) -> np.ndarray:
    n_vars = weights.shape[0]
    mask = np.zeros_like(weights, dtype=np.float32)
    if topk <= 0 or topk >= n_vars:
        mask[:] = 1.0
        return mask
    for i in range(n_vars):
        idx = np.argpartition(weights[i], -topk)[-topk:]
        mask[i, idx] = 1.0
    return mask


def _sym_norm(adj: np.ndarray) -> np.ndarray:
    deg = adj.sum(axis=1)
    deg = np.clip(deg, 1e-12, None)
    inv_sqrt = 1.0 / np.sqrt(deg)
    return adj * inv_sqrt[:, None] * inv_sqrt[None, :]


def build_prior_graph(
    dataset: str,
    root_path: str,
    data_path: str,
    method: str,
    topk: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
    use_norm: int,
    target: str,
):
    cfg = DATASET_CONFIGS.get(dataset)
    if cfg is None:
        raise ValueError(f"Unsupported dataset: {dataset}")
    data_path = data_path or cfg["data_path"]
    freq = cfg["freq"]
    DataCls = cfg["cls"]
    args = SimpleNamespace(augmentation_ratio=0)
    data_set = DataCls(
        args=args,
        root_path=root_path,
        data_path=data_path,
        flag="train",
        size=[seq_len, label_len, pred_len],
        features="M",
        target=target,
        scale=bool(use_norm),
        timeenc=0,
        freq=freq,
        seasonal_patterns=None,
    )
    data = data_set.data_x.astype(np.float32)
    data = data[~np.isnan(data).any(axis=1)]
    n_vars = data.shape[1]
    if n_vars < 2:
        raise ValueError("Need at least 2 variables to build prior graph.")

    if method == "pearson_abs":
        corr = np.corrcoef(data, rowvar=False)
        weights = np.abs(corr)
    elif method == "spearman_abs":
        corr = _spearman_corrcoef(data)
        weights = np.abs(corr)
    elif method == "mi":
        weights = _mutual_info_matrix(data)
    else:
        raise ValueError(f"Unsupported method: {method}")

    np.fill_diagonal(weights, 0.0)
    k = min(int(topk), n_vars - 1)
    mask = _topk_mask(weights, k)
    adj = weights * mask
    adj = _sym_norm(adj)
    return adj.astype(np.float32), {
        "dataset": dataset,
        "data_path": data_path,
        "method": method,
        "topk": k,
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len,
        "use_norm": bool(use_norm),
        "target": target,
        "num_vars": int(n_vars),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset-level prior graph (correlation/MI).")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--root_path", default="./datasets")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--method", default="pearson_abs", choices=["pearson_abs", "spearman_abs", "mi"])
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--out_dir", default="prior_graphs")
    args = parser.parse_args()

    adj, meta = build_prior_graph(
        dataset=args.dataset,
        root_path=args.root_path,
        data_path=args.data_path,
        method=args.method,
        topk=args.topk,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        use_norm=args.use_norm,
        target=args.target,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = f"{args.dataset}_{args.method}_topk{meta['topk']}.npy"
    out_path = os.path.join(args.out_dir, out_name)
    np.save(out_path, adj)

    meta_path = out_path.replace(".npy", ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved prior graph to {out_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
