import csv
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def _stack_adjs(adjs):
    if not adjs:
        return None
    if isinstance(adjs[0], torch.Tensor):
        return torch.stack(adjs, dim=0).float()
    return torch.from_numpy(np.stack(adjs, axis=0)).float()


def compute_graph_stats(adjs, topk=5):
    stacked = _stack_adjs(adjs)
    if stacked is None:
        return None
    eps = 1e-12
    entropy = -(stacked * (stacked + eps).log()).sum(-1)
    entropy_mean = entropy.mean().item()

    k = int(topk)
    if k <= 0:
        topk_mass = 0.0
    else:
        k = min(k, stacked.shape[-1])
        topk_mass = torch.topk(stacked, k, dim=-1).values.sum(-1).mean().item()

    adj_mean = stacked.mean().item()
    adj_var = stacked.var(unbiased=False).item()
    adj_max = stacked.max().item()

    if stacked.shape[0] > 1:
        l1_diff = torch.abs(stacked[1:] - stacked[:-1]).mean().item()
    else:
        l1_diff = 0.0

    return {
        "entropy_mean": entropy_mean,
        "topk_mass": topk_mass,
        "l1_adj_diff": l1_diff,
        "adj_mean": adj_mean,
        "adj_var": adj_var,
        "adj_max": adj_max,
        "segments": int(stacked.shape[0]),
        "num_vars": int(stacked.shape[-1]),
    }


def append_graph_stats(csv_path, stats):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(stats)


def _topk_neighbors(adj, topk):
    k = int(topk)
    if k <= 0:
        return {}
    c = adj.shape[0]
    k = min(k, c)
    scores = adj.copy()
    np.fill_diagonal(scores, -np.inf)
    idx = np.argsort(-scores, axis=1)[:, :k]
    vals = np.take_along_axis(adj, idx, axis=1)
    result = {}
    for i in range(c):
        result[str(i)] = [
            {"j": int(idx[i, j]), "w": float(vals[i, j])} for j in range(k)
        ]
    return result


def save_graph_visuals(adjs, out_dir, topk=5, num_segments=1):
    stacked = _stack_adjs(adjs)
    if stacked is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    mean_adjs = stacked.mean(dim=1).cpu().numpy()
    np.save(os.path.join(out_dir, "adj_segments_mean.npy"), mean_adjs)

    seg_count = mean_adjs.shape[0]
    num_segments = max(1, min(int(num_segments), seg_count))
    for seg_idx in range(num_segments):
        adj = mean_adjs[seg_idx]
        plt.figure(figsize=(4, 3))
        plt.imshow(adj, cmap="viridis")
        plt.colorbar()
        plt.title(f"Adjacency mean seg {seg_idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"adj_heatmap_seg{seg_idx}.png"))
        plt.close()

        if topk > 0:
            neighbors = _topk_neighbors(adj, topk)
            with open(os.path.join(out_dir, f"topk_neighbors_seg{seg_idx}.json"), "w") as f:
                json.dump(neighbors, f, indent=2)
