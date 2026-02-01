# Run: python plot/plot_topk_graph.py --exp_id v8_predlen_grinding_pearson_abs_pl30 --which A_mix --mode top_edges --top_edges 20 --graph_log_dir ./graph_logs --out_dir ./plot
# Or:  python plot/plot_topk_graph.py --exp_id v8_predlen_grinding_pearson_abs_pl30 --which prior --mode top_edges --top_edges 20 --graph_log_dir ./graph_logs --out_dir ./plot
# Or:  python plot/plot_topk_graph.py --exp_id v8_predlen_grinding_pearson_abs_pl30 --which A_mix --mode per_node_topk --topk 3 --directed --graph_log_dir ./graph_logs --out_dir ./plot
import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


def _latest_artifact_dir(exp_dir: str):
    if not os.path.isdir(exp_dir):
        return None
    best = None
    for name in os.listdir(exp_dir):
        m = re.match(r"epoch(\d+)_step(\d+)", name)
        if not m:
            continue
        epoch = int(m.group(1))
        step = int(m.group(2))
        key = (epoch, step, name)
        if best is None or key > best:
            best = key
    if best is None:
        return None
    return os.path.join(exp_dir, best[2], "A_mix")


def _circle_layout(n):
    coords = []
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        coords.append((math.cos(angle), math.sin(angle)))
    return coords


def _load_adj(art_dir: str, which: str, seg_idx: int):
    which_l = which.strip().lower()
    if which_l in ("a_mix", "amix", "mix", "a"):
        base_dir = art_dir
    else:
        base_dir = art_dir

    if seg_idx >= 0:
        if which_l in ("prior", "base"):
            path = os.path.join(base_dir, "base_adj.npy")
            arr = np.load(path)
            return arr
        if which_l in ("raw", "raw_a"):
            path = os.path.join(base_dir, "raw_adj_segments_mean.npy")
            arr = np.load(path)
            if arr.ndim != 3:
                raise ValueError(f"raw_adj_segments_mean.npy shape expected [S,N,N], got {arr.shape}")
            seg_idx = min(seg_idx, arr.shape[0] - 1)
            return arr[seg_idx]
        path = os.path.join(base_dir, "adj_segments_mean.npy")
        arr = np.load(path)
        if arr.ndim != 3:
            raise ValueError(f"adj_segments_mean.npy shape expected [S,N,N], got {arr.shape}")
        seg_idx = min(seg_idx, arr.shape[0] - 1)
        return arr[seg_idx]

    if which_l in ("prior", "base"):
        path = os.path.join(base_dir, "base_adj.npy")
        return np.load(path)
    if which_l in ("raw", "raw_a"):
        path = os.path.join(base_dir, "raw_adj_mean.npy")
        return np.load(path)
    path = os.path.join(base_dir, "adj_mean.npy")
    return np.load(path)


def _top_edges(adj: np.ndarray, top_edges: int):
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = float(adj[i, j])
            edges.append((w, i, j))
    edges.sort(reverse=True, key=lambda x: x[0])
    return edges[: max(0, int(top_edges))]


def _per_node_topk(adj: np.ndarray, topk: int):
    n = adj.shape[0]
    k = max(1, int(topk))
    k = min(k, n - 1)
    edges = []
    for i in range(n):
        row = adj[i].copy()
        row[i] = -np.inf
        idx = np.argsort(-row)[:k]
        for j in idx:
            edges.append((float(adj[i, j]), i, int(j)))
    return edges


def _draw_edges(ax, pos, edges, directed: bool):
    if not edges:
        return
    max_w = max(abs(w) for w, _, _ in edges) or 1.0
    for w, i, j in edges:
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        alpha = min(1.0, 0.15 + abs(w) / max_w)
        lw = 0.5 + 2.0 * abs(w) / max_w
        color = "tab:gray"
        if directed:
            arrow = FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=lw,
                color=color,
                alpha=alpha,
                shrinkA=8,
                shrinkB=8,
            )
            ax.add_patch(arrow)
        else:
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=lw)


def main():
    parser = argparse.ArgumentParser(description="Plot top-k neighbor graph from artifacts.")
    parser.add_argument("--exp_id", required=True)
    parser.add_argument("--which", default="A_mix", choices=["A_mix", "prior", "raw"],
                        help="which adjacency to plot (A_mix/prior/raw)")
    parser.add_argument("--mode", default="top_edges", choices=["top_edges", "per_node_topk"],
                        help="edge selection mode")
    parser.add_argument("--top_edges", type=int, default=20, help="number of strongest directed edges to draw")
    parser.add_argument("--topk", type=int, default=3, help="top-k per node when mode=per_node_topk")
    parser.add_argument("--directed", action="store_true", default=False, help="draw directed edges")
    parser.add_argument("--artifact", default="", help="specific artifact dir name: epochXXX_stepYYYYY (default latest)")
    parser.add_argument("--seg_idx", type=int, default=-1, help="segment index (>=0 uses adj_segments_mean)")
    parser.add_argument("--graph_log_dir", default="./graph_logs")
    parser.add_argument("--out_dir", default="./plot")
    args = parser.parse_args()

    exp_dir = os.path.join(args.graph_log_dir, args.exp_id)
    if args.artifact.strip():
        art_dir = os.path.join(exp_dir, args.artifact.strip(), "A_mix")
    else:
        art_dir = _latest_artifact_dir(exp_dir)
    if art_dir is None or not os.path.isdir(art_dir):
        raise SystemExit(f"No artifacts found in {exp_dir}")

    adj = _load_adj(art_dir, args.which, int(args.seg_idx))
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise SystemExit(f"Adjacency must be square 2D array, got {adj.shape}")
    n = int(adj.shape[0])
    coords = _circle_layout(n)
    pos = {i: coords[i] for i in range(n)}

    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(n):
        x, y = pos[i]
        ax.scatter([x], [y], s=80, color="tab:blue")
        ax.text(x, y, str(i), fontsize=8, ha="center", va="center", color="white")

    if args.mode == "per_node_topk":
        edges = _per_node_topk(adj, args.topk)
        subtitle = f"per_node_topk{args.topk}"
    else:
        edges = _top_edges(adj, args.top_edges)
        subtitle = f"top_edges{args.top_edges}"

    _draw_edges(ax, pos, edges, directed=bool(args.directed))

    seg_tag = f"seg{args.seg_idx}" if int(args.seg_idx) >= 0 else "mean"
    arrow_tag = "directed" if args.directed else "undirected"
    ax.set_title(f"{args.exp_id} {args.which} {seg_tag} {args.mode} {arrow_tag}")
    ax.set_axis_off()
    os.makedirs(os.path.join(args.out_dir, "topk_graph"), exist_ok=True)
    which_tag = args.which.lower()
    out_path = os.path.join(
        args.out_dir,
        "topk_graph",
        f"{args.exp_id}_{which_tag}_{seg_tag}_{args.mode}_{subtitle}_{arrow_tag}.png",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
