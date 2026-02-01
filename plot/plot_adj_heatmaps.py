# Run: python plot/plot_adj_heatmaps.py --exp_id v8_predlen_ETTm1_mi_pl96 --graph_log_dir ./graph_logs --out_dir ./plot
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


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


def _load_if_exists(path):
    if not os.path.exists(path):
        return None
    return np.load(path)


def main():
    parser = argparse.ArgumentParser(description="Plot adjacency heatmaps (A_mix, prior, diff).")
    parser.add_argument("--exp_id", required=True)
    parser.add_argument("--graph_log_dir", default="./graph_logs")
    parser.add_argument("--out_dir", default="./plot")
    args = parser.parse_args()

    exp_dir = os.path.join(args.graph_log_dir, args.exp_id)
    art_dir = _latest_artifact_dir(exp_dir)
    if art_dir is None:
        raise SystemExit(f"No artifacts found in {exp_dir}")

    adj = _load_if_exists(os.path.join(art_dir, "adj_mean.npy"))
    base = _load_if_exists(os.path.join(art_dir, "base_adj.npy"))
    raw = _load_if_exists(os.path.join(art_dir, "raw_adj_mean.npy"))
    if adj is None:
        raise SystemExit("adj_mean.npy not found in artifacts.")

    panels = [("A_mix", adj)]
    if base is not None:
        panels.append(("Prior", base))
        panels.append(("A_mix - Prior", adj - base))
    if raw is not None:
        panels.append(("Raw A", raw))

    os.makedirs(os.path.join(args.out_dir, "adj_heatmaps"), exist_ok=True)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (title, mat) in zip(axes, panels):
        im = ax.imshow(mat, cmap="viridis")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(args.exp_id)
    out_path = os.path.join(args.out_dir, "adj_heatmaps", f"{args.exp_id}_adj_heatmaps.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
