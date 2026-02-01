# Run: python plot/plot_pervar_error_heatmap.py --exp_id v8_predlen_ETTm1_mi_pl96 --results_dir ./results --out_dir ./plot
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def _find_result_dir(results_dir: str, exp_id: str):
    if not os.path.isdir(results_dir):
        return None
    exp_lower = exp_id.lower()
    candidates = []
    for name in os.listdir(results_dir):
        full = os.path.join(results_dir, name)
        if not os.path.isdir(full):
            continue
        if exp_lower in name.lower():
            candidates.append(full)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(description="Plot per-variable MAE heatmap.")
    parser.add_argument("--exp_id", required=True)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--result_dir", default="", help="explicit results/<setting> dir (overrides search)")
    parser.add_argument("--out_dir", default="./plot")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae")
    args = parser.parse_args()

    result_dir = args.result_dir.strip() or _find_result_dir(args.results_dir, args.exp_id)
    if result_dir is None:
        raise SystemExit(f"No results dir found for {args.exp_id}")

    pred_path = os.path.join(result_dir, "pred.npy")
    true_path = os.path.join(result_dir, "true.npy")
    if not (os.path.exists(pred_path) and os.path.exists(true_path)):
        raise SystemExit(f"Missing pred/true in {result_dir}")

    pred = np.load(pred_path)
    true = np.load(true_path)
    diff = pred - true
    if args.metric == "mae":
        per_var = np.abs(diff).mean(axis=(0, 1))
    else:
        per_var = (diff ** 2).mean(axis=(0, 1))

    heat = per_var.reshape(1, -1)
    os.makedirs(os.path.join(args.out_dir, "pervar_error"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 2))
    im = ax.imshow(heat, cmap="magma", aspect="auto")
    ax.set_yticks([])
    ax.set_xlabel("Variable Index")
    ax.set_title(f"{args.exp_id} per-var {args.metric}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = os.path.join(args.out_dir, "pervar_error", f"{args.exp_id}_{args.metric}_pervar.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
