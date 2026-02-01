# Run: python plot/plot_horizon_error.py --exp_id v8_predlen_ETTm1_mi_pl96 --results_dir ./results --out_dir ./plot
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


def _segment_errors(pred, true, segments=3, metric="mae"):
    pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    true = true.reshape(-1, true.shape[-2], true.shape[-1])
    diff = pred - true
    if metric == "mae":
        err = np.abs(diff).mean(axis=(0, 2))  # per horizon
    else:
        err = (diff ** 2).mean(axis=(0, 2))
    total = err.shape[0]
    seg_size = max(1, total // segments)
    results = []
    for i in range(segments):
        start = i * seg_size
        end = total if i == segments - 1 else min(total, (i + 1) * seg_size)
        results.append(err[start:end].mean())
    return results


def main():
    parser = argparse.ArgumentParser(description="Plot horizon-segment error.")
    parser.add_argument("--exp_id", required=True)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--result_dir", default="", help="explicit results/<setting> dir (overrides search)")
    parser.add_argument("--out_dir", default="./plot")
    parser.add_argument("--segments", type=int, default=3)
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

    errors = _segment_errors(pred, true, segments=args.segments, metric=args.metric)
    labels = [f"S{i+1}" for i in range(len(errors))]

    os.makedirs(os.path.join(args.out_dir, "horizon_error"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, errors, color="tab:blue")
    ax.set_title(f"{args.exp_id} {args.metric} by horizon segment")
    ax.set_ylabel(args.metric)
    out_path = os.path.join(
        args.out_dir, "horizon_error", f"{args.exp_id}_{args.metric}_segments.png"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
