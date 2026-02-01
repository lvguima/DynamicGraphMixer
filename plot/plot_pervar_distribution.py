# Run: python plot/plot_pervar_distribution.py --exp_id v8_predlen_grinding_pearson_abs_pl30 --metric conf --graph_log_dir ./graph_logs --out_dir ./plot
# Or:  python plot/plot_pervar_distribution.py --exp_id v8_predlen_grinding_pearson_abs_pl30 --metric conf --compare_first_last --graph_log_dir ./graph_logs --out_dir ./plot
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


METRIC_FILES = {
    "conf": "per_var_conf.npy",
    "entropy": "per_var_entropy.npy",
    "raw_conf": "per_var_raw_conf.npy",
    "raw_entropy": "per_var_raw_entropy.npy",
    "overlap": "per_var_overlap.npy",
    "l1": "per_var_l1.npy",
}


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

def _artifact_dirs(exp_dir: str):
    items = []
    if not os.path.isdir(exp_dir):
        return items
    for name in os.listdir(exp_dir):
        m = re.match(r"epoch(\d+)_step(\d+)", name)
        if not m:
            continue
        epoch = int(m.group(1))
        step = int(m.group(2))
        items.append(((epoch, step), os.path.join(exp_dir, name, "A_mix")))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def main():
    parser = argparse.ArgumentParser(description="Plot per-variable distribution from artifacts.")
    parser.add_argument("--exp_id", required=True)
    parser.add_argument("--metric", choices=sorted(METRIC_FILES.keys()), default="conf")
    parser.add_argument("--artifact", default="", help="explicit artifact dir name: epochXXX_stepYYYYY (default latest)")
    parser.add_argument("--compare_first_last", action="store_true", default=False,
                        help="plot two violins: first artifact vs last artifact")
    parser.add_argument("--graph_log_dir", default="./graph_logs")
    parser.add_argument("--out_dir", default="./plot")
    args = parser.parse_args()

    exp_dir = os.path.join(args.graph_log_dir, args.exp_id)
    if args.artifact.strip():
        art_dir = os.path.join(exp_dir, args.artifact.strip(), "A_mix")
        if not os.path.isdir(art_dir):
            raise SystemExit(f"Artifact dir not found: {art_dir}")
        art_dirs = [art_dir]
        tag = args.artifact.strip()
    elif args.compare_first_last:
        art_dirs = _artifact_dirs(exp_dir)
        if not art_dirs:
            raise SystemExit(f"No artifacts found in {exp_dir}")
        if len(art_dirs) == 1:
            art_dirs = [art_dirs[0], art_dirs[0]]
        else:
            art_dirs = [art_dirs[0], art_dirs[-1]]
        tag = "first_last"
    else:
        art_dir = _latest_artifact_dir(exp_dir)
        if art_dir is None:
            raise SystemExit(f"No artifacts found in {exp_dir}")
        art_dirs = [art_dir]
        tag = "latest"

    file_name = METRIC_FILES[args.metric]
    os.makedirs(os.path.join(args.out_dir, "pervar_dist"), exist_ok=True)
    if len(art_dirs) == 1:
        path = os.path.join(art_dirs[0], file_name)
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}")
        values = np.load(path).reshape(-1)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.violinplot(values, showmeans=True, showmedians=True)
        ax.set_title(f"{args.exp_id} {args.metric} ({tag})")
        ax.set_ylabel(args.metric)
        ax.set_xticks([])
        out_path = os.path.join(args.out_dir, "pervar_dist", f"{args.exp_id}_{args.metric}_violin_{tag}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        return

    # compare two artifacts
    vals = []
    labels = []
    for idx, d in enumerate(art_dirs):
        path = os.path.join(d, file_name)
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}")
        vals.append(np.load(path).reshape(-1))
        labels.append("first" if idx == 0 else "last")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.violinplot(vals, showmeans=True, showmedians=True)
    ax.set_title(f"{args.exp_id} {args.metric} (first vs last)")
    ax.set_ylabel(args.metric)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    out_path = os.path.join(args.out_dir, "pervar_dist", f"{args.exp_id}_{args.metric}_violin_first_last.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
