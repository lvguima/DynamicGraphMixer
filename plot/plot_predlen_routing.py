# Run: python plot/plot_predlen_routing.py --dataset flotation --exp_prefix v8_predlen_flotation --graph_log_dir ./graph_logs --out_dir ./plot
# Or:  python plot/plot_predlen_routing.py --exp_id v8_predlen_flotation_mi_pl2 --graph_log_dir ./graph_logs --out_dir ./plot
import argparse
import csv
import os
import re

import matplotlib.pyplot as plt


def _find_experiments(graph_log_dir: str, exp_prefix: str, exp_id: str):
    items = []
    if not os.path.isdir(graph_log_dir):
        return items
    if exp_id:
        stats_path = os.path.join(graph_log_dir, exp_id, "stats.csv")
        if os.path.exists(stats_path):
            items.append((None, exp_id, stats_path))
        return items
    for name in os.listdir(graph_log_dir):
        if exp_prefix and not name.startswith(exp_prefix):
            continue
        m = re.search(r"pl(\d+)", name)
        if not m:
            continue
        pred_len = int(m.group(1))
        stats_path = os.path.join(graph_log_dir, name, "stats.csv")
        if os.path.exists(stats_path):
            items.append((pred_len, name, stats_path))
    return sorted(items, key=lambda x: x[0])


def _read_rows(stats_path: str):
    with open(stats_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def _group_last_by_epoch(rows):
    by_epoch = {}
    for row in rows:
        epoch = row.get("epoch", "")
        if epoch == "":
            continue
        by_epoch[int(float(epoch))] = row
    return [by_epoch[k] for k in sorted(by_epoch.keys())]


def _get_float(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description="Plot routing confidence/alpha vs pred_len.")
    parser.add_argument("--dataset", required=True, help="dataset name used in exp prefix")
    parser.add_argument("--exp_prefix", default="", help="experiment prefix (default: v8_predlen_<dataset>)")
    parser.add_argument("--exp_id", default="", help="single experiment id (overrides exp_prefix)")
    parser.add_argument("--graph_log_dir", default="./graph_logs")
    parser.add_argument("--out_dir", default="./plot")
    args = parser.parse_args()

    exp_prefix = args.exp_prefix or f"v8_predlen_{args.dataset}"
    exps = _find_experiments(args.graph_log_dir, exp_prefix, args.exp_id)
    if not exps:
        raise SystemExit(f"No experiments found for prefix: {exp_prefix}")

    plot_dir = os.path.join(args.out_dir, "predlen_routing")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for pred_len, exp_name, stats_path in exps:
        rows = _read_rows(stats_path)
        rows = _group_last_by_epoch(rows)
        epochs = [int(float(r.get("epoch", 0))) for r in rows]
        conf_means = [_get_float(r, "routing_conf_mean") for r in rows]
        alpha_means = [_get_float(r, "routing_alpha_mean") for r in rows]
        label = exp_name if pred_len is None else f"pl{pred_len}"
        plt.plot(epochs, conf_means, marker="o", label=f"{label} conf")
        plt.plot(epochs, alpha_means, marker="s", label=f"{label} alpha")
    plt.xlabel("Epoch")
    plt.ylabel("Routing")
    plt.title(f"{args.dataset}: Routing vs epoch")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    out_path = os.path.join(plot_dir, f"{args.dataset}_epoch_routing.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
