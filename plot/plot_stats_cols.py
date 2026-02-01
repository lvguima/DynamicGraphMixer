# Run: python plot/plot_stats_cols.py --exp_id v8_predlen_flotation_mi_pl2 --cols routing_alpha_mean,routing_conf_mean --graph_log_dir ./graph_logs --out_dir ./plot
import argparse
import csv
import os

import matplotlib.pyplot as plt


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
        try:
            epoch_i = int(float(epoch))
        except ValueError:
            continue
        by_epoch[epoch_i] = row
    return [by_epoch[k] for k in sorted(by_epoch.keys())]


def _get_float(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot selected stats.csv columns vs epoch.")
    parser.add_argument("--exp_id", required=True, help="experiment id (graph_logs/<exp_id>/stats.csv)")
    parser.add_argument("--cols", required=True, help="comma-separated column names in stats.csv")
    parser.add_argument("--graph_log_dir", default="./graph_logs")
    parser.add_argument("--out_dir", default="./plot")
    args = parser.parse_args()

    stats_path = os.path.join(args.graph_log_dir, args.exp_id, "stats.csv")
    if not os.path.exists(stats_path):
        raise SystemExit(f"stats.csv not found: {stats_path}")

    rows = _read_rows(stats_path)
    rows = _group_last_by_epoch(rows)
    if not rows:
        raise SystemExit("No epoch rows found in stats.csv")

    epochs = [int(float(r.get("epoch", 0))) for r in rows]
    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    if not cols:
        raise SystemExit("No valid columns specified.")

    plot_dir = os.path.join(args.out_dir, "custom_stats")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    for col in cols:
        values = [_get_float(r, col) for r in rows]
        plt.plot(epochs, values, marker="o", label=col)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"{args.exp_id}: stats vs epoch")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    out_path = os.path.join(plot_dir, f"{args.exp_id}_" + "_".join(c.replace(':', '_') for c in cols) + ".png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
