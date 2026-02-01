# Run: python plot/plot_error_vs_metric.py --exp_ids v8_predlen_ETTm1_mi_pl96,v8_predlen_weather_mi_pl192 --metric dyn_vs_prior_l1 --target mse --graph_log_dir ./graph_logs --out_dir ./plot
import argparse
import csv
import os

import matplotlib.pyplot as plt


def _read_rows(stats_path: str):
    with open(stats_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _get_last_with_metric(rows, key):
    for row in reversed(rows):
        val = str(row.get(key, "")).strip()
        if val:
            return row
    return rows[-1] if rows else {}


def _get_float(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scatter plot: metric vs error.")
    parser.add_argument("--exp_ids", required=True, help="comma-separated exp ids")
    parser.add_argument("--metric", required=True, help="stats.csv column for x-axis")
    parser.add_argument("--target", choices=["mse", "mae"], default="mse")
    parser.add_argument("--graph_log_dir", default="./graph_logs")
    parser.add_argument("--out_dir", default="./plot")
    args = parser.parse_args()

    exp_ids = [e.strip() for e in args.exp_ids.split(",") if e.strip()]
    xs = []
    ys = []
    labels = []
    for exp_id in exp_ids:
        stats_path = os.path.join(args.graph_log_dir, exp_id, "stats.csv")
        if not os.path.exists(stats_path):
            continue
        rows = _read_rows(stats_path)
        row_metric = _get_last_with_metric(rows, args.metric)
        row_target = _get_last_with_metric(rows, args.target)
        x = _get_float(row_metric, args.metric)
        y = _get_float(row_target, args.target)
        xs.append(x)
        ys.append(y)
        labels.append(exp_id)

    if not xs:
        raise SystemExit("No valid points found.")

    os.makedirs(os.path.join(args.out_dir, "error_vs_metric"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, color="tab:blue")
    for x, y, label in zip(xs, ys, labels):
        ax.text(x, y, label, fontsize=7)
    ax.set_xlabel(args.metric)
    ax.set_ylabel(args.target)
    ax.set_title("Error vs Mechanism Metric")
    ax.grid(True, alpha=0.3)
    out_path = os.path.join(
        args.out_dir, "error_vs_metric", f"scatter_{args.metric}_{args.target}.png"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
