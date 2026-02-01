# Run: python plot/plot_dualstream_stack.py --exp_id v8_predlen_weather_mi_pl192 --graph_log_dir ./graph_logs --out_dir ./plot
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
    parser = argparse.ArgumentParser(description="Stackplot dual-stream energy vs epoch.")
    parser.add_argument("--exp_id", required=True)
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
    e_trend = [_get_float(r, "E_trend") for r in rows]
    e_season = [_get_float(r, "E_season") for r in rows]

    os.makedirs(os.path.join(args.out_dir, "dualstream_stack"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.stackplot(epochs, e_trend, e_season, labels=["E_trend", "E_season"], alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy")
    ax.set_title(f"{args.exp_id} Dual-Stream Energy")
    ax.legend(loc="upper right")
    out_path = os.path.join(args.out_dir, "dualstream_stack", f"{args.exp_id}_dualstream_stack.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
