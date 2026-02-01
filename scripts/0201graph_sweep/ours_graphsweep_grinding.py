import argparse
import subprocess
import sys
from pathlib import Path


DATASET = {
    "name": "grinding",
    "data": "custom",
    "data_path": "grinding.csv",
    "freq": "t",
    "enc_in": 12,
    "dec_in": 12,
    "c_out": 12,
    "pred_len": 30,
    "prior_method": "pearson_abs",
    "prior_topk": 6,
    "graph_log_interval": 50,
}


SWEEP = [
    {"tag": "S0_default", "conf_metric": "overlap_topk", "l1_scale": 2.0, "b_init": 0.0, "graph_scale": 16, "map_norm": "ma_detrend"},
    {"tag": "S1_l1s2", "conf_metric": "l1_distance", "l1_scale": 2.0, "b_init": 0.0, "graph_scale": 16, "map_norm": "ma_detrend"},
    {"tag": "S2_l1s1", "conf_metric": "l1_distance", "l1_scale": 1.0, "b_init": 0.0, "graph_scale": 16, "map_norm": "ma_detrend"},
    {"tag": "S3_l1s05", "conf_metric": "l1_distance", "l1_scale": 0.5, "b_init": 0.0, "graph_scale": 16, "map_norm": "ma_detrend"},
    {"tag": "S4_l1s1_b-2", "conf_metric": "l1_distance", "l1_scale": 1.0, "b_init": -2.0, "graph_scale": 16, "map_norm": "ma_detrend"},
    {"tag": "S5_l1s1_gs8", "conf_metric": "l1_distance", "l1_scale": 1.0, "b_init": 0.0, "graph_scale": 8, "map_norm": "ma_detrend"},
    {"tag": "S6_l1s1_diff1", "conf_metric": "l1_distance", "l1_scale": 1.0, "b_init": 0.0, "graph_scale": 16, "map_norm": "diff1"},
    {"tag": "S7_l1s1_b-2_gs8", "conf_metric": "l1_distance", "l1_scale": 1.0, "b_init": -2.0, "graph_scale": 8, "map_norm": "ma_detrend"},
]


def build_base_args(root_path: str, label_len: int) -> list[str]:
    topk = min(int(DATASET["prior_topk"]), DATASET["enc_in"] - 1)
    log_topk = min(6, DATASET["enc_in"] - 1)
    return [
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model",
        "DynamicGraphMixer",
        "--data",
        DATASET["data"],
        "--root_path",
        root_path,
        "--data_path",
        DATASET["data_path"],
        "--features",
        "M",
        "--target",
        "OT",
        "--freq",
        DATASET["freq"],
        "--seq_len",
        "96",
        "--label_len",
        str(label_len),
        "--pred_len",
        str(DATASET["pred_len"]),
        "--enc_in",
        str(DATASET["enc_in"]),
        "--dec_in",
        str(DATASET["dec_in"]),
        "--c_out",
        str(DATASET["c_out"]),
        "--e_layers",
        "2",
        "--d_model",
        "128",
        "--d_ff",
        "256",
        "--batch_size",
        "64",
        "--train_epochs",
        "15",
        "--patience",
        "3",
        "--use_norm",
        "1",
        "--temporal_encoder",
        "tcn",
        "--tcn_kernel",
        "3",
        "--tcn_dilation",
        "2",
        "--graph_rank",
        "8",
        "--adj_sparsify",
        "topk",
        "--adj_topk",
        "6",
        "--graph_base_mode",
        "mix",
        "--base_graph_type",
        "prior",
        "--prior_graph_method",
        DATASET["prior_method"],
        "--prior_graph_topk",
        str(topk),
        "--graph_base_l1",
        "0.0",
        "--gate_mode",
        "per_var",
        "--gate_init",
        "-6",
        "--decomp_mode",
        "ema",
        "--decomp_alpha",
        "0.1",
        "--trend_head",
        "linear",
        "--trend_head_share",
        "1",
        "--graph_mixer_type",
        "baseline",
        "--routing_mode",
        "affine_learned",
        "--routing_w_init",
        "2.0",
        "--routing_warmup_epochs",
        "0",
        "--graph_log_interval",
        str(DATASET["graph_log_interval"]),
        "--graph_log_topk",
        str(log_topk),
        "--graph_log_num_segments",
        "2",
        "--graph_log_dir",
        "./graph_logs",
        "--graph_log_artifacts",
    ]


def run_one(python_bin: str, base_args: list[str], cfg: dict, dry_run: bool) -> None:
    exp_id = f"0201_graphsweep_{DATASET['name']}_pl{DATASET['pred_len']}_{cfg['tag']}"
    model_id = f"DGmix_{exp_id}"
    args = [
        python_bin,
        "run.py",
        "--model_id",
        model_id,
        "--graph_log_exp_id",
        exp_id,
        "--graph_scale",
        str(cfg["graph_scale"]),
        "--graph_map_norm",
        str(cfg["map_norm"]),
        "--graph_map_window",
        "16",
        "--routing_conf_metric",
        str(cfg["conf_metric"]),
        "--routing_l1_scale",
        str(cfg["l1_scale"]),
        "--routing_b_init",
        str(cfg["b_init"]),
    ] + base_args
    print("Running", exp_id)
    if dry_run:
        print(" ".join(args))
        return
    result = subprocess.run(args, cwd=Path(__file__).resolve().parents[2])
    if result.returncode != 0:
        raise SystemExit(f"Run failed: {exp_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="0201 graph-dynamics sweep (our model) on grinding.")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--root_path", default="./datasets")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--tags", default="", help="comma-separated tags to run (default: all)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    selected = SWEEP
    if args.tags.strip():
        want = {t.strip() for t in args.tags.split(",") if t.strip()}
        selected = [c for c in SWEEP if c["tag"] in want]
        missing = sorted(want - {c["tag"] for c in SWEEP})
        if missing:
            raise SystemExit(f"Unknown tags: {missing}")

    base_args = build_base_args(args.root_path, args.label_len)
    for cfg in selected:
        run_one(args.python_bin, base_args, cfg, args.dry_run)


if __name__ == "__main__":
    main()

