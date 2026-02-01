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
    "pred_lens": [30, 60, 90],
    "prior_method": "pearson_abs",
    "prior_topk": 6,
    "graph_log_interval": 50,
}

# Best-so-far from 0201_graphsweep_* on grinding (pl30): S5_l1s1_gs8
CFG = {
    "tag": "S5_l1s1_gs8",
    "routing_conf_metric": "l1_distance",
    "routing_l1_scale": 1.0,
    "routing_b_init": 0.0,
    "graph_scale": 8,
    "graph_map_norm": "ma_detrend",
    "graph_map_window": 16,
}


def build_base_args(root_path: str, label_len: int, pred_len: int) -> list[str]:
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
        str(pred_len),
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
        "--graph_scale",
        str(CFG["graph_scale"]),
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
        "--graph_map_norm",
        str(CFG["graph_map_norm"]),
        "--graph_map_window",
        str(CFG["graph_map_window"]),
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
        "--routing_conf_metric",
        str(CFG["routing_conf_metric"]),
        "--routing_l1_scale",
        str(CFG["routing_l1_scale"]),
        "--routing_w_init",
        "2.0",
        "--routing_b_init",
        str(CFG["routing_b_init"]),
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
    ]


def run_one(python_bin: str, base_args: list[str], pred_len: int, dry_run: bool) -> None:
    exp_id = f"0201_ours_best_{DATASET['name']}_pl{pred_len}_{CFG['tag']}"
    model_id = f"DGmix_{exp_id}"
    cmd = [
        python_bin,
        "run.py",
        "--model_id",
        model_id,
        "--graph_log_exp_id",
        exp_id,
    ] + base_args
    print("Running", exp_id)
    if dry_run:
        print(" ".join(cmd))
        return
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2])
    if result.returncode != 0:
        raise SystemExit(f"Run failed: {exp_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run our best config across pred_len on grinding.")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--root_path", default="./datasets")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_lens", default="", help="comma-separated pred_lens to run (default: 30,60,90)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    pred_lens = DATASET["pred_lens"]
    if args.pred_lens.strip():
        pred_lens = [int(x) for x in args.pred_lens.split(",") if x.strip()]

    for pred_len in pred_lens:
        base_args = build_base_args(args.root_path, args.label_len, pred_len)
        run_one(args.python_bin, base_args, pred_len, args.dry_run)


if __name__ == "__main__":
    main()

