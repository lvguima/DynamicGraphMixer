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
}

PRED_LENS = [30, 60, 90]
PRIOR_METHOD = "pearson_abs"
PRIOR_TOPK = 6
ROUTING_MODE = "affine_learned"
ROUTING_GAMMA = 2.0
ROUTING_WARMUP = 1


def build_base_args(root_path: str, label_len: int, pred_len: int, graph_log_interval: int) -> list[str]:
    topk = min(int(PRIOR_TOPK), DATASET["enc_in"] - 1)
    log_topk = min(6, DATASET["enc_in"] - 1)
    args = [
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
        "16",
        "--adj_sparsify",
        "topk",
        "--adj_topk",
        "6",
        "--graph_base_mode",
        "mix",
        "--base_graph_type",
        "prior",
        "--prior_graph_method",
        PRIOR_METHOD,
        "--prior_graph_topk",
        str(topk),
        "--graph_base_alpha_init",
        "-8",
        "--graph_base_l1",
        "0.0",
        "--gate_mode",
        "per_var",
        "--gate_init",
        "-6",
        "--graph_map_norm",
        "ma_detrend",
        "--graph_map_window",
        "16",
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
        ROUTING_MODE,
        "--routing_conf_metric",
        "overlap_topk",
        "--routing_gamma",
        str(ROUTING_GAMMA),
        "--routing_warmup_epochs",
        str(ROUTING_WARMUP),
        "--graph_log_interval",
        str(graph_log_interval),
        "--graph_log_topk",
        str(log_topk),
        "--graph_log_num_segments",
        "2",
        "--graph_log_dir",
        "./graph_logs",
    ]
    return args


def run_exp(python_bin: str, base_args: list[str], pred_len: int, dry_run: bool) -> None:
    exp_id = f"v8_predlen_{DATASET['name']}_{PRIOR_METHOD}_pl{pred_len}"
    model_id = f"DGmix_v8_predlen_{DATASET['name']}_{PRIOR_METHOD}_pl{pred_len}"
    cmd = [
        python_bin,
        "run.py",
        "--model_id",
        model_id,
        "--graph_log_exp_id",
        exp_id,
    ] + base_args
    print(f"Running {exp_id}")
    if dry_run:
        print(" ".join(cmd))
        return
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2])
    if result.returncode != 0:
        raise SystemExit(f"Run failed: {exp_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="v8 pred_len sweep for grinding.")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--root_path", default="./datasets")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--graph_log_interval", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for pred_len in PRED_LENS:
        base_args = build_base_args(
            args.root_path,
            args.label_len,
            pred_len,
            args.graph_log_interval,
        )
        run_exp(args.python_bin, base_args, pred_len, args.dry_run)


if __name__ == "__main__":
    main()
