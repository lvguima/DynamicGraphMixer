import argparse
import subprocess
import sys


DATASET = {
    "name": "grinding",
    "data": "custom",
    "data_path": "grinding.csv",
    "freq": "t",
    "enc_in": 12,
    "dec_in": 12,
    "c_out": 12,
}


ROUTING_SWEEP = [
    {"tag": "det_g2", "mode": "deterministic", "gamma": 2.0},
    {"tag": "affine", "mode": "affine_learned", "gamma": 2.0},
    {"tag": "det_g1", "mode": "deterministic", "gamma": 1.0},
]


def build_args(root_path: str, label_len: int, pred_len: int, graph_log_interval: int, method: str, topk: int, mode: str, gamma: float, warmup: int):
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
        "16",
        "--graph_smooth_lambda",
        "0",
        "--adj_sparsify",
        "topk",
        "--adj_topk",
        "6",
        "--graph_base_mode",
        "mix",
        "--base_graph_type",
        "prior",
        "--prior_graph_method",
        method,
        "--prior_graph_topk",
        str(topk),
        "--graph_base_alpha_init",
        "-8",
        "--graph_base_l1",
        "0.001",
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
        mode,
        "--routing_conf_metric",
        "overlap_topk",
        "--routing_gamma",
        str(gamma),
        "--routing_warmup_epochs",
        str(warmup),
        "--graph_log_interval",
        str(graph_log_interval),
        "--graph_log_topk",
        "5",
        "--graph_log_num_segments",
        "2",
        "--graph_log_dir",
        "./graph_logs",
    ]


def run_one(python_bin, base_args, tag, method, topk, dry_run):
    exp_id = f"v8_step3_routing_{DATASET['name']}_{method}_k{topk}_{tag}"
    model_id = f"DGmix_v8_step3_routing_{DATASET['name']}_{method}_k{topk}_{tag}"
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
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="v8 Step3 routing sweep for grinding.")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--root_path", default="./datasets")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--graph_log_interval", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--method", type=str, default="mi")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    topk = min(int(args.topk), DATASET["enc_in"] - 1)
    for cfg in ROUTING_SWEEP:
        base_args = build_args(
            args.root_path,
            args.label_len,
            args.pred_len,
            args.graph_log_interval,
            args.method,
            topk,
            cfg["mode"],
            cfg["gamma"],
            args.warmup,
        )
        run_one(args.python_bin, base_args, cfg["tag"], args.method, topk, args.dry_run)


if __name__ == "__main__":
    main()
