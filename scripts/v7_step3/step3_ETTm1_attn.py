import argparse
import subprocess
import sys


DATASET = {
    "name": "ETTm1",
    "data": "ETTm1",
    "data_path": "ETTm1.csv",
    "freq": "t",
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
}

HEADS_LIST = [4, 8]
WARMUP_LIST = [0, 1]


def build_args(root_path: str, label_len: int, pred_len: int, graph_log_interval: int) -> list[str]:
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
        "attn_token",
        "--attn_layers",
        "1",
        "--attn_topk",
        "0",
        "--residual_scale_init",
        "0.0",
        "--graph_log_interval",
        str(graph_log_interval),
        "--graph_log_topk",
        "5",
        "--graph_log_num_segments",
        "2",
        "--graph_log_dir",
        "./graph_logs",
    ]


def run_one(python_bin, base_args, heads, warmup, dry_run):
    exp_id = f"v7_step3_attn_{DATASET['name']}_H{heads}_W{warmup}_L1"
    model_id = f"DGmix_v7_step3_attn_{DATASET['name']}_H{heads}_W{warmup}_L1"
    cmd = [
        python_bin,
        "run.py",
        "--model_id",
        model_id,
        "--graph_log_exp_id",
        exp_id,
        "--attn_heads",
        str(heads),
        "--warmup_epochs",
        str(warmup),
    ] + base_args
    print("Running", exp_id)
    if dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="v7 Step3 token-attn sweep for ETTm1 (pred_len=96).")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--root_path", default="./datasets")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--graph_log_interval", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base_args = build_args(args.root_path, args.label_len, args.pred_len, args.graph_log_interval)
    for heads in HEADS_LIST:
        for warmup in WARMUP_LIST:
            run_one(args.python_bin, base_args, heads, warmup, args.dry_run)


if __name__ == "__main__":
    main()
