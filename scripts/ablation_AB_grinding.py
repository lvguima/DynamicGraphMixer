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

COMMON = [
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--model", "DynamicGraphMixer",
    "--root_path", "./datasets",
    "--features", "M",
    "--target", "OT",
    "--seq_len", "96",
    "--label_len", "48",
    "--pred_len", "96",
    "--e_layers", "2",
    "--d_model", "128",
    "--d_ff", "256",
    "--batch_size", "64",
    "--train_epochs", "15",
    "--patience", "3",
    "--use_norm", "1",
    "--temporal_encoder", "tcn",
    "--tcn_kernel", "3",
    "--tcn_dilation", "2",
    "--graph_rank", "8",
    "--graph_scale", "8",
    "--graph_smooth_lambda", "0",
    "--adj_sparsify", "topk",
    "--adj_topk", "6",
    "--graph_base_mode", "mix",
    "--graph_base_alpha_init", "-8",
    "--graph_base_l1", "0.001",
    "--gate_mode", "per_var",
    "--gate_init", "-6",
    "--graph_map_norm", "ma_detrend",
    "--graph_map_window", "16",
    "--decomp_mode", "ema",
    "--decomp_alpha", "0.1",
    "--trend_head", "linear",
    "--trend_head_share", "1",
    "--graph_log_interval", "200",
    "--graph_log_topk", "5",
    "--graph_log_num_segments", "2",
    "--graph_log_dir", "./graph_logs",
]

EXPS = [
    ("A1_no_dualstream", ["--decomp_mode", "none", "--trend_head", "none"]),
    ("A2_no_SMGP", ["--graph_map_norm", "none"]),
    ("A3_gate_off", ["--gate_init", "-20"]),
    ("A4_no_base_graph", ["--graph_base_mode", "none", "--graph_base_l1", "0"]),
    ("A5_no_sparsify", ["--adj_sparsify", "none"]),
    ("B1_no_TCN_linear", ["--temporal_encoder", "linear"]),
    ("B2_trend_only", ["--trend_only"]),
]


def run_exp(exp_id, extra_args):
    model_id = f"DGmix_AB_{DATASET['name']}_96_96_{exp_id}"
    log_id = f"{DATASET['name']}_{exp_id}"
    args = [
        sys.executable,
        "-u",
        "run.py",
        "--model_id",
        model_id,
        "--graph_log_exp_id",
        log_id,
        "--data",
        DATASET["data"],
        "--data_path",
        DATASET["data_path"],
        "--freq",
        DATASET["freq"],
        "--enc_in",
        str(DATASET["enc_in"]),
        "--dec_in",
        str(DATASET["dec_in"]),
        "--c_out",
        str(DATASET["c_out"]),
    ]
    args += COMMON + extra_args
    print("Running", DATASET["name"], exp_id)
    subprocess.run(args, check=True)


if __name__ == "__main__":
    for exp_id, extra in EXPS:
        run_exp(exp_id, extra)
