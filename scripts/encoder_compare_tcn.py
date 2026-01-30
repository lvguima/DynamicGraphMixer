import subprocess
import sys


DATASETS = {
    "ETTh1": {
        "data": "ETTh1",
        "data_path": "ETTh1.csv",
        "freq": "h",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
    },
    "ETTh2": {
        "data": "ETTh2",
        "data_path": "ETTh2.csv",
        "freq": "h",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
    },
    "flotation": {
        "data": "custom",
        "data_path": "flotation.csv",
        "freq": "t",
        "enc_in": 12,
        "dec_in": 12,
        "c_out": 12,
    },
    "grinding": {
        "data": "custom",
        "data_path": "grinding.csv",
        "freq": "t",
        "enc_in": 12,
        "dec_in": 12,
        "c_out": 12,
    },
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


def run_exp(dataset_name, cfg):
    model_id = f"DGmix_B5_TCN_{dataset_name}_96_96"
    log_id = f"{dataset_name}_B5_TCN"
    args = [
        sys.executable,
        "-u",
        "run.py",
        "--model_id",
        model_id,
        "--graph_log_exp_id",
        log_id,
        "--data",
        cfg["data"],
        "--data_path",
        cfg["data_path"],
        "--freq",
        cfg["freq"],
        "--enc_in",
        str(cfg["enc_in"]),
        "--dec_in",
        str(cfg["dec_in"]),
        "--c_out",
        str(cfg["c_out"]),
    ]
    args += COMMON
    print("Running", dataset_name, "(TCN)")
    subprocess.run(args, check=True)


if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        run_exp(name, cfg)
