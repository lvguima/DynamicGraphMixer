import subprocess
import sys


DATASETS = [
    {
        "name": "ETTm1",
        "data": "ETTm1",
        "data_path": "ETTm1.csv",
        "freq": "t",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
    },
    {
        "name": "weather",
        "data": "custom",
        "data_path": "weather.csv",
        "freq": "t",
        "enc_in": 21,
        "dec_in": 21,
        "c_out": 21,
    },
    {
        "name": "flotation",
        "data": "custom",
        "data_path": "flotation.csv",
        "freq": "t",
        "enc_in": 12,
        "dec_in": 12,
        "c_out": 12,
    },
    {
        "name": "grinding",
        "data": "custom",
        "data_path": "grinding.csv",
        "freq": "t",
        "enc_in": 12,
        "dec_in": 12,
        "c_out": 12,
    },
]

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
    "--gate_mode", "per_var",
    "--gate_init", "-20",
    "--graph_map_norm", "none",
    "--decomp_mode", "ema",
    "--decomp_alpha", "0.1",
    "--trend_head", "linear",
    "--trend_head_share", "1",
    "--graph_log_interval", "200",
    "--graph_log_topk", "5",
    "--graph_log_num_segments", "2",
    "--graph_log_dir", "./graph_logs",
    "--adj_sparsify", "none",
    "--graph_base_mode", "none",
    "--graph_base_l1", "0",
]


def run_one(cfg):
    model_id = f"DGmix_GDIS_{cfg['name']}_96_96"
    log_id = f"{cfg['name']}_GDIS"
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
    print("Running", cfg["name"], "graph-disabled")
    subprocess.run(args, check=True)


if __name__ == "__main__":
    for cfg in DATASETS:
        run_one(cfg)
