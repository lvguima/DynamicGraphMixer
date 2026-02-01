import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "ETTm1",
        "data": "ETTm1",
        "data_path": "ETTm1.csv",
        "freq": "t",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "pred_len": 96,
        "prior_method": "mi",
        "graph_log_interval": 200,
    },
    {
        "name": "weather",
        "data": "custom",
        "data_path": "weather.csv",
        "freq": "t",
        "enc_in": 21,
        "dec_in": 21,
        "c_out": 21,
        "pred_len": 192,
        "prior_method": "mi",
        "graph_log_interval": 200,
    },
    {
        "name": "grinding",
        "data": "custom",
        "data_path": "grinding.csv",
        "freq": "t",
        "enc_in": 12,
        "dec_in": 12,
        "c_out": 12,
        "pred_len": 30,
        "prior_method": "pearson_abs",
        "graph_log_interval": 200,
    },
    {
        "name": "flotation",
        "data": "custom",
        "data_path": "flotation.csv",
        "freq": "t",
        "enc_in": 12,
        "dec_in": 12,
        "c_out": 12,
        "pred_len": 4,
        "prior_method": "mi",
        "graph_log_interval": 20,
    },
]


def build_base_args(cfg, root_path: str, label_len: int) -> list[str]:
    topk = min(6, cfg["enc_in"] - 1)
    log_topk = min(6, cfg["enc_in"] - 1)
    return [
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model",
        "DynamicGraphMixer",
        "--data",
        cfg["data"],
        "--root_path",
        root_path,
        "--data_path",
        cfg["data_path"],
        "--features",
        "M",
        "--target",
        "OT",
        "--freq",
        cfg["freq"],
        "--seq_len",
        "96",
        "--label_len",
        str(label_len),
        "--pred_len",
        str(cfg["pred_len"]),
        "--enc_in",
        str(cfg["enc_in"]),
        "--dec_in",
        str(cfg["dec_in"]),
        "--c_out",
        str(cfg["c_out"]),
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
        cfg["prior_method"],
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
        "affine_learned",
        "--routing_conf_metric",
        "overlap_topk",
        "--routing_gamma",
        "2.0",
        "--routing_warmup_epochs",
        "1",
        "--graph_log_interval",
        str(cfg["graph_log_interval"]),
        "--graph_log_topk",
        str(log_topk),
        "--graph_log_num_segments",
        "2",
        "--graph_log_dir",
        "./graph_logs",
        "--graph_log_artifacts",
    ]


def run_one(python_bin: str, cfg, root_path: str, label_len: int, dry_run: bool) -> None:
    exp_id = f"v8_predlen_{cfg['name']}_{cfg['prior_method']}_pl{cfg['pred_len']}"
    model_id = f"DGmix_{exp_id}"
    cmd = [
        python_bin,
        "run.py",
        "--model_id",
        model_id,
        "--graph_log_exp_id",
        exp_id,
    ] + build_base_args(cfg, root_path, label_len)
    print("Running", exp_id)
    if dry_run:
        print(" ".join(cmd))
        return
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2])
    if result.returncode != 0:
        raise SystemExit(f"Run failed: {exp_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Representative pred_len experiments for paper plots.")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--root_path", default="./datasets")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for cfg in EXPERIMENTS:
        run_one(args.python_bin, cfg, args.root_path, args.label_len, args.dry_run)


if __name__ == "__main__":
    main()
