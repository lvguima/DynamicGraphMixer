import argparse
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
    "ETTm1": {
        "data": "ETTm1",
        "data_path": "ETTm1.csv",
        "freq": "t",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
    },
    "ETTm2": {
        "data": "ETTm2",
        "data_path": "ETTm2.csv",
        "freq": "t",
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
    },
    "weather": {
        "data": "custom",
        "data_path": "weather.csv",
        "freq": "t",
        "enc_in": 21,
        "dec_in": 21,
        "c_out": 21,
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

BASE_GRAPH = {
    "graph_rank": 8,
    "graph_scale": 8,
    "adj_topk": 6,
    "graph_base_alpha_init": -8,
    "graph_smooth_lambda": 0.0,
}

SWEEPS = {
    "graph_rank": [4, 8, 16],
    "graph_scale": [4, 8, 16],
    "adj_topk": [4, 6, 8],
    "graph_base_alpha_init": [-10, -8, -6],
    "graph_smooth_lambda": [0.0, 0.05, 0.1],
}

PARAM_TO_FLAG = {
    "graph_rank": "--graph_rank",
    "graph_scale": "--graph_scale",
    "adj_topk": "--adj_topk",
    "graph_base_alpha_init": "--graph_base_alpha_init",
    "graph_smooth_lambda": "--graph_smooth_lambda",
}


def fmt_value(val):
    text = str(val)
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    return text


def build_base_args(dataset_name, cfg, graph_log_interval):
    return [
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model",
        "DynamicGraphMixer",
        "--root_path",
        "./datasets",
        "--features",
        "M",
        "--target",
        "OT",
        "--seq_len",
        "96",
        "--label_len",
        "48",
        "--pred_len",
        "96",
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
        str(BASE_GRAPH["graph_rank"]),
        "--graph_scale",
        str(BASE_GRAPH["graph_scale"]),
        "--graph_smooth_lambda",
        str(BASE_GRAPH["graph_smooth_lambda"]),
        "--adj_sparsify",
        "topk",
        "--adj_topk",
        str(BASE_GRAPH["adj_topk"]),
        "--graph_base_mode",
        "mix",
        "--graph_base_alpha_init",
        str(BASE_GRAPH["graph_base_alpha_init"]),
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
        "--graph_log_interval",
        str(graph_log_interval),
        "--graph_log_topk",
        "5",
        "--graph_log_num_segments",
        "2",
        "--graph_log_dir",
        "./graph_logs",
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


def make_experiments():
    experiments = [("GBASE", {})]
    for param, values in SWEEPS.items():
        base_val = BASE_GRAPH[param]
        for val in values:
            if val == base_val:
                continue
            exp_id = f"G_{param}_{fmt_value(val)}"
            experiments.append((exp_id, {param: val}))
    return experiments


def run_exp(python_bin, dataset_name, base_args, exp_id, overrides, dry_run):
    model_id = f"DGmix_GS_{dataset_name}_96_96_{exp_id}"
    log_id = f"{dataset_name}_{exp_id}"
    args = [python_bin, "-u", "run.py", "--model_id", model_id, "--graph_log_exp_id", log_id]
    args += base_args
    for param, val in overrides.items():
        args += [PARAM_TO_FLAG[param], str(val)]
    print("Running", dataset_name, exp_id)
    if dry_run:
        print(" ".join(args))
        return
    subprocess.run(args, check=True)


def parse_dataset_list(text):
    text = text.strip()
    if text.lower() == "all":
        return list(DATASETS.keys())
    names = [item.strip() for item in text.split(",") if item.strip()]
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Graph-structure parameter sweeps on 7 datasets."
    )
    parser.add_argument(
        "--datasets",
        default="ETTh1,ETTh2,ETTm1,ETTm2,flotation,grinding,weather",
        help="Comma-separated dataset list or 'all'.",
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        default=sys.executable,
        help="Python executable to run run.py.",
    )
    parser.add_argument(
        "--graph_log_interval",
        type=int,
        default=200,
        help="Log graph stats every N steps.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running.",
    )
    args = parser.parse_args()

    datasets = parse_dataset_list(args.datasets)
    for name in datasets:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}")

    experiments = make_experiments()
    for dataset_name in datasets:
        cfg = DATASETS[dataset_name]
        base_args = build_base_args(dataset_name, cfg, args.graph_log_interval)
        for exp_id, overrides in experiments:
            run_exp(
                args.python_bin,
                dataset_name,
                base_args,
                exp_id,
                overrides,
                args.dry_run,
            )


if __name__ == "__main__":
    main()
