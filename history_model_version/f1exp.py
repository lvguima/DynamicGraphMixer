import argparse
import subprocess
import sys
from pathlib import Path


def build_base_args(
    graph_log_interval: int,
    data_args: list[str],
    features: str,
    target: str,
    freq: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    enc_in: int,
    dec_in: int,
    c_out: int,
) -> list[str]:
    return [
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model",
        "DynamicGraphMixer",
        *data_args,
        "--features",
        features,
        "--target",
        target,
        "--freq",
        freq,
        "--seq_len",
        str(seq_len),
        "--label_len",
        str(label_len),
        "--pred_len",
        str(pred_len),
        "--e_layers",
        "2",
        "--d_model",
        "128",
        "--d_ff",
        "256",
        "--enc_in",
        str(enc_in),
        "--dec_in",
        str(dec_in),
        "--c_out",
        str(c_out),
        "--batch_size",
        "64",
        "--train_epochs",
        "15",
        "--patience",
        "3",
        "--use_norm",
        "1",
        "--graph_scale",
        "8",
        "--graph_rank",
        "8",
        "--graph_smooth_lambda",
        "0",
        "--temporal_encoder",
        "tcn",
        "--tcn_kernel",
        "3",
        "--tcn_dilation",
        "2",
        "--graph_source",
        "content_mean",
        "--graph_base_mode",
        "mix",
        "--graph_base_alpha_init",
        "-8",
        "--graph_base_l1",
        "0.001",
        "--adj_sparsify",
        "topk",
        "--adj_topk",
        "6",
        "--graph_log_interval",
        str(graph_log_interval),
        "--graph_log_topk",
        "5",
        "--graph_log_num_segments",
        "2",
        "--graph_log_dir",
        "./graph_logs",
    ]


def run_exp(
    python_bin: str,
    exp_id: str,
    model_id: str,
    base_args: list[str],
    extra_args: list[str],
    dry_run: bool,
) -> None:
    args = [
        python_bin,
        "run.py",
        "--model_id",
        model_id,
        "--graph_log_exp_id",
        exp_id,
    ] + base_args + extra_args
    print(f"Running {exp_id}")
    if dry_run:
        print(" ".join(args))
        return
    result = subprocess.run(args, cwd=Path(__file__).resolve().parent)
    if result.returncode != 0:
        raise SystemExit(f"Run failed: {exp_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="F-series experiments for DynamicGraphMixer v3 (SMGP step).")
    parser.add_argument(
        "--stage",
        choices=["F0", "F1", "F2", "all"],
        default="all",
        help="Which F stage to run.",
    )
    parser.add_argument(
        "--graph_log_interval",
        type=int,
        default=200,
        help="Log graph stats every N steps.",
    )
    parser.add_argument(
        "--python",
        dest="python_bin",
        default=sys.executable,
        help="Python executable to run run.py.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running.",
    )
    parser.add_argument(
        "--data",
        default="ETTm1",
        help="Dataset name (default: ETTm1).",
    )
    parser.add_argument(
        "--root_path",
        default="./datasets",
        help="Dataset root path (default: ./datasets).",
    )
    parser.add_argument(
        "--data_path",
        default="ETTm1.csv",
        help="Dataset file name (default: ETTm1.csv).",
    )
    parser.add_argument(
        "--features",
        default="M",
        help="Feature mode (default: M).",
    )
    parser.add_argument(
        "--target",
        default="OT",
        help="Target column (default: OT).",
    )
    parser.add_argument(
        "--freq",
        default="t",
        help="Time feature frequency (default: t).",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=96,
        help="Input sequence length (default: 96).",
    )
    parser.add_argument(
        "--label_len",
        type=int,
        default=48,
        help="Label length (default: 48).",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=96,
        help="Prediction length (default: 96).",
    )
    parser.add_argument(
        "--enc_in",
        type=int,
        default=7,
        help="Encoder input size (default: 7 for ETTm1).",
    )
    parser.add_argument(
        "--dec_in",
        type=int,
        default=7,
        help="Decoder input size (default: 7 for ETTm1).",
    )
    parser.add_argument(
        "--c_out",
        type=int,
        default=7,
        help="Output size (default: 7 for ETTm1).",
    )
    args = parser.parse_args()

    if args.data.lower() == "weather":
        data_args = ["--data", "custom", "--root_path", args.root_path, "--data_path", "weather.csv"]
        features = "M"
        target = "OT"
        freq = args.freq
        seq_len = args.seq_len
        label_len = args.label_len
        pred_len = args.pred_len
        enc_in = 21
        dec_in = 21
        c_out = 21
        dataset_tag = "Weather"
    else:
        data_args = ["--data", args.data, "--root_path", args.root_path, "--data_path", args.data_path]
        features = args.features
        target = args.target
        freq = args.freq
        seq_len = args.seq_len
        label_len = args.label_len
        pred_len = args.pred_len
        enc_in = args.enc_in
        dec_in = args.dec_in
        c_out = args.c_out
        dataset_tag = args.data

    base_args = build_base_args(
        args.graph_log_interval,
        data_args,
        features,
        target,
        freq,
        seq_len,
        label_len,
        pred_len,
        enc_in,
        dec_in,
        c_out,
    )

    if args.stage in ("F0", "all"):
        exp_id = f"v3_F0_{dataset_tag}_v2_best"
        model_id = f"DynamicGraphMixer_TCN_{dataset_tag}_96_96_F0"
        run_exp(
            args.python_bin,
            exp_id,
            model_id,
            base_args,
            ["--gate_mode", "per_var", "--gate_init", "-6"],
            args.dry_run,
        )

    if args.stage in ("F1", "all"):
        variants = [
            ("v3_F1_ema_a0.1", ["--graph_map_norm", "ema_detrend", "--graph_map_alpha", "0.1"]),
            ("v3_F1_ema_a0.3", ["--graph_map_norm", "ema_detrend", "--graph_map_alpha", "0.3"]),
            ("v3_F1_ema_a0.5", ["--graph_map_norm", "ema_detrend", "--graph_map_alpha", "0.5"]),
            ("v3_F1_diff1", ["--graph_map_norm", "diff1"]),
            ("v3_F1_ma_w16", ["--graph_map_norm", "ma_detrend", "--graph_map_window", "16"]),
        ]
        for suffix, map_args in variants:
            exp_id = f"{suffix}_{dataset_tag}"
            model_id = f"DynamicGraphMixer_TCN_{dataset_tag}_96_96_{suffix}"
            run_exp(
                args.python_bin,
                exp_id,
                model_id,
                base_args,
                ["--gate_mode", "per_var", "--gate_init", "-6", *map_args],
                args.dry_run,
            )

    if args.stage in ("F2", "all"):
        exp_id = f"v3_F2_ema_a0.3_gate_-2_{dataset_tag}"
        model_id = f"DynamicGraphMixer_TCN_{dataset_tag}_96_96_F2"
        run_exp(
            args.python_bin,
            exp_id,
            model_id,
            base_args,
            [
                "--gate_mode",
                "per_var",
                "--gate_init",
                "-2",
                "--graph_map_norm",
                "ema_detrend",
                "--graph_map_alpha",
                "0.3",
            ],
            args.dry_run,
        )


if __name__ == "__main__":
    main()
