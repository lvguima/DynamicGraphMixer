import argparse
import subprocess
import sys
from pathlib import Path


def build_base_args(graph_log_interval: int, data_args: list[str]) -> list[str]:
    return [
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model",
        "DynamicGraphMixer",
        *data_args,
        "--features",
        "M",
        "--target",
        "OT",
        "--freq",
        "t",
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
        "--enc_in",
        "7",
        "--dec_in",
        "7",
        "--c_out",
        "7",
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
    args = parser.parse_args()

    data_args = ["--data", args.data, "--root_path", args.root_path, "--data_path", args.data_path]
    base_args = build_base_args(args.graph_log_interval, data_args)

    if args.stage in ("F0", "all"):
        exp_id = "v3_F0_v2_best"
        model_id = "DynamicGraphMixer_TCN_ETTm1_96_96_F0"
        run_exp(
            args.python_bin,
            exp_id,
            model_id,
            base_args,
            ["--gate_mode", "per_var", "--gate_init", "-6"],
            args.dry_run,
        )

    if args.stage in ("F1", "all"):
        exp_id = "v3_F1_sm_gp_ema"
        model_id = "DynamicGraphMixer_TCN_ETTm1_96_96_F1"
        run_exp(
            args.python_bin,
            exp_id,
            model_id,
            base_args,
            [
                "--gate_mode",
                "per_var",
                "--gate_init",
                "-6",
                "--graph_map_norm",
                "ema_detrend",
                "--graph_map_alpha",
                "0.3",
            ],
            args.dry_run,
        )

    if args.stage in ("F2", "all"):
        exp_id = "v3_F2_sm_gp_ema_gate_-2"
        model_id = "DynamicGraphMixer_TCN_ETTm1_96_96_F2"
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
