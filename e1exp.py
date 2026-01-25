import argparse
import subprocess
import sys
from pathlib import Path

GATE_MODES = ["none", "scalar", "per_var", "per_token"]
GATE_INITS = [-8, -6, -4, -2, 0]
GRAPH_SCALES = [1, 2, 4, 8, 16]


def build_base_args(graph_log_interval: int) -> list[str]:
    return [
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model",
        "DynamicGraphMixer",
        "--data",
        "ETTm1",
        "--root_path",
        "./datasets",
        "--data_path",
        "ETTm1.csv",
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
    parser = argparse.ArgumentParser(description="E1 experiments for DynamicGraphMixer v2.")
    parser.add_argument(
        "--stage",
        choices=["E1-A", "E1-B", "E1-C", "all"],
        default="all",
        help="Which E1 stage to run.",
    )
    parser.add_argument(
        "--best_gate_mode",
        default="scalar",
        help="Best gate mode for E1-C (default: scalar).",
    )
    parser.add_argument(
        "--best_gate_init",
        type=float,
        default=-4.0,
        help="Best gate init for E1-C (default: -4).",
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
    args = parser.parse_args()

    base_args = build_base_args(args.graph_log_interval)

    if args.stage in ("E1-A", "all"):
        for mode in GATE_MODES:
            exp_id = f"v2_1_E1-A_gate_{mode}"
            model_id = f"DynamicGraphMixer_TCN_ETTm1_96_96_E1A_{mode}"
            run_exp(
                args.python_bin,
                exp_id,
                model_id,
                base_args,
                ["--gate_mode", mode, "--gate_init", "-4"],
                args.dry_run,
            )

    if args.stage in ("E1-B", "all"):
        for mode in GATE_MODES:
            for init in GATE_INITS:
                exp_id = f"v2_1_E1-B_{mode}_init{init}"
                model_id = f"DynamicGraphMixer_TCN_ETTm1_96_96_E1B_{mode}_init{init}"
                run_exp(
                    args.python_bin,
                    exp_id,
                    model_id,
                    base_args,
                    ["--gate_mode", mode, "--gate_init", str(init)],
                    args.dry_run,
                )

    if args.stage in ("E1-C", "all"):
        for scale in GRAPH_SCALES:
            exp_id = f"v2_1_E1-C_{args.best_gate_mode}_scale{scale}"
            model_id = f"DynamicGraphMixer_TCN_ETTm1_96_96_E1C_{args.best_gate_mode}_scale{scale}"
            run_exp(
                args.python_bin,
                exp_id,
                model_id,
                base_args,
                [
                    "--gate_mode",
                    args.best_gate_mode,
                    "--gate_init",
                    str(args.best_gate_init),
                    "--graph_scale",
                    str(scale),
                ],
                args.dry_run,
            )


if __name__ == "__main__":
    main()
