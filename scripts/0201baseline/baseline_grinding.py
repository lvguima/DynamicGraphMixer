import argparse
import subprocess
import sys
from pathlib import Path


DATASET = {
    "name": "grinding",
    "data": "custom",
    "data_path": "grinding.csv",
    "freq": "t",
    "enc_in": 12,
    "dec_in": 12,
    "c_out": 12,
    "pred_lens": [30, 60, 90],
}

MODELS = [
    "iTransformer",
    "PatchTST",
    "TiDE",
    "TimesNet",
    "DLinear",
    "SCINet",
    "Nonstationary_Transformer",
    "Autoformer",
    "Informer",
]


def build_base_args(root_path: str, label_len: int) -> list[str]:
    return [
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
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
        "--enc_in",
        str(DATASET["enc_in"]),
        "--dec_in",
        str(DATASET["dec_in"]),
        "--c_out",
        str(DATASET["c_out"]),
        "--itr",
        "1",
        "--train_epochs",
        "15",
        "--batch_size",
        "64",
        "--patience",
        "3",
        "--learning_rate",
        "0.0001",
        "--use_norm",
        "1",
        "--d_model",
        "128",
        "--d_ff",
        "256",
        "--n_heads",
        "8",
        "--e_layers",
        "2",
        "--d_layers",
        "1",
        "--dropout",
        "0.1",
        "--embed",
        "timeF",
        "--activation",
        "gelu",
        "--factor",
        "1",
        "--des",
        "baseline",
    ]


def run_one(python_bin: str, base_args: list[str], model: str, pred_len: int, dry_run: bool) -> None:
    model_id = f"BL0201_{DATASET['name']}_{model}_pl{pred_len}"
    cmd = [
        python_bin,
        "run.py",
        "--model",
        model,
        "--model_id",
        model_id,
        "--pred_len",
        str(pred_len),
    ] + base_args
    print("Running", model_id)
    if dry_run:
        print(" ".join(cmd))
        return
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2])
    if result.returncode != 0:
        raise SystemExit(f"Run failed: {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="0201 baselines on grinding.")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--root_path", default="./datasets")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--models", default="", help="comma-separated model names (default: all)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    selected = MODELS
    if args.models.strip():
        want = {m.strip() for m in args.models.split(",") if m.strip()}
        selected = [m for m in MODELS if m in want]
        missing = sorted(want - set(MODELS))
        if missing:
            raise SystemExit(f"Unknown/unsupported models: {missing}")

    base_args = build_base_args(args.root_path, args.label_len)
    for pred_len in DATASET["pred_lens"]:
        for model in selected:
            run_one(args.python_bin, base_args, model, pred_len, args.dry_run)


if __name__ == "__main__":
    main()
