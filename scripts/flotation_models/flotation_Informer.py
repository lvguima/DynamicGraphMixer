import argparse
import subprocess
import sys
from pathlib import Path

MODEL_NAME = 'Informer'


def build_base_args(
    root_path: str,
    data_path: str,
    features: str,
    target: str,
    freq: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    enc_in: int,
    dec_in: int,
    c_out: int,
    graph_log_interval: int,
) -> list[str]:
    return [
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--model', MODEL_NAME,
        '--data', 'custom',
        '--root_path', root_path,
        '--data_path', data_path,
        '--features', features,
        '--target', target,
        '--freq', freq,
        '--seq_len', str(seq_len),
        '--label_len', str(label_len),
        '--pred_len', str(pred_len),
        '--enc_in', str(enc_in),
        '--dec_in', str(dec_in),
        '--c_out', str(c_out),
        '--batch_size', '64',
        '--train_epochs', '15',
        '--patience', '3',
        '--use_norm', '1',
        '--graph_log_interval', str(graph_log_interval),
        '--graph_log_topk', '5',
        '--graph_log_num_segments', '2',
        '--graph_log_dir', './graph_logs',
    ]


def run_exp(python_bin: str, base_args: list[str], dry_run: bool) -> None:
    exp_id = f'flotation_{MODEL_NAME}_96_96'
    model_id = f'{MODEL_NAME}_flotation_96_96'
    args = [
        python_bin,
        'run.py',
        '--model_id', model_id,
        '--graph_log_exp_id', exp_id,
    ] + base_args
    print(f'Running {MODEL_NAME} on flotation (96/96)')
    if dry_run:
        print(' '.join(args))
        return
    result = subprocess.run(args, cwd=Path(__file__).resolve().parents[1])
    if result.returncode != 0:
        raise SystemExit(f'Run failed: {exp_id}')


def main() -> None:
    parser = argparse.ArgumentParser(description=f'Flotation 96/96 run for {MODEL_NAME}.')
    parser.add_argument('--graph_log_interval', type=int, default=200)
    parser.add_argument('--python', dest='python_bin', default=sys.executable)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--root_path', default='./datasets')
    parser.add_argument('--data_path', default='flotation.csv')
    parser.add_argument('--features', default='M')
    parser.add_argument('--target', default='OT')
    parser.add_argument('--freq', default='t')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=12)
    parser.add_argument('--dec_in', type=int, default=12)
    parser.add_argument('--c_out', type=int, default=12)
    args = parser.parse_args()

    base_args = build_base_args(
        args.root_path,
        args.data_path,
        args.features,
        args.target,
        args.freq,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.enc_in,
        args.dec_in,
        args.c_out,
        args.graph_log_interval,
    )

    run_exp(args.python_bin, base_args, args.dry_run)


if __name__ == '__main__':
    main()
