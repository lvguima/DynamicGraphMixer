import argparse
import subprocess
import sys
from pathlib import Path

DATASET_NAME = 'ETTh2'
DATA_NAME = 'ETTh2'
DATA_PATH = 'ETTh2.csv'
FREQ = 'h'
ENC_IN = 7
DEC_IN = 7
C_OUT = 7
PRED_LENS = [96, 192, 336, 720]


def build_base_args(root_path: str, label_len: int, graph_log_interval: int) -> list[str]:
    return [
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--model', 'DynamicGraphMixer',
        '--data', DATA_NAME,
        '--root_path', root_path,
        '--data_path', DATA_PATH,
        '--features', 'M',
        '--target', 'OT',
        '--freq', FREQ,
        '--seq_len', '96',
        '--label_len', str(label_len),
        '--enc_in', str(ENC_IN),
        '--dec_in', str(DEC_IN),
        '--c_out', str(C_OUT),
        '--e_layers', '2',
        '--d_model', '128',
        '--d_ff', '256',
        '--batch_size', '64',
        '--train_epochs', '15',
        '--patience', '3',
        '--use_norm', '1',
        '--temporal_encoder', 'tcn',
        '--tcn_kernel', '3',
        '--tcn_dilation', '2',
        '--graph_rank', '8',
        '--graph_scale', '8',
        '--graph_smooth_lambda', '0',
        '--adj_sparsify', 'topk',
        '--adj_topk', '6',
        '--graph_base_mode', 'mix',
        '--graph_base_alpha_init', '-8',
        '--graph_base_l1', '0.001',
        '--gate_mode', 'per_var',
        '--gate_init', '-6',
        '--graph_map_norm', 'ma_detrend',
        '--graph_map_window', '16',
        '--decomp_mode', 'ema',
        '--decomp_alpha', '0.1',
        '--trend_head', 'linear',
        '--trend_head_share', '1',
        '--graph_log_interval', str(graph_log_interval),
        '--graph_log_topk', '5',
        '--graph_log_num_segments', '2',
        '--graph_log_dir', './graph_logs',
    ]


def run_exp(python_bin: str, base_args: list[str], pred_len: int, dry_run: bool) -> None:
    exp_id = f'B5_{DATASET_NAME}_96_{pred_len}'
    model_id = f'DGmix_B5_{DATASET_NAME}_96_{pred_len}'
    args = [
        python_bin,
        'run.py',
        '--model_id', model_id,
        '--graph_log_exp_id', exp_id,
        '--pred_len', str(pred_len),
    ] + base_args
    print(f'Running {DATASET_NAME} B5 pred_len={pred_len}')
    if dry_run:
        print(' '.join(args))
        return
    result = subprocess.run(args, cwd=Path(__file__).resolve().parents[2])
    if result.returncode != 0:
        raise SystemExit(f'Run failed: {exp_id}')


def main() -> None:
    parser = argparse.ArgumentParser(description=f'B5 sweep for {DATASET_NAME} (pred_len 96/192/336/720).')
    parser.add_argument('--graph_log_interval', type=int, default=200)
    parser.add_argument('--python', dest='python_bin', default=sys.executable)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--root_path', default='./datasets')
    parser.add_argument('--label_len', type=int, default=48)
    args = parser.parse_args()

    base_args = build_base_args(args.root_path, args.label_len, args.graph_log_interval)

    for pred_len in PRED_LENS:
        run_exp(args.python_bin, base_args, pred_len, args.dry_run)


if __name__ == '__main__':
    main()
