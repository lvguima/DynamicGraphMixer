# Run: python plot/plot_results_pack.py --exp_id v8_predlen_ETTm1_mi_pl96 --result_dir "D:\\pyproject\\DynamicGraphMixer-main\\results\\long_term_forecast_DGmix_v8_predlen_ETTm1_mi_pl96_DynamicGraphMixer_ETTm1_ftM_sl96_ll48_pl96_dm128_nh8_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_test_0_v8_predlen_ETTm1_mi_pl96" --out_dir ./plot
import argparse
import os

from plot_horizon_error import main as horizon_main
from plot_prediction_case import main as pred_main
from plot_pervar_error_heatmap import main as pervar_main


def main():
    parser = argparse.ArgumentParser(description="Run result-based plots (horizon + case + per-var).")
    parser.add_argument("--exp_id", required=True)
    parser.add_argument("--result_dir", required=True, help="explicit results/<setting> directory")
    parser.add_argument("--out_dir", default="./plot")
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae")
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--var_idx", type=int, default=0)
    args = parser.parse_args()

    # reuse the underlying scripts via their CLIs
    os.system(
        f'python plot/plot_horizon_error.py --exp_id {args.exp_id} '
        f'--result_dir "{args.result_dir}" --out_dir "{args.out_dir}" '
        f'--metric {args.metric} --segments {args.segments}'
    )
    os.system(
        f'python plot/plot_prediction_case.py --exp_id {args.exp_id} '
        f'--result_dir "{args.result_dir}" --out_dir "{args.out_dir}" '
        f'--sample_idx {args.sample_idx} --var_idx {args.var_idx}'
    )
    os.system(
        f'python plot/plot_pervar_error_heatmap.py --exp_id {args.exp_id} '
        f'--result_dir "{args.result_dir}" --out_dir "{args.out_dir}" '
        f'--metric {args.metric}'
    )


if __name__ == "__main__":
    main()
