"""Run the forecasting protocol and write the result tables (results/<dataset>/).

    python -m emaforecast.run --dataset all                   # the four datasets in turn
    python -m emaforecast.run --dataset fisher                # one dataset
    python -m emaforecast.run --dataset rowland --subjects 3  # quick check: first 3 subjects
"""
import argparse

from emaforecast import config
from emaforecast.data import fisher, geschwind, reeves, rowland
from emaforecast.protocol import rolling_origin, tables

DATASETS = {"fisher": fisher, "reeves": reeves, "rowland": rowland, "geschwind": geschwind}


def run(name, subjects=None):
    """Return the three result tables of one dataset as DataFrames:
    per_subject_mae (Fig 2), mae_by_window and off_diagonal_norm_by_window (Fig 3)."""
    dataset = DATASETS[name].build()
    protocol = config.PROTOCOLS[name]
    forecasts, lasso_B = rolling_origin(dataset, protocol, subjects)
    return {"per_subject_mae": tables.per_subject_mae(forecasts, protocol),
            "mae_by_window": tables.mae_by_window(forecasts, protocol),
            "off_diagonal_norm_by_window": tables.off_diagonal_norm(lasso_B, protocol)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the rolling-origin protocol.")
    parser.add_argument("--dataset", required=True, choices=[*DATASETS, "all"])
    parser.add_argument("--subjects", type=int, help="only the first N subjects (quick check)")
    args = parser.parse_args(argv)
    subjects = list(range(args.subjects)) if args.subjects else None
    for name in DATASETS if args.dataset == "all" else [args.dataset]:
        out = config.RESULTS[name]
        out.mkdir(parents=True, exist_ok=True)
        for table_name, table in run(name, subjects).items():
            table.to_csv(out / f"{table_name}.csv", index=False)
        print(f"{name}: tables written to {out}", flush=True)


if __name__ == "__main__":
    main()
