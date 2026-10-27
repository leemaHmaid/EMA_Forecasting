"""File locations and the per-dataset evaluation settings. Every path in the package
comes from here. The raw data are not distributed with the code: docs/data.md says where
to download each file and where to put it."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # repository root
DATA = ROOT / "data"

# Raw data: one openESM file per dataset (docs/data.md), placed in data/
RAW = {
    "fisher":    DATA / "0033_fisher_ts.tsv",
    "reeves":    DATA / "0034_reeves_ts.tsv",
    "rowland":   DATA / "0006_rowland_ts.tsv",
    "geschwind": DATA / "0010_geschwind_ts.tsv",
}


def raw_file(dataset):
    """Path of a dataset's raw file; a clear error if it has not been downloaded."""
    path = RAW[dataset]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. The raw data are not included in this "
                                f"repository; docs/data.md says where to download it.")
    return path


@dataclass(frozen=True)
class Protocol:
    """How the rolling-origin evaluation is run for one dataset (docs/methods.md)."""
    # Which lag-1 transitions the models learn from:
    #   'sequential' every consecutive row (Fisher, Reeves: equidistant resampled series)
    #   'within_day' consecutive beeps of the same day with all items observed (Rowland,
    #                Geschwind, as in the original analyses)
    pairs: str
    # Start of the one-day forecast: 'last_train_row' (the last row before the test day)
    # or 'first_test_beep' (the first fully observed beep of the test day).
    one_day_seed: str
    one_day_h: int      # steps from that start to the last beep of the test day
    # A training window is used only if it has at least this many lag pairs; otherwise
    # no model is fitted and no forecast of that window is scored (baselines included).
    min_fit_pairs: int
    min_window: int     # first training window (days) included in the results
    # Each dataset's original network model: 'sem' (stepwise SEM; Fisher, Reeves),
    # 'var_ols' (unregularised VAR; Rowland), 'mlvar_ind' (multilevel VAR with
    # individual networks; Geschwind).
    var_model: str


PROTOCOLS = {
    "fisher": Protocol(pairs="sequential", one_day_seed="last_train_row", one_day_h=4,
                       min_fit_pairs=9, min_window=1, var_model="sem"),
    "reeves": Protocol(pairs="sequential", one_day_seed="last_train_row", one_day_h=4,
                       min_fit_pairs=9, min_window=1, var_model="sem"),
    "rowland": Protocol(pairs="within_day", one_day_seed="first_test_beep", one_day_h=5,
                        min_fit_pairs=10, min_window=7, var_model="var_ols"),
    "geschwind": Protocol(pairs="within_day", one_day_seed="first_test_beep", one_day_h=9,
                          min_fit_pairs=10, min_window=2, var_model="mlvar_ind"),
}

# Result tables (one folder per dataset) and figures
RESULTS = {name: ROOT / "results" / name for name in PROTOCOLS}
FIGURES = ROOT / "results" / "figures"
