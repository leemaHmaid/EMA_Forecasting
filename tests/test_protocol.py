"""
Protocol tests: the rolling-origin loop reproduces the tables behind Figure 2.

The reference tables in tests/reference/fig2/ are the ones Figure 2 is drawn from.
Every table produced by `emaforecast.run` must equal them to 1e-12.

Fast (default): the first subjects of Fisher, Reeves and Rowland (~4 min).
Slow (pytest -m slow): every subject of all four datasets, Geschwind included
(pooled mlVAR fits); about 2 h in total, Fisher alone ~50 min.
The table tests need the data in data/ (docs/data.md) and are skipped otherwise.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _helpers import requires_data
from emaforecast.config import PROTOCOLS
from emaforecast.protocol import rolling_origin
from emaforecast.protocol.pairs import lag_pairs
from emaforecast.run import DATASETS, run

REF = Path(__file__).parent / "reference" / "fig2"
ATOL = 1e-12


def _check_tables(name, subjects=None):
    ids = DATASETS[name].build().subject_ids
    keep = [str(s) for s in (ids if subjects is None else ids[subjects])]
    for table, new in run(name, subjects).items():
        ref = pd.read_csv(REF / f"{name}_{table}.csv")
        ref = ref[ref["subject_id"].astype(str).isin(keep)].reset_index(drop=True)
        new = new.reset_index(drop=True)
        assert list(new.columns) == list(ref.columns), table
        assert new.shape == ref.shape, f"{table}: {new.shape} vs {ref.shape}"
        assert (new["subject_id"].astype(str) == ref["subject_id"].astype(str)).all(), table
        num = [c for c in ref.select_dtypes("number").columns if c != "subject_id"]
        for c in ref.columns.difference(num + ["subject_id"]):
            assert (new[c] == ref[c]).all(), f"{table}.{c}"
        x, y = new[num].to_numpy(float), ref[num].to_numpy(float)
        assert (np.isnan(x) == np.isnan(y)).all(), f"{table}: NaN pattern"
        assert np.nanmax(np.abs(x - y)) <= ATOL, f"{table}: max diff {np.nanmax(np.abs(x - y)):.2e}"


# ---------------------------------------------------------------- lag pairs
def test_within_day_pairs_skip_nights_and_missing_beeps():
    z = np.array([[1.], [2.], [np.nan], [4.], [5.], [6.]])     # 2 days x 3 beeps
    X, Y = lag_pairs(z, beeps_per_day=3, mode="within_day")
    assert X.ravel().tolist() == [1., 4., 5.]                  # (2->nan) dropped, (nan->4) is overnight
    assert Y.ravel().tolist() == [2., 5., 6.]


def test_sequential_pairs_cross_nights():
    z = np.arange(6.).reshape(6, 1)
    X, Y = lag_pairs(z, beeps_per_day=3, mode="sequential")
    assert len(X) == 5 and X[2, 0] == 2. and Y[2, 0] == 3.


# ---------------------------------------------------------------- scoring
@requires_data("fisher")
def test_baselines_scored_only_where_models_are_fitted():
    """Fisher subject 0: the models need >= 9 pairs, so windows 1-2 (3 and 7 pairs)
    are skipped for every model, persistence and the median included."""
    forecasts, _ = rolling_origin(DATASETS["fisher"].build(), PROTOCOLS["fisher"], subjects=[0])
    windows = forecasts.groupby("model")["window_days"].apply(set)
    assert windows["last_step"] == windows["median"] == windows["ar"] == windows["sem"]
    assert min(windows["ar"]) == 3


# ---------------------------------------------------------------- Figure 2 tables
@pytest.mark.parametrize("name,subjects", [
    pytest.param(n, s, marks=requires_data(n)) for n, s in [("fisher", [0]), ("reeves", [0]), ("rowland", [0, 1, 2])]])
def test_reproduces_fig2_tables_subset(name, subjects):
    _check_tables(name, subjects)


@pytest.mark.slow
@pytest.mark.parametrize("name", [
    pytest.param(n, marks=requires_data(n)) for n in ["geschwind", "rowland", "reeves", "fisher"]])
def test_reproduces_fig2_tables_full(name):
    _check_tables(name)
