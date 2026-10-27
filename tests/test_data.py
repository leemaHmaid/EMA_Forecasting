"""
Data layer tests (emaforecast.data). They need the openESM files in data/ (docs/data.md)
and are skipped otherwise.

1. Every dataset built from the openESM files matches the reference fingerprints in
   tests/reference/data_fingerprints.json exactly (values, missing pattern, subject
   order, id type): the data behind Figure 2.
2. Geschwind's loader yields the complete 6-day x 10-beep baseline grid.
3. EMADataset.to_long() is a lossless view of the per-subject arrays.
4. Optional, with the authors' "R Data.zip" (Fisher et al., 2017, https://osf.io/5ybxt/)
   in data/: the openESM item values equal the authors' R workspaces, and the Fisher
   port reproduces the authors' own R output (`dirdat` = detrended, `dircub` = spline-
   resampled) for all 40 participants, with R's scale() step (the authors' exact
   pipeline) and without it (the variant used here).

Run:  pytest tests/test_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _helpers import fingerprint, read_fisher_rdata, requires_data, requires_fisher_rdata
from emaforecast import config
from emaforecast.data import fisher, geschwind, openesm, reeves, rowland

pytestmark = requires_data("fisher", "reeves", "rowland", "geschwind")

REF = json.loads((Path(__file__).parent / "reference" / "data_fingerprints.json").read_text())
BUILT = {"fisher": fisher.build, "reeves": reeves.build, "rowland": rowland.build,
         "geschwind": geschwind.build}
R_TOLERANCE = 1e-8          # observed worst case ~1e-11; catches any real regression


@pytest.fixture(scope="module")
def datasets():
    """Build each dataset once per test module."""
    return {name: build() for name, build in BUILT.items()}


# --------------------------------------------------------------------------- 1. reference
@pytest.mark.parametrize("name", list(BUILT))
def test_series_match_reference(datasets, name):
    ds, ref = datasets[name], REF[name]
    assert list(ds.items) == ref["items"] and ds.beeps_per_day == ref["beeps_per_day"]
    got = [fingerprint(s) for s in ds.series]
    assert len(got) == len(ref["series_sha256"])
    bad = [i for i, (a, b) in enumerate(zip(got, ref["series_sha256"])) if a != b]
    assert not bad, f"{name}: series differ for subject positions {bad}"


@pytest.mark.parametrize("name", list(BUILT))
def test_subject_ids_match_reference(datasets, name):
    ids = datasets[name].subject_ids
    assert ids.dtype.str == REF[name]["subject_ids_dtype"]
    assert list(ids) == REF[name]["subject_ids"]


def test_fisher_day_starts_match_reference(datasets):
    got = [fingerprint(d.astype(float)) for d in datasets["fisher"].day_starts]
    assert got == REF["fisher"]["day_starts_sha256"]


# --------------------------------------------------------------------------- 2. geschwind
def test_geschwind_raw_loader():
    df = geschwind.load_raw()
    assert df["subject_id"].nunique() == 129
    assert sorted(df["day"].unique()) == list(range(1, geschwind.BASELINE_DAYS + 1))
    assert df.groupby(["subject_id", "day"]).size().eq(geschwind.BEEPS_PER_DAY).all()
    assert int(df[geschwind.GSCHWIND_VARS].isna().sum().sum()) == 10078


def test_geschwind_build_is_complete_grid(datasets):
    ds = datasets["geschwind"]
    assert len(ds.series) == 129
    assert all(s.shape == (60, 6) for s in ds.series)
    assert list(ds.subject_ids) == list(range(129))


# --------------------------------------------------------------------------- 3. to_long
@pytest.mark.parametrize("name", list(BUILT))
def test_to_long_is_lossless(datasets, name):
    ds = datasets[name]
    long = ds.to_long()
    assert len(long) == sum(len(s) for s in ds.series)
    assert list(long.columns) == ["subject_id", "t", "day", "beep", *ds.items]
    for sid, arr in zip(ds.subject_ids, ds.series):
        back = long.loc[long["subject_id"] == sid, ds.items].to_numpy(float)
        assert np.array_equal(back, arr, equal_nan=True)


def test_rowland_long_matches_raw_file(datasets):
    assert len(datasets["rowland"].to_long()) == len(openesm.read("rowland"))   # 125 x 40 x 6


# --------------------------------------------------------------------------- 4. fisher vs R
FISHER_IDS = list(fisher.list_participant_ids()) if config.RAW["fisher"].exists() else []


def _max_abs_diff(a, b):
    with np.errstate(invalid="ignore"):          # NaN cells (missed beeps) are expected
        return float(np.nanmax(np.abs(a - b)))


@requires_fisher_rdata
def test_fisher_openesm_matches_rdata():
    """openESM's item values are row-for-row identical to the authors' R workspaces, and
    its prompt times, converted back to local time, equal the authors' timestamps."""
    tsv = openesm.read("fisher")
    rdata_cols = ['energetic', 'enthusiastic', 'content', 'irritable', 'restless',
                  'worried', 'guilty', 'afraid', 'anhedonia', 'angry', 'hopeless',
                  'down', 'positive', 'fatigue', 'tension', 'concentrate',
                  'accepted', 'threatened', 'ruminate', 'avoid_act', 'reassure',
                  'procrast', 'avoid_people']
    tsv_cols = (fisher.TRIM_COLS_TSV[:16] + ['accepted', 'threatened'] +
                fisher.TRIM_COLS_TSV[16:20] + ['avoid_people'])
    for pid in FISHER_IDS:
        raw = read_fisher_rdata(pid)["data"]
        sub = tsv[tsv["id"] == pid].reset_index(drop=True)
        assert len(sub) == len(raw), f"{pid}: {len(sub)} openESM rows vs {len(raw)} RData rows"
        diff = _max_abs_diff(raw[rdata_cols].astype("float64").values,
                             sub[tsv_cols].astype("float64").values)
        assert diff == 0, f"{pid}: item values differ (max {diff})"
        local = openesm.local_times(sub["start"])
        authors = pd.to_datetime(raw["start"], format="%m/%d/%Y %H:%M", errors="coerce")
        assert pd.Series(local.values).equals(pd.Series(authors.values)), f"{pid}: prompt times differ"


def _diff_to_r(pid, apply_scale):
    """Worst |port - R| over all items, for the detrended and the resampled series.

    With scale() the port must equal R's dirdat/dircub directly. Without it, R was
    never run, but the target follows exactly: OLS residuals are equivariant to the
    affine scale() (only the SD factor survives), and the fmm spline is linear in y,
    so  no-scale result = SD x R result,  SD = per-item sample SD (ddof=1)."""
    trim, cumsum_t = fisher.load_participant(pid)
    detrended = fisher.zscore_detrend(trim, cumsum_t, apply_scale=apply_scale)
    resampled = fisher.spline_resample(detrended, cumsum_t)
    ref = read_fisher_rdata(pid, names=("dirdat", "dircub"))
    worst = 0.0
    with np.errstate(invalid="ignore"):          # R's NA cells propagate as NaN
        for j, col in enumerate(fisher.TRIM_COLS_R):
            y = trim[col].values.astype(float)
            sd = 1.0 if apply_scale else np.std(y[~np.isnan(y)], ddof=1)
            worst = max(worst,
                        _max_abs_diff(detrended[col].values.astype(float),
                                      ref["dirdat"].iloc[:, j].values.astype(float) * sd),
                        _max_abs_diff(resampled[col].values.astype(float),
                                      ref["dircub"].iloc[:, j].values.astype(float) * sd))
    return worst


@requires_fisher_rdata
@pytest.mark.parametrize("pid", FISHER_IDS)
def test_fisher_matches_r_with_scale(pid):
    assert _diff_to_r(pid, apply_scale=True) < R_TOLERANCE


@requires_fisher_rdata
@pytest.mark.parametrize("pid", FISHER_IDS)
def test_fisher_matches_r_without_scale(pid):
    assert _diff_to_r(pid, apply_scale=False) < R_TOLERANCE
