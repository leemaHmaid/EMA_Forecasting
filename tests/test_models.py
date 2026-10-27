"""
Model tests (emaforecast.models).

Each estimator must return the reference matrices in tests/reference/models/, to 1e-12.
The inputs are lag pairs from all four datasets, rebuilt from the data in data/ (only
their fingerprints are stored, so these cases are skipped without the data), plus a
few synthetic edge cases.

Run:  pytest tests/test_models.py          (mlVAR: pytest -m slow tests/test_models.py)
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from _helpers import fingerprint, requires_data
from emaforecast.data import fisher, geschwind, reeves, rowland
from emaforecast.models import PER_SUBJECT, clip_spectral, fit_mlvar_population
from emaforecast.protocol.pairs import lag_pairs
from emaforecast.protocol.rolling import _memory_layout, global_scale
from emaforecast.config import PROTOCOLS

REF = Path(__file__).parent / "reference" / "models"
ATOL = 1e-12
# case -> (dataset, how the lag pairs are taken): a short and a long training window of
# Fisher/Reeves (sequential pairs), one subject's within-day pairs of Rowland/Geschwind
DATA_CASES = {
    "fisher_short": ("fisher", "sequential", 0, 12), "fisher_long": ("fisher", "sequential", 5, 100),
    "reeves_short": ("reeves", "sequential", 0, 12), "reeves_long": ("reeves", "sequential", 5, 100),
    "rowland_pairs": ("rowland", "within_day", 3, 20), "geschwind_pairs": ("geschwind", "within_day", 3, 4),
}
BUILD = {"fisher": fisher.build, "reeves": reeves.build, "rowland": rowland.build, "geschwind": geschwind.build}


def _load(name):
    with np.load(REF / f"{name}.npz", allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


@lru_cache(maxsize=None)
def _dataset(name):
    ds = BUILD[name]()
    return ds, _memory_layout(ds.series, PROTOCOLS[name])


def _pairs(case):
    """Rebuild a case's lag pairs from the data, as the protocol builds them, and check
    them against the stored fingerprints."""
    name, mode, subject, size = DATA_CASES[case]
    ds, series = _dataset(name)
    scale = global_scale(series)
    if mode == "sequential":                     # first `size` rows
        train = series[subject][:size]
        z = (train - np.nanmean(train, axis=0)) / scale
        X, Y = z[:-1], z[1:]
    else:                                        # first `size` days
        train = series[subject][: size * ds.beeps_per_day]
        X, Y = lag_pairs((train - np.nanmean(train, axis=0)) / scale, ds.beeps_per_day, "within_day")
    ref = _load(case)
    assert fingerprint(X) == ref["x_sha256"] and fingerprint(Y) == ref["y_sha256"], f"{case}: inputs differ"
    return X, Y


def _model_cases():
    for case in [*DATA_CASES, "constant_item"]:
        for model in _load(case):
            if model not in ("X", "Y", "x_sha256", "y_sha256"):
                marks = [requires_data(DATA_CASES[case][0])] if case in DATA_CASES else []
                yield pytest.param(case, model, marks=marks, id=f"{case}-{model}")


@pytest.mark.parametrize("case,model", list(_model_cases()))
def test_matches_reference(case, model):
    ref = _load(case)
    X, Y = (ref["X"], ref["Y"]) if case not in DATA_CASES else _pairs(case)
    B = PER_SUBJECT[model](X, Y)
    np.testing.assert_allclose(B, ref[model], rtol=0, atol=ATOL)


def test_sem_reference_cases_exercise_the_search():
    """The long SEM cases must actually add cross-lagged paths, otherwise the
    test above would only check the AR start."""
    for case in ("fisher_long", "reeves_long"):
        B = _load(case)["sem"]
        assert np.count_nonzero(B - np.diag(np.diag(B))) > 0, case


def test_ar_is_diagonal_and_zero_for_constant_item():
    ref = _load("constant_item")
    B = PER_SUBJECT["ar"](ref["X"], ref["Y"])
    assert np.count_nonzero(B - np.diag(np.diag(B))) == 0
    assert B[2, 2] == 0.0                      # item 2 is constant in X


def test_clip_spectral():
    ref = _load("clip")
    for B, expected in zip(ref["B"], ref["clipped"]):
        np.testing.assert_allclose(clip_spectral(B), expected, rtol=0, atol=ATOL)
    assert any(not np.allclose(B, c) for B, c in zip(ref["B"], ref["clipped"])), \
        "reference must contain at least one explosive (clipped) matrix"
    for B in ref["clipped"]:
        assert np.max(np.abs(np.linalg.eigvals(B))) <= 1.0


@pytest.mark.slow
@requires_data("geschwind")
def test_mlvar_matches_reference():
    """Geschwind training window W=2 (1577 pooled pairs, 129 subjects): the population
    matrix, which items keep random slopes, and each subject's slope deviations."""
    ds, series = _dataset("geschwind")
    scale = global_scale(series)
    grand = np.nanmean(np.vstack(series), axis=0)
    Xs, Ys, groups = [], [], []
    for i, s in enumerate(series):                 # as rolling._MultilevelVAR._fit, W = 2
        X, Y = lag_pairs((s[: 2 * ds.beeps_per_day] - grand) / scale, ds.beeps_per_day, "within_day")
        Xs.append(X); Ys.append(Y); groups.append(np.full(len(X), i))
    X, Y, groups = np.vstack(Xs), np.vstack(Ys), np.concatenate(groups)
    ref = _load("mlvar_geschwind_w2")
    assert fingerprint(X) == ref["x_sha256"] and fingerprint(Y) == ref["y_sha256"]
    assert fingerprint(groups.astype(float)) == ref["groups_sha256"]

    items = ['cheerful', 'pleasant', 'worry', 'fearful', 'sad', 'relaxed']
    fit = fit_mlvar_population(X, Y, groups, items)
    np.testing.assert_allclose(fit.B_pop, ref["B_pop"], rtol=0, atol=1e-10)
    assert [fit.used_random_slopes[v] for v in items] == list(ref["used_random_slopes"])

    effects = _load("mlvar_geschwind_w2_effects")
    u0 = []
    for sid, slopes in zip(effects["subjects"], effects["slopes"]):
        u0_s, b_s = fit.subject_effects(int(sid))
        np.testing.assert_allclose(b_s, slopes, rtol=0, atol=1e-10)
        u0.append(u0_s)
    assert np.all(effects["slopes"][:, 3, :] == 0)            # 'fearful' fell back: no slope deviations
    assert np.allclose(np.mean(u0, axis=0), 0, atol=1e-6)      # random intercepts are deviations around 0
    assert np.all(np.std(u0, axis=0) > 0.05)                   # ... and people really differ in level
