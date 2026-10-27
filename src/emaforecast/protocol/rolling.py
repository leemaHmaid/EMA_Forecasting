"""The rolling-origin (expanding-window) forecasting protocol.

For every subject and every training window W = 1, 2, ... (in days):

  1. TRAIN   rows of days 1..W. Centre them on their own mean, divide by the
             global item SD, build lag pairs, fit each model. A window with fewer
             than `min_fit_pairs` pairs is skipped entirely, so persistence, the
             median and every model are scored on exactly the same forecasts.
  2. TEST    day W+1.
             one-step : every consecutive observed pair of beeps; each forecast
                        starts from the actual previous beep
             one-day  : one forecast iterated `one_day_h` steps from a start row
  3. SCORE   error = mean over items k of |actual_k - forecast_k| / global SD_k

A "day" is a block of `beeps_per_day` consecutive rows. For Fisher/Reeves
(spline-resampled, equidistant) that is 4 rows, about one calendar day.

Returns one row per forecast: subject_id, model, window_days, horizon, mae.
"""
import numpy as np
import pandas as pd

from emaforecast.models import PER_SUBJECT, clip_spectral, fit_mlvar_population

from .forecast import (complete, constant_predictor, forecast_error, linear_predictor,
                       persistence_predictor)
from .pairs import lag_pairs


def global_scale(series):
    """Per-item SD over all observations of all subjects (all days), giving one fixed
    scale per item for standardising and for the error."""
    sd = np.nanstd(np.vstack(series), axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-10)] = 1.0
    return sd


def _memory_layout(series, protocol):
    """Within-day datasets are processed as column-major arrays. This only affects
    the last bits of floating-point sums, and keeps results bit-identical to the
    reference tables in tests/reference/fig2."""
    if protocol.pairs == "within_day":
        return [np.asfortranarray(s) for s in series]
    return list(series)


def rolling_origin(dataset, protocol, subjects=None):
    """Run the protocol. `subjects` = list of positions to restrict to (quick checks).

    Returns (forecasts, lasso_B) where lasso_B[(subject_id, W)] is the LASSO matrix
    before spectral clipping, used for the cross-symptom norm (Fig 3).
    """
    bpd = dataset.beeps_per_day
    all_series = _memory_layout(dataset.series, protocol)
    scale = global_scale(all_series)
    multilevel = protocol.var_model == "mlvar_ind"
    models = ["ar", "lasso"] + ([] if multilevel else [protocol.var_model])
    mlvar = _MultilevelVAR(all_series, dataset.items, scale, bpd) if multilevel else None

    rows, lasso_B = [], {}
    positions = range(len(all_series)) if subjects is None else subjects
    for i in positions:
        sid, series = dataset.subject_ids[i], all_series[i]
        for W in range(1, len(series) // bpd):            # test day W+1 must exist
            train, test = series[: W * bpd], series[W * bpd:(W + 1) * bpd]
            centre = np.nanmean(train, axis=0)
            centre = np.where(np.isfinite(centre), centre, 0.0)
            X, Y = lag_pairs((train - centre) / scale, bpd, protocol.pairs)
            if len(X) < protocol.min_fit_pairs:
                continue

            median = np.nanmedian(train, axis=0)
            median = np.where(np.isfinite(median), median, centre)
            predictors = {"last_step": persistence_predictor(),
                          "median": constant_predictor(median)}
            for name in models:
                B = PER_SUBJECT[name](X, Y)
                if B is None:
                    continue
                if name == "lasso":
                    lasso_B[(sid, W)] = B.copy()
                predictors[name] = linear_predictor(clip_spectral(B), centre, scale)
            if mlvar is not None:
                predictors[protocol.var_model] = mlvar.predictor(i, W)

            for name, predict in predictors.items():
                for horizon, actual, pred in _score_test_day(predict, train, test, protocol):
                    rows.append((sid, name, W, horizon, forecast_error(actual, pred, scale)))

    forecasts = pd.DataFrame(rows, columns=["subject_id", "model", "window_days", "horizon", "mae"])
    return forecasts, lasso_B


def _score_test_day(predict, train, test, protocol):
    """Yield (horizon, actual, forecast) for every scored forecast of the test day."""
    # one-step: each forecast starts from the actual previous beep
    if protocol.pairs == "sequential":
        steps = [(train[-1], test[0])] + list(zip(test[:-1], test[1:]))
    else:                                              # within_day: never across the night
        steps = list(zip(test[:-1], test[1:]))
    for prev, actual in steps:
        if complete(prev) and complete(actual):
            yield "one_step", actual, predict(prev, 1)

    # one-day: one forecast iterated h steps ahead
    h = protocol.one_day_h
    if protocol.one_day_seed == "last_train_row":
        day = np.vstack([train[-1:], test])            # start = row 0, target = row h
        seed_idx = 0
    else:                                              # first fully observed beep of the test day
        day = test
        seed_idx = next((k for k in range(len(day)) if complete(day[k])), None)
    if seed_idx is not None and seed_idx + h < len(day):
        seed, actual = day[seed_idx], day[seed_idx + h]
        if complete(seed) and complete(actual):
            yield "one_day", actual, predict(seed, h)


class _MultilevelVAR:
    """Multilevel VAR (Bringmann et al., 2013), used for Geschwind.

    Fitted once per training window on all subjects' pooled within-day pairs of days
    1..W. All subjects are centred on ONE common value, the grand mean of the data,
    so that differences in level between people stay in the data and are absorbed by
    the random intercepts. Each subject is forecast with Bringmann et al.'s individual
    network, B_pop + b_s (fixed plus random lag effects; intercepts are not used),
    around a leave-one-out centre: the mean of all other subjects' data plus this
    subject's days 1..W. The matrices are not spectrally clipped.
    """

    def __init__(self, series, items, scale, bpd):
        self.series, self.items, self.scale, self.bpd = series, items, scale, bpd
        grand = np.nanmean(np.vstack(series), axis=0)
        self.grand_mean = np.where(np.isfinite(grand), grand, 0.0)
        self.fits = {}

    def _fit(self, W):
        Xs, Ys, groups = [], [], []
        for i, s in enumerate(self.series):
            X, Y = lag_pairs((s[: W * self.bpd] - self.grand_mean) / self.scale, self.bpd, "within_day")
            Xs.append(X); Ys.append(Y); groups.append(np.full(len(X), i))
        return fit_mlvar_population(np.vstack(Xs), np.vstack(Ys), np.concatenate(groups), self.items)

    def predictor(self, i, W):
        """Forecast function for subject position i at training window W."""
        if W not in self.fits:
            self.fits[W] = self._fit(W)
        fit = self.fits[W]
        B = fit.B_pop + fit.subject_effects(i)[1]
        rows = [s if j != i else s[: W * self.bpd] for j, s in enumerate(self.series)]
        loo = np.nanmean(np.vstack(rows), axis=0)
        return linear_predictor(B, loo, self.scale)
