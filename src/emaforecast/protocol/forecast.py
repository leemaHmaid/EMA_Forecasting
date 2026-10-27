"""Forecasting one test day and scoring it.

Every model is reduced to a `predict(previous, h)` function: given an observed
row `previous` (raw units), return the forecast h steps ahead (raw units).

Two evaluation targets per test day:
  one-step : every consecutive pair of observed rows on the test day; the
             forecast for row i+1 starts from the ACTUAL row i (teacher-forced).
  one-day  : one forecast iterated h steps ahead from a single seed row.
All errors are in units of the global item SD:  mean_k |actual_k - pred_k| / sigma_k.
"""
import numpy as np


def forecast_error(actual, pred, scale):
    return float(np.mean(np.abs((actual - pred) / scale)))


def linear_predictor(B, centre, scale):
    """Iterate a transition matrix in the centred/scaled space it was fitted in."""
    def predict(previous, h):
        state = (previous - centre) / scale
        for _ in range(h):
            state = B @ state
        return state * scale + centre
    return predict


def persistence_predictor():
    """Last observation carried forward (the denominator of every ratio)."""
    return lambda previous, h: previous


def constant_predictor(value):
    """Always predict the same row (the training-window median)."""
    return lambda previous, h: value


def complete(row):
    return not np.isnan(row).any()
