"""Which lag-1 transitions (y(t-1) -> y(t)) a model is trained on."""
import numpy as np


def lag_pairs(z, beeps_per_day, mode):
    """Lag pairs from one subject's training rows.

    z : (T, K) training rows in time order, already centred and scaled.
    mode
      'sequential' : every consecutive pair of rows, also across nights.
                     Fisher/Reeves: the spline-resampled series is equidistant
                     and has no missing values.
      'within_day' : consecutive beeps of the same day only; a pair is dropped
                     if either beep has a missing item (no interpolation).
                     Rowland/Geschwind, as in the original analyses.
    Returns X (pairs at t-1) and Y (pairs at t), each (n_pairs, K).
    """
    # The memory layout (row- vs column-major) only changes the last bits of LassoCV's
    # results; it is fixed per mode to keep the reference tables bit-identical.
    if mode == "sequential":
        z = np.ascontiguousarray(z)
        return z[:-1], z[1:]
    if mode != "within_day":
        raise ValueError(f"unknown pairs mode {mode!r}")
    t = np.arange(1, len(z))
    same_day = t // beeps_per_day == (t - 1) // beeps_per_day
    complete = ~np.isnan(z[t]).any(axis=1) & ~np.isnan(z[t - 1]).any(axis=1)
    keep = t[same_day & complete]
    return np.asfortranarray(z[keep - 1]), np.asfortranarray(z[keep])
