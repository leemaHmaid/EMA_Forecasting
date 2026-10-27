"""From individual forecasts to the three result tables behind Figures 2 and 3."""
import numpy as np
import pandas as pd

BASE_MODELS = ["last_step", "median", "ar", "lasso"]


def per_subject_mae(forecasts, protocol):
    """One row per subject and horizon: mean error over ALL that subject's forecasts
    (pooled over windows >= min_window, each forecast weighted equally). Subjects
    without both AR and VAR forecasts are dropped. This is the table behind Fig 2."""
    var = protocol.var_model
    reported = forecasts[forecasts["window_days"] >= protocol.min_window]
    parts = []
    for horizon in ("one_step", "one_day"):
        psm = (reported[reported["horizon"] == horizon]
               .groupby(["subject_id", "model"])["mae"].mean().unstack())
        for m in BASE_MODELS + [var]:
            if m not in psm.columns:
                psm[m] = np.nan
        psm = psm.dropna(subset=["ar", var])
        part = pd.DataFrame({"subject_id": psm.index, "horizon": horizon,
                             **{f"mae_{m}": psm[m].values for m in BASE_MODELS + [var]}})
        part["ar_over_var_ratio"] = part["mae_ar"] / part[f"mae_{var}"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def mae_by_window(forecasts, protocol):
    """One-step error per subject, model and training window (Fig 3, top row)."""
    one_step = forecasts[(forecasts["horizon"] == "one_step") &
                         (forecasts["window_days"] >= protocol.min_window)]
    g = one_step.groupby(["subject_id", "model", "window_days"])["mae"]
    out = pd.DataFrame({"mae": g.mean(), "n_test_forecasts": g.size()}).reset_index()
    return out.rename(columns={"window_days": "training_window_days"})


def off_diagonal_norm(lasso_B, protocol):
    """Frobenius norm of the off-diagonal (cross-symptom) part of the LASSO matrix
    (before spectral clipping) per subject and window, also normalised to the
    subject's first window (Fig 3, bottom row)."""
    rows = []
    for (sid, W), B in lasso_B.items():
        if W >= protocol.min_window:
            off = B - np.diag(np.diag(B))
            rows.append((sid, W, float(np.sqrt((off ** 2).sum()))))
    out = (pd.DataFrame(rows, columns=["subject_id", "training_window_days", "frobenius_off_diagonal"])
           .sort_values(["subject_id", "training_window_days"]))
    first = out.groupby("subject_id")["frobenius_off_diagonal"].transform("first")
    out["frobenius_off_diagonal_normalized"] = out["frobenius_off_diagonal"] / first
    return out
