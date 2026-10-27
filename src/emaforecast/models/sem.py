"""Stepwise network search of Fisher et al. (2017), fast OLS/score-test version.

Start from the autoregressive model (B diagonal). Repeatedly add the single
cross-lagged path with the largest modification index -- the Lagrange-multiplier
(score) statistic, here n * r^2 with r the correlation between the target's
current residual and the candidate predictor after partialling out the target's
included predictors -- while it is >= chi_sq_threshold (5.0 ~ p < .025 for
chi^2_1). Coefficients are re-estimated by OLS (no intercept) after each step.
"""
import numpy as np


def fit_sem_stepwise(X, Y, chi_sq_threshold=5.0, max_iterations=100):
    """Same signature as the linear models: lag pairs in, K x K matrix B out."""
    n, K = Y.shape
    included = np.eye(K, dtype=bool)
    B = np.zeros((K, K))

    # Start: autoregressive paths only
    for i in range(K):
        B[i, i] = np.linalg.lstsq(X[:, i:i + 1], Y[:, i], rcond=None)[0][0]

    for _ in range(max_iterations):
        residuals = Y - X @ B.T
        best_mi, best_path = 0, None
        for i in range(K):                      # target
            for j in range(K):                  # candidate predictor
                if included[i, j]:
                    continue
                pred = X[:, j]
                current = X[:, included[i, :]]
                if current.shape[1] > 0:        # partial out the included predictors
                    pred = pred - current @ np.linalg.lstsq(current, pred, rcond=None)[0]
                if np.std(pred) > 1e-10:
                    r = np.corrcoef(residuals[:, i], pred)[0, 1]
                    if not np.isnan(r):
                        mi = n * r ** 2
                        if mi > best_mi:
                            best_mi, best_path = mi, (i, j)

        if best_mi < chi_sq_threshold or best_path is None:
            break
        i, _j = best_path
        included[i, _j] = True
        B[i] = _refit_row(X, Y[:, i], included[i])

    # Final refit of every equation
    for i in range(K):
        if included[i].any():
            B[i] = _refit_row(X, Y[:, i], included[i])
    return B


def _refit_row(X, y, mask):
    """OLS of y on the included predictors; returns a full-length coefficient row."""
    cols = np.where(mask)[0]
    beta = np.linalg.lstsq(X[:, cols], y, rcond=None)[0]
    row = np.zeros(X.shape[1])
    row[cols] = beta
    return row
