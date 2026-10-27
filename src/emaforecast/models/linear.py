"""Per-subject linear lag-1 models fitted on lag pairs.

Every function takes the same input -- lag pairs X (predictors at t-1) and Y
(targets at t), both (n_pairs, K), already centred and scaled by the protocol --
and returns the K x K transition matrix B, with B[i, j] = effect of item j at
t-1 on item i at t, so that the forecast is  y_hat(t) = B @ y(t-1).

None of these functions decides whether there is "enough" data; that is a
protocol rule (see emaforecast.protocol).
"""
import numpy as np
from sklearn.linear_model import LassoCV


def fit_ar(X, Y):
    """AR(1): one univariate regression per item, B diagonal.

    b_ii = cov(x, y) / var(x), with the covariance computed with n - 1 and the
    variance with n in the denominator, i.e. the least-squares slope (with
    intercept) times n / (n - 1). Items without variance get b_ii = 0.
    """
    K = X.shape[1]
    B = np.zeros((K, K))
    for i in range(K):
        var_x = np.var(X[:, i])
        if var_x > 1e-10:
            B[i, i] = np.cov(X[:, i], Y[:, i])[0, 1] / var_x
    return B


def fit_ols_var(X, Y):
    """Unregularised VAR(1): least squares per equation, no intercept.
    Returns None if the solver fails."""
    K = X.shape[1]
    B = np.zeros((K, K))
    try:
        for i in range(K):
            B[i], *_ = np.linalg.lstsq(X, Y[:, i], rcond=None)
    except Exception:
        return None
    return B


def fit_lasso_cv(X, Y):
    """LASSO-VAR(1): one L1-penalised regression per equation, penalty chosen by
    5-fold cross-validation within the training pairs (sklearn LassoCV, unshuffled
    contiguous folds). The fitted intercept is not used (the data are centred), and
    an equation whose fit fails keeps an all-zero row.
    """
    K = X.shape[1]
    B = np.zeros((K, K))
    for i in range(K):
        try:
            B[i] = LassoCV(cv=5, max_iter=5000).fit(X, Y[:, i]).coef_
        except Exception:
            pass
    return B


def clip_spectral(B):
    """Shrink an explosive transition matrix: if its spectral radius rho > 1,
    return B / (rho + 0.01); otherwise return B unchanged.

    Applied to every fitted per-subject matrix before forecasting, so that
    iterated (one-day) forecasts cannot diverge.
    """
    rho = float(np.max(np.abs(np.linalg.eigvals(B))))
    return B / (rho + 0.01) if rho > 1.0 else B
