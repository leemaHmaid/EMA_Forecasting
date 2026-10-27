"""Forecasting models.

Per-subject models share one signature,  fit(X, Y) -> B  (lag pairs in, K x K
transition matrix out), so the protocol can treat them interchangeably via
PER_SUBJECT. The mlVAR is the exception: it is fitted once per window on all
subjects' pooled pairs (fit_mlvar_population).
"""
from emaforecast.models.linear import clip_spectral, fit_ar, fit_lasso_cv, fit_ols_var
from emaforecast.models.mlvar import MLVARFit, fit_mlvar_population
from emaforecast.models.sem import fit_sem_stepwise

PER_SUBJECT = {
    "ar": fit_ar,
    "lasso": fit_lasso_cv,
    "var_ols": fit_ols_var,
    "sem": fit_sem_stepwise,
}

__all__ = ["PER_SUBJECT", "fit_ar", "fit_lasso_cv", "fit_ols_var", "fit_sem_stepwise",
           "clip_spectral", "fit_mlvar_population", "MLVARFit"]
