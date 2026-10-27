"""Multilevel VAR(1), Bringmann et al. (2013) pseudo-likelihood approach.

Unlike the per-subject models, the mlVAR is fitted ONCE per training window on
the pooled lag pairs of all subjects. One mixed model per outcome item j:

    Y_j(t) = b0_j + sum_k B_pop[j, k] * Y_k(t-1) + u_0s + sum_k u_ks * Y_k(t-1) + e

with a random intercept u_0s and random slopes u_ks on all K lags per subject,
fitted by maximum likelihood (as lme4 with REML = FALSE in Bringmann et al.'s R
code). If the model with random slopes does not converge for an outcome, that
outcome is refitted with a random intercept only. The fixed lag effects form the
population matrix B_pop.

Per subject, the fit also gives the random effects (statsmodels' conditional
modes, equivalent to lme4's ranef()): a random intercept u_0s per item (the
subject's level) and slope deviations u_ks. Bringmann et al.'s individual network
is B_pop + u (fixef + ranef, lag effects only); see MLVARFit.subject_effects.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class MLVARFit:
    B_pop: np.ndarray                                  # K x K population transition matrix
    used_random_slopes: dict                           # item -> True (full) / False (intercept-only fallback)
    results: dict = field(repr=False, default_factory=dict)   # item -> statsmodels MixedLMResults | None

    def subject_effects(self, group):
        """Random effects of one subject: (u0 (K,), b (K, K)).

        u0[j]   = random intercept of item j (deviation of the subject's level),
        b[j, k] = deviation of the effect of item k (t-1) on item j (t).
        statsmodels orders each item's random-effect vector as
        [intercept, lag_1, ..., lag_K]. Items that fell back to a random intercept
        only have no slope deviations (row of zeros); a subject absent from the
        fit gets zeros (= the population model).
        """
        K = len(self.B_pop)
        u0, b = np.zeros(K), np.zeros((K, K))
        for j, r in enumerate(self.results.values()):
            if r is None or group not in r.random_effects:
                continue
            vals = np.asarray(r.random_effects[group], dtype=float)
            u0[j] = vals[0]
            if vals.size >= K + 1:
                b[j] = vals[1:K + 1]
        return u0, b


def fit_mlvar_population(X, Y, groups, items):
    """Fit the population mlVAR on pooled lag pairs.

    X, Y   : (n_pairs, K) lagged predictors / outcomes of all subjects, stacked
    groups : (n_pairs,) subject id of each pair
    items  : the K item names (used as column names in the model formulas)
    """
    K = len(items)
    lags = [f'{v}_lag1' for v in items]
    df = pd.DataFrame(np.column_stack([Y, X]), columns=[*items, *lags])
    df['subject_id'] = np.asarray(groups)

    predictors = ' + '.join(lags)
    B_pop = np.zeros((K, K))
    results, used_full = {}, {}
    for j, outcome in enumerate(items):
        fitted, full = _fit_outcome(df, f'{outcome} ~ {predictors}', '~ ' + predictors, K)
        results[outcome], used_full[outcome] = fitted, full
        if fitted is not None:
            for k, lag in enumerate(lags):
                if lag in fitted.fe_params.index:
                    B_pop[j, k] = float(fitted.fe_params[lag])
    return MLVARFit(B_pop=B_pop, used_random_slopes=used_full, results=results)


def _fit_outcome(df, formula, re_formula, K):
    """Full random slopes first; random intercept only if that fails to converge."""
    try:
        model = smf.mixedlm(formula, df, groups=df['subject_id'], re_formula=re_formula)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = model.fit(reml=False, method='powell', maxiter=800, disp=False)
        if r.converged and np.all(np.isfinite(list(r.params))) and r.cov_re.shape[0] == K + 1:
            return r, True
    except Exception:
        pass
    try:
        model = smf.mixedlm(formula, df, groups=df['subject_id'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = model.fit(reml=False, method='powell', maxiter=800, disp=False)
        if r.converged:
            return r, False
    except Exception:
        pass
    return None, False
