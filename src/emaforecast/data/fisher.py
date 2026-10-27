"""Fisher et al. (2017): 40 adults with GAD/MDD, 21 items, 4 prompts/day.

Raw: openESM 0033_fisher (docs/data.md), item values and prompt times. build() reproduces
the authors' R preprocessing (detrend + R fmm spline resampling, without scale())."""
import numpy as np
import pandas as pd

from emaforecast.data import openesm
from emaforecast.data.dataset import EMADataset
from emaforecast.data.r_spline import fmm_spline_coef, fmm_spline_eval, r_spline_fmm

# R trim=data[,c(3:18,21:24,28)] column order (Complete_R_Code.txt:30),
# expressed as the TSV's (Open-ESM-renamed) column labels. Excludes
# 'accepted' and 'threatened' (collected but deliberately dropped by the
# original authors before modeling); includes 'avoid_people'.
TRIM_COLS_TSV = [
    'energetic', 'enthusiastic', 'content', 'irritable', 'restless',
    'worried', 'guilty', 'afraid', 'anhedonia', 'angry', 'hopeless', 'down',
    'positive', 'fatigue', 'muscle_tension', 'difficulty_concentrating',
    'rumination', 'avoid_activity', 'reassure', 'procrastination',
    'avoid_people',
]

# Column names as R's `dedat`/`trim` objects have them (colnames(dedat) <-
# colnames(trim), Complete_R_Code.txt:32) -- this is what "the R output"'s
# column identity actually is, distinct from the TSV's fuller relabeling.
TRIM_COLS_R = [
    'energetic', 'enthusiastic', 'content', 'irritable', 'restless',
    'worried', 'guilty', 'afraid', 'anhedonia', 'angry', 'hopeless', 'down',
    'positive', 'fatigue', 'tension', 'concentrate', 'ruminate',
    'avoid_act', 'reassure', 'procrast', 'avoid_people',
]

DEFAULT_TZ = 'America/Los_Angeles'  # any US zone with daylight saving time gives
                                     # the same elapsed times as the authors' R
                                     # code: only the DST calendar matters.


def list_participant_ids():
    return sorted(openesm.read("fisher")["id"].unique())


# ---------------------------------------------------------------------------
# Step 0 + Step 1: load, trim, cumulative time (Complete_R_Code.txt:11-30)
# ---------------------------------------------------------------------------

def load_participant(pid, tsv=None):
    """One participant's 21-item series and elapsed time.

    Elapsed time follows the authors' R code (Complete_R_Code.txt:18-27): differences
    of the local prompt times (strptime + difftime, daylight saving applied by the
    US time zone of their computer), missing differences set to 0, cumulated. The
    local times are recovered from openESM's UTC values (openesm.local_times).

    Returns
    -------
    trim : pd.DataFrame, shape (T, 21), columns = TRIM_COLS_R
    cumsumT : np.ndarray, shape (T,) -- hours since the first prompt
    """
    tsv = openesm.read("fisher") if tsv is None else tsv
    sub = tsv[tsv['id'] == pid].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f'no rows for participant {pid}')

    # Complete_R_Code.txt:30  trim=data[,c(3:18,21:24,28)]
    trim = sub[TRIM_COLS_TSV].astype('float64').copy()
    trim.columns = TRIM_COLS_R

    start = openesm.local_times(sub['start']).dt.tz_localize(
        DEFAULT_TZ, ambiguous='infer', nonexistent='shift_forward')
    tdif = start.diff().dt.total_seconds().values / 3600.0
    tdif = np.nan_to_num(tdif, nan=0.0)  # L24 data$tdif[is.na(tdif)] <- 0
    cumsumT = np.cumsum(tdif)            # L27
    return trim, cumsumT


# ---------------------------------------------------------------------------
# Step 2: z-score then detrend, na.exclude-padded (Complete_R_Code.txt:35-55)
# ---------------------------------------------------------------------------

def _zscore_detrend_column(y, cumsumT, apply_scale=True):
    """resid(lm(scale(y) ~ cumsumT, na.action=na.exclude)) if apply_scale,
    else resid(lm(y ~ cumsumT, na.action=na.exclude)).

    scale(): R default centers/scales with na.rm=TRUE mean and a sample sd
    (ddof=1) computed only over non-NA values; NA stays NA. lm(na.exclude):
    fit uses only non-NA rows; residuals are padded back to full length
    with NaN at the excluded rows (row order/positions preserved).

    apply_scale=False skips R's scale() call (Complete_R_Code.txt:35-55).
    That step served the authors' SEM software, not the preprocessing itself;
    the forecasting protocol does its own centring (per training window) and
    scaling (global item SD), which should be the only ones. Detrending
    (OLS residual against cumsumT) is the same either way."""
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    if apply_scale:
        mu = np.mean(y[mask])
        sd = np.std(y[mask], ddof=1)
        y_work = (y - mu) / sd  # NaN stays NaN automatically
    else:
        y_work = y

    x_fit = cumsumT[mask]
    y_fit = y_work[mask]
    X = np.column_stack([np.ones_like(x_fit), x_fit])
    beta, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
    resid_fit = y_fit - X @ beta

    resid_full = np.full(len(y), np.nan)
    resid_full[mask] = resid_fit
    return resid_full


def zscore_detrend(trim, cumsumT, apply_scale=True):
    """Apply _zscore_detrend_column to every column of `trim`.
    Returns a DataFrame same shape/columns as `trim` (R's `dedat`)."""
    out = {col: _zscore_detrend_column(trim[col].values, cumsumT, apply_scale=apply_scale)
           for col in trim.columns}
    return pd.DataFrame(out, columns=trim.columns)


# ---------------------------------------------------------------------------
# Step 3: FMM cubic spline resample (Complete_R_Code.txt:57-79)
# ---------------------------------------------------------------------------

def spline_resample(dedat, cumsumT):
    """spline(x=cumsumT, y=dedat[,i], nrow(data), method='fmm')$y for every
    column. NaN (x,y) pairs are dropped before fitting (R's internal
    regularize.values behavior), then evaluated at n = len(cumsumT) points
    evenly spaced over [min(cumsumT), max(cumsumT)] -- NOT a fixed 6-hour
    step; R's third positional arg to spline() is a point COUNT, not a
    stride. Returns a DataFrame, same columns as `dedat`, same row count
    as the input (R's `datcub`/`dircub`, unnamed in R itself but given
    `dedat`'s names here as the natural per-timepoint variable identity)."""
    n = len(cumsumT)
    out = {}
    for col in dedat.columns:
        y = dedat[col].values
        mask = ~np.isnan(y)
        _, y_grid = r_spline_fmm(cumsumT[mask], y[mask], n)
        out[col] = y_grid
    return pd.DataFrame(out, columns=dedat.columns)


# ---------------------------------------------------------------------------
# Build the dataset
# ---------------------------------------------------------------------------

def build():
    """Fisher et al. (2017) preprocessing as in the authors' R code: linear
    detrend against real elapsed time, then R 'fmm' cubic-spline resampling
    onto an equidistant grid with as many points as raw observations -- without
    the R scale() step. Also returns the calendar-day start of each grid row."""
    tsv = openesm.read("fisher")
    pids = sorted(tsv["id"].unique())

    all_data = []
    all_bounds = []
    kept_ids = []
    for pid in pids:
        trim, cumsumT = load_participant(pid, tsv)
        dedat = zscore_detrend(trim, cumsumT, apply_scale=False)
        dircub = spline_resample(dedat, cumsumT)
        arr = dircub[TRIM_COLS_R].values.astype(float)
        assert arr.shape[1] == 21, f'{pid}: expected 21 columns, got {arr.shape[1]}'

        # Calendar-day starts of the resampled grid. spline_resample() evaluates
        # each column on np.linspace(cumsumT[mask].min(), cumsumT[mask].max(), n),
        # where mask is THAT column's non-NaN rows -- not the raw cumsumT range.
        # 12/40 participants have a leading/trailing fully skipped survey that
        # shrinks this range by up to ~5 h. All 21 columns agree on the trimmed
        # range per participant (asserted below), so one shared grid is valid,
        # but it must be the trimmed one.
        n = len(cumsumT)
        col_grids = set()
        for col in dedat.columns:
            m = ~np.isnan(dedat[col].values)
            if m.sum() > 0:
                col_grids.add((round(float(cumsumT[m].min()), 9),
                               round(float(cumsumT[m].max()), 9)))
        if len(col_grids) != 1:
            raise RuntimeError(
                f'{pid}: columns disagree on spline grid extent ({col_grids}) -- '
                f'single-shared-grid assumption violated, day boundaries would '
                f'be ambiguous per-column')
        grid_min, grid_max = col_grids.pop()
        t_regular = np.linspace(grid_min, grid_max, n)
        day_idx = np.floor(t_regular / 24.0).astype(int)
        bounds = np.zeros(n, dtype=bool)
        bounds[0] = True
        bounds[1:] = day_idx[1:] != day_idx[:-1]

        all_data.append(arr)
        all_bounds.append(bounds)
        kept_ids.append(pid)

    return EMADataset(name="fisher", items=TRIM_COLS_R, subject_ids=np.array(kept_ids),
                      series=all_data, beeps_per_day=4, day_starts=all_bounds)
