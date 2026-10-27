"""Reeves & Fisher (2020): 20 adults with PTSD, 20 symptom variables, 4 prompts/day.

Raw: openESM 0034_reeves (see the README). build() reproduces the authors' R preprocessing
(composites, detrend + R fmm spline resampling, without scale())."""
import numpy as np
import pandas as pd
from scipy.stats import linregress

from emaforecast.data import openesm
from emaforecast.data.dataset import EMADataset
from emaforecast.data.r_spline import fmm_spline_coef, fmm_spline_eval



ORIG_COLS = [
    'start', 'finish',
    'enthus', 'hyper', 'agg', 'fear', 'self', 'cogav', 'relive', 'dreams',
    'mem', 'TST', 'SQ', 'SOL', 'horror', 'angry', 'guilt', 'shame',
    'other', 'world', 'blameS', 'blameO', 'behav', 'upset', 'phys',
    'amnesia', 'anhed', 'distant', 'difpos', 'reck', 'startle', 'concen',
    'sleepy', 'positive', 'happy', 'content', 'calm', 'fatigue', 'tense',
]
US_TZ = 'America/Los_Angeles'   # as for Fisher: any US zone with daylight saving time gives
                                 # the same elapsed times; only the DST calendar matters

# openESM column names, in the order of ORIG_COLS[2:]
OPENESM_COLS = [
    'felt_enthusiastic', 'hypervigilant', 'aggressive', 'afraid', 'neg_thoughts_self',
    'avoidance', 'reliving', 'unpleasant_dreams', 'intrusions', 'sleep_duration',
    'restless_sleep', 'sleep_troubles', 'horrified', 'angry', 'guilty', 'ashamed',
    'neg_thoughts_others', 'neg_thoughts_world', 'self_blame', 'others_blame',
    'avoid_places', 'upset', 'physical_reactions', 'amnesia', 'anhedonia', 'distant',
    'difficulty_positive', 'reckless', 'startled', 'difficulty_concentrating', 'sleepy',
    'positive', 'happy', 'content', 'calm', 'fatigued', 'muscle_tension',
]
SLEEP_DUP_COLS = ['dreams', 'SQ', 'SOL']
COMPOSITE_DEFS = {
    'neg':    ['fear', 'horror', 'angry', 'guilt', 'shame'],
    'blame':  ['blameS', 'blameO'],
    'belief': ['self', 'other', 'world'],
    'sleep':  ['SQ', 'SOL'],
}
PTSD_ITEM_COLS = [
    'mem', 'dreams', 'relive', 'upset', 'phys', 'cogav', 'behav', 'amnesia',
    'belief', 'blame', 'neg', 'anhed', 'distant', 'difpos',
    'agg', 'reck', 'hyper', 'startle', 'concen', 'sleepy',
]

# The 20th variable, 'sleep', is the raw 'sleepy' rating (how sleepy the participant felt), as in
# the authors' R code; the sleep-disturbance composite of SQ and SOL above is not used.
PTSD_NAMES = [
    'intru', 'dreams', 'relive', 'upset', 'phys', 'cogav', 'behav', 'amnesia',
    'beliefs', 'blame', 'neg', 'anhed', 'distant', 'difpos', 'agg', 'reck',
    'hyper', 'startle', 'concen', 'sleep',
]
N_ITEMS = len(PTSD_ITEM_COLS)



def participants():
    """(id, DataFrame) per participant in id order, with the columns ORIG_COLS and the
    prompt times as the authors' local wall-clock times (openesm.local_times)."""
    raw = openesm.read("reeves")
    raw["pid"] = raw["id"].str.split("-").str[0]          # e.g. 'C032-PhoneSurveys' -> 'C032'
    for pid, g in sorted(raw.groupby("pid"), key=lambda kv: kv[0]):
        df = pd.DataFrame({"start": openesm.local_times(g["creation_date"]).to_numpy(),
                           "finish": openesm.local_times(g["completion_date"]).to_numpy()})
        for src, dst in zip(OPENESM_COLS, ORIG_COLS[2:]):
            df[dst] = g[src].to_numpy(dtype=float)
        yield pid, df


def load_and_prepare(df):
    """Steps 1-5 of the authors' R preprocessing for one participant: copy each day's
    sleep answers to every prompt of that day, average the PCL items into composites,
    and compute elapsed time. Returns (ptsd, cumsum_t), or None if there are fewer
    than 5 prompts.

    Elapsed time follows the authors' R code (Idiographic PTSD Networks_Master.R,
    lines 84-90): differences of the local prompt times (strptime + difftime, daylight
    saving applied by the US time zone of their computer), missing differences set to 0,
    cumulated; ties are broken by 1e-6 min so that the spline grid is strictly
    increasing."""
    df = df.copy()
    n_rows = len(df)
    if n_rows < 5:
        print(f"  WARNING: only {n_rows} rows, skipping")
        return None

    start_dt = pd.to_datetime(df['start'])
    df['_date'] = start_dt.dt.date

    for col in SLEEP_DUP_COLS:
        df[col] = df.groupby('_date')[col].transform(lambda x: x.iloc[0])

    for comp_name, src_cols in COMPOSITE_DEFS.items():
        df[comp_name] = df[src_cols].mean(axis=1, skipna=False)

    start_us = start_dt.dt.tz_localize(US_TZ, ambiguous='infer', nonexistent='shift_forward')
    tdif_min = start_us.diff().dt.total_seconds().div(60.0)
    tdif_min.iloc[0] = 0.0
    tdif_min = tdif_min.fillna(0.0)
    cumsum_t = tdif_min.cumsum().values

    for i in range(1, n_rows):
        if cumsum_t[i] <= cumsum_t[i - 1]:
            cumsum_t[i] = cumsum_t[i - 1] + 1e-6

    ptsd = df[PTSD_ITEM_COLS].values.astype(float)
    return ptsd, cumsum_t


def preprocess_participant(df):
    """One participant: steps 1-5 as in the authors' R code (load_and_prepare),
    step 6 linear detrend against real elapsed time WITHOUT the R scale() step,
    step 7 R 'fmm' cubic-spline resampling (not scipy's not-a-knot)."""
    result = load_and_prepare(df)
    if result is None:
        return None
    ptsd, cumsum_t = result
    n_rows = len(cumsum_t)

    # Step 6: detrend only -- resid(lm(x ~ cumsumT)), no scale()
    detrended = np.full_like(ptsd, np.nan)
    for j in range(N_ITEMS):
        col = ptsd[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 3:
            continue
        x_valid = cumsum_t[valid]
        y_valid = col[valid]
        slope, intercept, *_ = linregress(x_valid, y_valid)
        detrended[valid, j] = y_valid - (slope * x_valid + intercept)

    # Step 7: FMM cubic spline (R's method='fmm'), not scipy not-a-knot
    t_min, t_max = cumsum_t[0], cumsum_t[-1]
    t_regular = np.linspace(t_min, t_max, n_rows)

    cube = np.full((n_rows, N_ITEMS), np.nan)
    for j in range(N_ITEMS):
        col = detrended[:, j]
        valid = ~np.isnan(col)
        t_v = cumsum_t[valid]
        y_v = col[valid]
        _, unique_idx = np.unique(t_v, return_index=True)
        t_v = t_v[unique_idx]
        y_v = y_v[unique_idx]
        if len(t_v) < 4:
            continue
        b, c, d = fmm_spline_coef(t_v, y_v)
        cube[:, j] = fmm_spline_eval(t_v, b, c, d, y_v, t_regular)

    return cube



def build():
    """Reeves & Fisher (2020) preprocessing as in the authors' R code (the
    Fisher et al. 2017 procedure): 28 PCL-5 items averaged into 20 symptom
    variables, linear detrend against real elapsed time, R 'fmm' cubic-spline
    resampling -- without the R scale() step."""
    all_data, subject_ids = [], []
    for subj_id, df in participants():
        data = preprocess_participant(df)
        if data is None:
            continue
        all_data.append(data)
        subject_ids.append(subj_id)

    return EMADataset(name="reeves", items=PTSD_NAMES, subject_ids=np.array(subject_ids),
                      series=all_data, beeps_per_day=4)
