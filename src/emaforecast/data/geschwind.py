"""Geschwind et al. (2011), analysed by Bringmann et al. (2013): 129 adults with
residual depressive symptoms, 6 items, 10 prompts/day; baseline period (days 1-6) used.

Raw: openESM 0010_geschwind (docs/data.md), built from Bringmann et al. (2013), Data S1."""
import numpy as np
import pandas as pd

from emaforecast.data import openesm
from emaforecast.data.dataset import EMADataset

# Canonical variable order used throughout this module
GSCHWIND_VARS = ['cheerful', 'pleasant', 'worry', 'fearful', 'sad', 'relaxed']

# openESM column names -> the labels used here
RAW_TO_LABEL = {
    'cheerful': 'cheerful',
    'pleasantness': 'pleasant',
    'worried': 'worry',
    'fearful': 'fearful',
    'sad': 'sad',
    'relaxed': 'relaxed',
}

BEEPS_PER_DAY = 10
BASELINE_DAYS = 6   # the baseline block has 6 days (later days are essentially empty)

# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw(period='pre'):
    """
    Load and minimally clean the openESM file.

    Parameters
    ----------
    period : 'pre' | 'post' | 'both'
        'pre'  -> study_period == 0, days 1-6 (baseline, used by Bringmann 2013)
        'post' -> study_period == 1
        'both' -> all rows

    Returns
    -------
    pd.DataFrame with columns:
        subject_id (int, 0-indexed), day (1-indexed), beep (1-indexed),
        cheerful, pleasant, worry, fearful, sad, relaxed
    Time enters only through the day and beep numbers.
    """
    df = openesm.read("geschwind")

    period_map = {'pre': 0, 'post': 1}
    if period in period_map:
        df = df[df['study_period'] == period_map[period]].copy()

    # Drop rows without day/beep identifiers
    df = df.dropna(subset=['day', 'beep'])
    df['day'] = df['day'].astype(int)
    df['beep'] = df['beep'].astype(int)
    if period == 'pre':
        df = df[df['day'] <= BASELINE_DAYS]

    df = df.rename(columns=RAW_TO_LABEL)

    # Compact 0-indexed subject id
    subj_ids = sorted(df['id'].unique())
    df['subject_id'] = df['id'].map({s: i for i, s in enumerate(subj_ids)})
    df = df[['subject_id', 'day', 'beep'] + GSCHWIND_VARS].copy()

    # Drop subjects without any observed answer, then re-compact the ids
    valid_subjs = (df.groupby('subject_id')[GSCHWIND_VARS]
                     .apply(lambda x: x.notna().any().any()))
    valid_subjs = valid_subjs[valid_subjs].index
    df = df[df['subject_id'].isin(valid_subjs)].copy()
    id_remap = {old: new for new, old in enumerate(sorted(valid_subjs))}
    df['subject_id'] = df['subject_id'].map(id_remap)
    return df


def build(period='pre'):
    """Baseline period as an EMADataset: every subject has a complete
    6-day x 10-beep grid (missing answers are NaN, never missing rows)."""
    df = load_raw(period=period).sort_values(['subject_id', 'day', 'beep'])
    groups = df.groupby('subject_id', sort=True)
    series = [g[GSCHWIND_VARS].to_numpy(dtype=float) for _, g in groups]
    assert all(len(s) == BASELINE_DAYS * BEEPS_PER_DAY for s in series), 'incomplete day x beep grid'
    return EMADataset(name="geschwind", items=GSCHWIND_VARS,
                      subject_ids=np.array(list(groups.groups), dtype=int),
                      series=series, beeps_per_day=BEEPS_PER_DAY)
