"""Rowland & Wenzel (2020): 125 students, 8 affect items, 6 prompts/day for 40 days.

Raw: openESM 0006_rowland (see the README). build() reshapes it losslessly into one
240-row grid per subject (no interpolation, no detrending)."""
import numpy as np
import pandas as pd

from emaforecast.data import openesm
from emaforecast.data.dataset import EMADataset

MOOD_LABELS = ['happy', 'excited', 'relaxed', 'satisfied',
               'angry', 'anxious', 'depressed', 'sad']
N_DAYS = 40
N_BEEPS = 6
T_MAX = N_DAYS * N_BEEPS


def build():
    """Rowland & Wenzel (2020) data as the authors' R script uses it: a lossless
    reshape into one 40-day x 6-beep grid per subject (missing beeps stay NaN; no
    interpolation, no detrending, no centring). Time enters only through the day
    and beep numbers."""
    df = openesm.read("rowland").rename(columns={"id": "subjno", "day": "dayno"})
    df = df.sort_values(['subjno', 'dayno', 'beep']).reset_index(drop=True)
    subj_ids = sorted(df['subjno'].unique())

    all_data = []
    for sid in subj_ids:
        sub = df[df['subjno'] == sid]
        arr = np.full((T_MAX, len(MOOD_LABELS)), np.nan)
        for _, row in sub.iterrows():
            day, beep = int(row['dayno']), int(row['beep'])
            if day < 1 or day > N_DAYS or beep < 1 or beep > N_BEEPS:
                continue
            slot = (day - 1) * N_BEEPS + (beep - 1)
            arr[slot, :] = row[MOOD_LABELS].values.astype(float)
        all_data.append(arr)

    return EMADataset(name="rowland", items=MOOD_LABELS,
                      subject_ids=np.array(subj_ids, dtype=int),
                      series=all_data, beeps_per_day=N_BEEPS)
