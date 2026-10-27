"""Common in-memory format for all EMA datasets."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class EMADataset:
    name: str
    items: list[str]
    subject_ids: np.ndarray            # str for Fisher/Reeves (file ids), int for Rowland/Geschwind
    series: list[np.ndarray]           # one (T_i, K) float array per subject, rows in time order, NaN = missing
    beeps_per_day: int
    day_starts: list[np.ndarray] | None = None   # Fisher only: True at the first row of each calendar day
    extra: dict = field(default_factory=dict)

    def to_long(self) -> pd.DataFrame:
        """One row per observation: subject_id, t, day, beep, <items>.
        day/beep are positional (row // beeps_per_day). For Fisher/Reeves that is the
        4-row block used by the forecasting protocol, NOT the calendar day (see day_starts)."""
        frames = []
        for sid, arr in zip(self.subject_ids, self.series):
            t = np.arange(len(arr))
            df = pd.DataFrame(arr, columns=self.items)
            df.insert(0, "beep", t % self.beeps_per_day + 1)
            df.insert(0, "day", t // self.beeps_per_day + 1)
            df.insert(0, "t", t)
            df.insert(0, "subject_id", sid)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
