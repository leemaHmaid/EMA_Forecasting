"""Reading the openESM files (one tab-separated file per dataset, docs/data.md).

openESM stores prompt times in UTC, converted from the authors' local wall-clock times
as if those were Europe/Berlin time. Converting back with the same zone recovers the
authors' original local times exactly (checked against their files for every prompt of
Fisher and Reeves); local_times() does that.
"""
import pandas as pd

from emaforecast import config

SOURCE_TZ = "Europe/Berlin"


def read(dataset):
    """The dataset's openESM file as a DataFrame ('NA' = missing)."""
    return pd.read_csv(config.raw_file(dataset), sep="\t")


def local_times(utc):
    """openESM UTC timestamps -> the authors' local wall-clock times (naive)."""
    return pd.to_datetime(utc, utc=True).dt.tz_convert(SOURCE_TZ).dt.tz_localize(None)
