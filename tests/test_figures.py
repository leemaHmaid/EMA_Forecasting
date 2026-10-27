"""
Figure 2 is drawn from the reference tables (tests/reference/fig2/) and must give
the reference summary (median and mean ratio per dataset, horizon and model) and
statistics (AR vs VAR). The PNG itself is not compared here: its bytes depend on
the installed fonts. On the reference machine its md5 is in fig2.png.md5.
"""
import shutil
from pathlib import Path

import pandas as pd

from emaforecast.figures import fig2

REF = Path(__file__).parent / "reference" / "fig2"


def test_fig2_from_reference_tables(tmp_path):
    results = {}
    for ds in fig2.DATASETS:
        folder = tmp_path / "results" / ds["key"]
        folder.mkdir(parents=True)
        shutil.copyfile(REF / f"{ds['key']}_per_subject_mae.csv", folder / "per_subject_mae.csv")
        results[ds["key"]] = folder
    out = tmp_path / "figures"
    fig2.main(out_dir=out, results=results)

    assert (out / "fig2.png").stat().st_size > 0 and (out / "fig2.pdf").stat().st_size > 0
    for name in ("fig2_summary.csv", "fig2_stats.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(out / name), pd.read_csv(REF / name),
                                      check_exact=False, rtol=0, atol=1e-12)
