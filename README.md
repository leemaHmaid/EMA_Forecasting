# EMA forecasting

Code for *Cross-Symptom Interactions Fail to Predict Future Symptom States Across Four
Ecological Momentary Assessment Datasets*

## Setup (Python 3.12)
```
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -e .
```
On Windows, clone into a short path (e.g. `C:\code\`): some dependency files exceed the
260-character path limit otherwise.

## Get the data
Download the four files from
[openESM](https://openesmdata.org/) (CC BY-NC 4.0; version 1.0.0 of each record) and put
them in `data/`:

| Dataset | openESM record | File | SHA-256 |
|---|---|---|---|
| Fisher et al. (2017) | [10.5281/zenodo.17348039](https://doi.org/10.5281/zenodo.17348039) | `0033_fisher_ts.tsv` | `96df875c8974808cd693c71118bb6bba4a17dbb51ce1d05c3e9989dba356ee28` |
| Reeves & Fisher (2020) | [10.5281/zenodo.17348020](https://doi.org/10.5281/zenodo.17348020) | `0034_reeves_ts.tsv` | `3c106f61c6e67bce5fc21944d9a5d3d579bb4933faef1cef9652dfb8b889f710` |
| Rowland & Wenzel (2020) | [10.5281/zenodo.17347352](https://doi.org/10.5281/zenodo.17347352) | `0006_rowland_ts.tsv` | `fe2355cd8a9b809a7b5c902fd493009c59c1592502b6d22efae53f74bf2b18c5` |
| Geschwind et al. (2011) | [10.5281/zenodo.17347474](https://doi.org/10.5281/zenodo.17347474) | `0010_geschwind_ts.tsv` | `811f41aeaeb8dbfaa67ca8f8a5387d222701c42fc316f9eb3d61326d138bc8ee` |

A missing file stops the run with a message pointing here. When using the data, please
cite the original publications and openESM (see [References](#references)).

**Preprocessing** (`src/emaforecast/data/`), following each dataset's original analysis:
- **Fisher** (40 participants, 21 items, 4 prompts/day): linear detrending against elapsed
  time, then cubic-spline resampling (R's "fmm" spline) onto an equidistant grid with as
  many points as prompts, as in the authors' R code; their initial z-scoring is left out,
  because the protocol does its own centring and scaling.
- **Reeves** (20 participants, 20 variables, 4 prompts/day): PCL items averaged into 20
  symptom variables, then
  as Fisher. The 20th variable.
- **Rowland** (125 participants, 8 items, 40 days x 6 prompts): used as recorded; missed
  prompts stay missing.
- **Geschwind** (129 participants, 6 items): the baseline block only (6 days x 10 prompts),
  as in Bringmann et al. (2013); missed prompts stay missing. One participant without any
  answer is not included.

## Reproduce Figure 2
```
python -m emaforecast.run --dataset all     # result tables -> results/<dataset>/   (~2 h)
python -m emaforecast.figures.fig2          # figure + summary -> results/figures/   (seconds)
```
The first command evaluates every participant of every dataset; Fisher takes about
50 min, Geschwind (multilevel model) about 10 min. A single dataset: `--dataset fisher`;
a quick check on the first participants: `--dataset rowland --subjects 3`.

Outputs:
- `results/<dataset>/per_subject_mae.csv`: mean error per participant, horizon and model
  (the data of Figure 2); `mae_by_window.csv` and `off_diagonal_norm_by_window.csv` per
  training window.
- `results/figures/fig2.pdf|png`, `fig2_summary.csv` (median and mean ratio to persistence
  per box), `fig2_stats.csv` (AR(1) vs VAR: participants, how often AR(1) has the lower
  error, two-sided Wilcoxon p).


## Layout
```
data/                     put the four openESM files here (not included)
src/emaforecast/
    config.py             paths and the per-dataset evaluation settings
    data/                 loaders and preprocessing, one module per dataset
    models/               AR(1), LASSO-VAR, OLS VAR, stepwise SEM, multilevel VAR
    protocol/             rolling-origin evaluation, forecasts, result tables
    figures/fig2.py       Figure 2
    run.py                command line: run the evaluation
```

## References
- Bringmann, L. F., Vissers, N., Wichers, M., Geschwind, N., Kuppens, P., Peeters, F.,
  Borsboom, D., & Tuerlinckx, F. (2013). A network approach to psychopathology: New
  insights into clinical longitudinal data. *PLoS ONE, 8*(4), e60188.
- Fisher, A. J., Reeves, J. W., Lawyer, G., Medaglia, J. D., & Rubel, J. A. (2017).
  Exploring the idiographic dynamics of mood and anxiety via network analysis.
  *Journal of Abnormal Psychology, 126*(8), 1044-1056.
- Geschwind, N., Peeters, F., Drukker, M., van Os, J., & Wichers, M. (2011). Mindfulness
  training increases momentary positive emotions and reward experience in adults
  vulnerable to depression: A randomized controlled trial. *Journal of Consulting and
  Clinical Psychology, 79*(5), 618-628.
- Reeves, J. W., & Fisher, A. J. (2020). An examination of idiographic networks of
  posttraumatic stress disorder symptoms. *Journal of Traumatic Stress, 33*(1), 84-95.
- Rowland, Z., & Wenzel, M. (2020). Mindfulness and affect-network density: Does
  mindfulness facilitate disengagement from affective experiences in daily life?
  *Mindfulness, 11*, 1253-1266.
- Siepe, B. S., Haslbeck, J. M. B., Kloft, M., Büchner, A., Zhang, Y., Fried, E. I., &
  Heck, D. W. (2026). Introducing openESM: A database of openly available experience
  sampling datasets. *Behavior Research Methods, 58*(8), 240.
