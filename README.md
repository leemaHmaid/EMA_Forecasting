# EMA forecasting

Code for *Cross-Symptom Interactions Fail to Predict Future Symptom States Across Four
Ecological Momentary Assessment Datasets*: a rolling-origin forecasting benchmark of
AR(1), LASSO-VAR and each dataset's original network model (stepwise SEM, OLS VAR,
multilevel VAR) on four public EMA datasets. This repository reproduces Figure 2.

## Setup (Python 3.12)
```
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -e ".[dev]"
```
On Windows, clone into a short path (e.g. `C:\code\`): some dependency files exceed the
260-character path limit otherwise.

## Get the data
The data are not included. Download the four openESM files (CC BY-NC 4.0) and put them
in `data/`:

| Dataset | openESM record | File |
|---|---|---|
| Fisher et al. (2017) | [10.5281/zenodo.17348039](https://doi.org/10.5281/zenodo.17348039) | `0033_fisher_ts.tsv` |
| Reeves & Fisher (2020) | [10.5281/zenodo.17348020](https://doi.org/10.5281/zenodo.17348020) | `0034_reeves_ts.tsv` |
| Rowland & Wenzel (2020) | [10.5281/zenodo.17347352](https://doi.org/10.5281/zenodo.17347352) | `0006_rowland_ts.tsv` |
| Geschwind et al. (2011) | [10.5281/zenodo.17347474](https://doi.org/10.5281/zenodo.17347474) | `0010_geschwind_ts.tsv` |

Checksums, citations and the preprocessing of each dataset: [docs/data.md](docs/data.md).

**Prompt times.** Only Fisher and Reeves use prompt times (for detrending and
resampling); Rowland and Geschwind use day and prompt numbers only. openESM stores the
authors' local prompt times in UTC as if they were Europe/Berlin time; the loaders
convert them back, which recovers the authors' local times exactly. Elapsed time is
then computed as in the authors' R code for both datasets: differences of the local
prompt times in a US time zone, so daylight saving is applied. Details per dataset:
[docs/data.md](docs/data.md).

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
  error, Wilcoxon p).

What the numbers mean, step by step: [docs/methods.md](docs/methods.md).

## Tests
```
pytest                  # data, models, Figure 2 tables on a few participants (~10 min)
pytest -m slow          # Figure 2 tables for every participant + the multilevel model (~2 h)
```
The reference tables in `tests/reference/fig2/` are those Figure 2 is drawn from; every
run must reproduce them to 1e-12. The processed data are checked against fingerprints
(`tests/reference/data_fingerprints.json`), so the tests need the data in `data/`; without
it they are skipped and only the data-free tests run.

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
tests/                    tests and reference outputs
docs/                     methods and data
```
