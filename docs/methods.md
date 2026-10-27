# Methods: how the Figure 2 numbers are computed

Code: `src/emaforecast/protocol/` (evaluation), `src/emaforecast/models/` (models),
settings per dataset in `src/emaforecast/config.py`.

## Rolling-origin evaluation
Each participant is evaluated separately. For every training window of W days
(W = 1, 2, ...), the models are fitted on days 1..W and tested on day W + 1; the window
then grows by one day. A "day" is a block of rows: 4 for Fisher and Reeves (their
resampled series are equidistant), 6 for Rowland, 10 for Geschwind.

**Standardisation.** Within a window, the training rows are centred on the
participant's training mean (days 1..W only) and divided by a global SD per item (one
value per item over all participants and days). A forecast is made in these units and
converted back: forecast = training mean + SD x (model output).

**Lag pairs.** Models are fitted on lag-1 pairs (row t - 1 -> row t):
- Fisher, Reeves: every consecutive pair of the resampled series, overnight included.
- Rowland, Geschwind: only pairs of the same day, and only if every item is observed at
  both prompts (as in the original analyses).

A window is used only if it has at least 9 (Fisher, Reeves) or 10 (Rowland, Geschwind)
pairs. Otherwise no model is fitted and **nothing in that window is scored**, so the
baselines and all models are always compared on the same forecasts. Results are reported
from W = 1 (Fisher, Reeves), 7 (Rowland) and 2 (Geschwind).

## Forecasts
On the test day:
- **one-step**: every consecutive pair of observed prompts (within the day for Rowland
  and Geschwind; for Fisher and Reeves the first pair starts from the last training row).
  Each forecast starts from the actual previous prompt.
- **one-day**: one forecast iterated to the last prompt of the test day: 4 steps from the
  last training row (Fisher, Reeves), or from the first prompt of the test day, 5 steps
  (Rowland) and 9 steps (Geschwind). The forecast is made only if its start and its
  target are fully observed.

## Models
| Model | Fitted on | Description |
|---|---|---|
| persistence | - | the previous prompt (the reference for every ratio) |
| median | participant, days 1..W | the per-item median of the training window |
| AR(1) | participant | one lag-1 regression per item on its own past (diagonal matrix) |
| LASSO | participant | a full VAR(1), one L1-penalised regression per item, penalty chosen by 5-fold cross-validation |
| SEM (Fisher, Reeves) | participant | the stepwise network search of Fisher et al. (2017): start from AR(1), add the cross-lagged path with the largest modification index while it is >= 5 |
| OLS VAR (Rowland) | participant | an unregularised VAR(1), as in Rowland & Wenzel (2020) |
| mlVAR (Geschwind) | all participants | the multilevel VAR of Bringmann et al. (2013), see below |

The per-participant matrices are shrunk if their spectral radius exceeds 1
(B / (radius + 0.01)), so iterated forecasts stay bounded.

**Multilevel VAR.** Fitted once per window on the pooled within-day pairs of all
participants (days 1..W), all centred on one common value (the grand mean) so that
differences in level between people stay in the data: per item, a mixed model with
fixed and random intercepts and lag effects, fitted by maximum likelihood. If the model
with random lag effects does not converge for an item, that item is refitted with a
random intercept only. Each participant is forecast with their individual network, the
fixed plus their random lag effects (as Bringmann et al. define individual networks),
around a leave-one-out centre: the mean of all other participants' data plus this
participant's days 1..W.

## Error and aggregation
- Error of one forecast: mean over items of |actual - forecast| / global SD of the item.
- Per participant and horizon: the mean error over all their forecasts of all reported
  windows (each forecast weighted equally); `results/<dataset>/per_subject_mae.csv`.
- Figure 2: per participant, MAE(model) / MAE(persistence); below 1 = better than
  persistence. Boxes show the distribution over participants. The brackets test AR(1)
  against the dataset's VAR with a paired two-sided Wilcoxon signed-rank test
  (`results/figures/fig2_stats.csv`).
