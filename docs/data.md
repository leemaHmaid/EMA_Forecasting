# Data

The data are **not** included in this repository. All four datasets are taken from
[openESM](https://openesmdata.org/), a database of openly available experience-sampling
datasets, where they are published under **CC BY-NC 4.0**. Download one file per
dataset and place it in `data/`:

| Dataset | openESM record (DOI) | File to place in `data/` | SHA-256 |
|---|---|---|---|
| Fisher et al. (2017) | [0033_fisher](https://doi.org/10.5281/zenodo.17348039) | `0033_fisher_ts.tsv` | `96df875c8974808cd693c71118bb6bba4a17dbb51ce1d05c3e9989dba356ee28` |
| Reeves & Fisher (2020) | [0034_reeves](https://doi.org/10.5281/zenodo.17348020) | `0034_reeves_ts.tsv` | `3c106f61c6e67bce5fc21944d9a5d3d579bb4933faef1cef9652dfb8b889f710` |
| Rowland & Wenzel (2020) | [0006_rowland](https://doi.org/10.5281/zenodo.17347352) | `0006_rowland_ts.tsv` | `fe2355cd8a9b809a7b5c902fd493009c59c1592502b6d22efae53f74bf2b18c5` |
| Geschwind et al. (2011) | [0010_geschwind](https://doi.org/10.5281/zenodo.17347474) | `0010_geschwind_ts.tsv` | `811f41aeaeb8dbfaa67ca8f8a5387d222701c42fc316f9eb3d61326d138bc8ee` |

These are version 1.0.0 of each record. A missing file stops the run with a message
pointing here. `pytest tests/test_data.py` checks that the processed data are exactly
those behind Figure 2.

When using the data, cite the original publications (below) and openESM, as each record
asks.

## The datasets

| Dataset | Participants | Items | Prompts per day | Used |
|---|---|---|---|---|
| Fisher et al. (2017), generalised anxiety / major depression | 40 | 21 | 4 | full record |
| Reeves & Fisher (2020), post-traumatic stress disorder | 20 | 20 | 4 | full record |
| Rowland & Wenzel (2020), students | 125 | 8 | 6 | 40 days |
| Geschwind et al. (2011), residual depressive symptoms | 129 | 6 | 10 | baseline block, 6 days |

The openESM files hold the same item values as the authors' original files (checked
prompt by prompt against Fisher et al.'s R workspaces on OSF, the Reeves participant
files, Rowland & Wenzel's `data_affect.csv` on OSF, and Data S1 of Bringmann et al.,
2013); only column names and the format of the prompt times differ.

## Preprocessing and how time is handled

Fisher and Reeves are preprocessed as in the authors' R code: detrending against elapsed
time, then cubic-spline resampling onto an equidistant grid. Elapsed time therefore
matters for these two datasets. Rowland and Geschwind are used as recorded; time enters
only through the day and prompt numbers.

**Prompt times in openESM.** openESM gives prompt times in UTC, converted from the
authors' local wall-clock times as if these were Europe/Berlin time (for example a
Fisher prompt at 17:55 local time is stored as 15:55 UTC). The participants were in
the US, so these UTC values are not the true UTC times, and elapsed time computed from
them would be off by one hour between the European and US daylight-saving dates. The
loaders therefore convert back with Europe/Berlin, which recovers the authors' local
times exactly (0 differences over all 5,327 Fisher and 2,543 Reeves prompts), and then
follow the original code:

| Dataset | Timestamp path of the original code | Here |
|---|---|---|
| Fisher | Authors' R code (`Complete_R_Code.txt`, lines 18-27): `strptime(start, "%m/%d/%Y %H:%M")` on the local prompt times, `difftime` between consecutive prompts, missing differences set to 0, cumulated. `strptime` uses the computer's time zone, so daylight saving is applied (US calendar; the port reproduces the authors' stored R output, `dirdat`/`dircub`, to 1e-8). | Local times recovered from openESM, placed in a US time zone (`America/Los_Angeles`, daylight saving applied), differences in hours, missing = 0, cumulated. |
| Reeves | Authors' R code (`Idiographic PTSD Networks_Master.R`, lines 84-90): the same `strptime`/`difftime` steps as Fisher, so daylight saving is applied (US calendar). The authors' stored R output is not available for Reeves, so this port is checked against their code, not their output. | The same as Fisher: local times recovered from openESM, US time zone (daylight saving applied), differences in minutes, missing = 0, cumulated; ties broken by 1e-6 min. Daylight saving matters for 7 of the 20 participants, whose recordings span a US clock change. |
| Rowland | Authors' R code (`Syntax1_IndividualNetworks.R`): `mlGraphicalVAR(..., dayvar = "dayno", beepvar = "beep")`, no timestamps. | Day and prompt numbers only (40 x 6 grid). |
| Geschwind | Bringmann et al.'s R code (Appendix S1): lags within the day from day and prompt numbers, no timestamps. | Day and prompt numbers only (6 x 10 baseline grid). |

## Per-dataset details

**Fisher et al. (2017).** The 21 modelled items of the authors' R code (`accepted` and
`threatened` are not modelled); linear detrending against elapsed time; R's "fmm" cubic
spline onto an equidistant grid with as many points as prompts, which also fills missed
prompts. The authors' initial z-scoring is not applied: the forecasting protocol does
its own centring and scaling.

**Reeves & Fisher (2020).** The PCL items are averaged into 20 symptom variables, each
day's sleep answers being copied to all prompts of that day; then detrending and spline
resampling as for Fisher. The 20th variable, "sleep", is the "felt sleepy" rating, as in
the authors' code (their sleep-disturbance composite is computed but not used).

**Rowland & Wenzel (2020).** Reshaped into one 240 x 8 series (40 days x 6 prompts) per
participant; missed prompts stay missing.

**Geschwind et al. (2011)**, as analysed by Bringmann et al. (2013). Only the baseline
block (study period 0), days 1-6 with 10 prompts each; the post-treatment block is not
used, and days 7-10 of the baseline block are essentially empty. Missed prompts stay
missing.

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
  https://doi.org/10.3758/s13428-026-03112-y
