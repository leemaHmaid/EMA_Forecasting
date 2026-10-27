"""Figure 2: forecast error of every model relative to persistence, per dataset.

    python -m emaforecast.figures.fig2        # after: python -m emaforecast.run --dataset all

Layout: 2 x 2 datasets (Fisher | Reeves ; Rowland | Geschwind). Each dataset has
two sub-panels, one-step and one-day; each sub-panel shows, per model, the
per-subject ratio  MAE(model) / MAE(persistence)  as a boxplot with the individual
subjects (dots) and the mean (diamond). Ratios below 1 mean the model beats
persistence. Models: the training-window median, AR(1), LASSO-VAR and each
dataset's original network model (stepwise SEM for Fisher/Reeves, OLS VAR for
Rowland, the individual multilevel-VAR networks for Geschwind).

The one-step panels carry a bracket for the paired two-sided Wilcoxon signed-rank
test of AR(1) against the dataset's VAR (*** p < .001, ** p < .01, * p < .05).

Reads results/<dataset>/per_subject_mae.csv; writes to results/figures/:
    fig2.pdf, fig2.png
    fig2_summary.csv   median and mean ratio per dataset, horizon and model
    fig2_stats.csv     AR(1) vs VAR per dataset and horizon: subjects, how often AR
                       has the lower error, median ratios, Wilcoxon p (two-sided)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import wilcoxon

from emaforecast import config
from emaforecast.figures.style import set_pub_style

YLIM = (0.65, 1.5)          # Fisher, Reeves
YLIM_BOTTOM = (0.4, 1.9)    # Rowland, Geschwind
YTICKS = [0.4, 0.8, 1.0, 1.4, 1.9]
BLUE_GRADIENT = [plt.cm.Blues(x) for x in (0.30, 0.52, 0.74, 0.96)]

# order: Fisher, Reeves (top); Rowland, Geschwind (bottom)
DATASETS = [
    dict(key='fisher',    name='Fisher',    var_col='mae_sem',       var_label='VAR\n(SEM)',
         subtitle='(GAD / MDD)'),
    dict(key='reeves',    name='Reeves',    var_col='mae_sem',       var_label='VAR\n(SEM)',
         subtitle='(PTSD)'),
    dict(key='rowland',   name='Rowland',   var_col='mae_var_ols',   var_label='VAR\n(OLS)',
         subtitle='(Non-clinical)'),
    dict(key='geschwind', name='Geschwind', var_col='mae_mlvar_ind', var_label='VAR\n(mlVAR ind.)',
         subtitle='(Residual depressive symptoms)'),
]
MODEL_ORDER = ['Median', 'AR(1)', 'LASSO', 'VAR']           # 'VAR' = each dataset's var_col
MODEL_COL = {'Median': 'mae_median', 'AR(1)': 'mae_ar', 'LASSO': 'mae_lasso'}
HORIZONS = [('one-step', 'one_step'), ('one-day', 'one_day')]


def load_ds(folder):
    return pd.read_csv(folder / 'per_subject_mae.csv')


def ratios(df, horizon, mae_col):
    """Per-subject MAE(model) / MAE(persistence) for one horizon."""
    sub = df[df['horizon'] == horizon]
    num = sub[mae_col].values.astype(float)
    base = sub['mae_last_step'].values.astype(float)
    mask = np.isfinite(num) & np.isfinite(base) & (base > 0)
    return num[mask] / base[mask]


def ar_vs_var(df, var_col, horizon):
    """Paired comparison of per-subject AR and VAR errors: (n, AR lower in, p), with p
    from the two-sided Wilcoxon signed-rank test."""
    sub = df[df['horizon'] == horizon]
    a = sub['mae_ar'].values.astype(float)
    v = sub[var_col].values.astype(float)
    m = np.isfinite(a) & np.isfinite(v)
    a, v = a[m], v[m]
    p = np.nan
    if len(a) >= 1 and not np.all(a == v):
        try:
            p = float(wilcoxon(a, v, alternative='two-sided').pvalue)
        except ValueError:
            pass
    return len(a), int((a < v).sum()), p


def _upper_whisker(a):
    """Upper whisker of a matplotlib boxplot: largest datum <= Q3 + 1.5 IQR."""
    a = np.asarray(a, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    hi = q3 + 1.5 * (q3 - q1)
    within = a[a <= hi]
    return within.max() if within.size else a.max()


def _sig_stars(p):
    if not np.isfinite(p):
        return 'n.s.'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def add_ar_var_bracket(ax, span_ratios, p, x1=1, x2=3, y_pad_frac=0.02, h_frac=0.02):
    """Significance bracket between the AR(1) (x=1) and VAR (x=3) boxes, placed
    above the tallest whisker of every box it spans (AR(1), LASSO, VAR)."""
    arrs = [a for a in span_ratios if len(a) > 0]
    if len(arrs) < 2:
        return
    top = max(_upper_whisker(a) for a in arrs)
    ylo, yhi = ax.get_ylim()
    span = yhi - ylo
    y = top + span * y_pad_frac
    h = span * h_frac
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=0.8, color='black', clip_on=False, zorder=8)
    ax.text((x1 + x2) / 2.0, y + h, _sig_stars(p), ha='center', va='bottom',
            fontsize=10, zorder=8)


def draw_subpanel(ax, ratios_by_model, xlabels, show_yticklabels, ref_label, ylim):
    for xi, (label, xlab) in enumerate(zip(MODEL_ORDER, xlabels)):
        rat = ratios_by_model[label]
        if len(rat) == 0:
            continue
        col = BLUE_GRADIENT[xi]
        bp = ax.boxplot(
            rat, positions=[xi], widths=0.62, patch_artist=True,
            medianprops=dict(color='black', linewidth=0.9),
            whiskerprops=dict(color='0.35', linewidth=0.7),
            capprops=dict(color='0.35', linewidth=0.7),
            flierprops=dict(marker=''),
            boxprops=dict(facecolor=col, edgecolor='0.25', linewidth=0.5), zorder=3)
        for med in bp['medians']:
            med.set_zorder(6)
        rng = np.random.RandomState(42 + xi * 7)
        jitter = rng.uniform(-0.16, 0.16, size=len(rat))
        ax.scatter(xi + jitter, rat, color='black', alpha=0.28, s=3.2,
                   linewidths=0, zorder=4, clip_on=True)
        ax.plot(xi, np.mean(rat), marker='D', markersize=3.2, markerfacecolor='white',
                markeredgecolor='#111111', markeredgewidth=0.6, zorder=7, linestyle='none')

    ref = ax.axhline(1.0, color='#333333', linestyle='--', lw=1.0,
                     label='Persistence = 1.0', zorder=1)
    # set the ticks first and the limits last, so no tick can widen the view
    ax.set_yticks([t for t in YTICKS if ylim[0] <= t <= ylim[1]])
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.yaxis.grid(True, alpha=0.16, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    if not show_yticklabels:
        ax.set_yticklabels([])
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(xlabels, fontsize=6.8, linespacing=0.95)
    ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
    ax.tick_params(axis='x', which='both', length=0, pad=2)
    ax.tick_params(axis='y', labelsize=7)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if ref_label:
        leg = ax.legend(handles=[ref], loc='upper left', fontsize=6.2, frameon=True,
                        handlelength=1.5, borderaxespad=0.4, borderpad=0.5)
        fr = leg.get_frame()
        fr.set_facecolor('#f2f2f2'); fr.set_edgecolor('0.75'); fr.set_linewidth(0.5); fr.set_alpha(0.9)
        for t in leg.get_texts():
            t.set_color('#333333')


def main(out_dir=None, results=None):
    """Draw Figure 2. `results` maps dataset -> folder with per_subject_mae.csv
    (default config.RESULTS); `out_dir` defaults to config.FIGURES."""
    out_dir = config.FIGURES if out_dir is None else out_dir
    results = config.RESULTS if results is None else results
    out_dir.mkdir(parents=True, exist_ok=True)
    set_pub_style(serif=True)
    np.random.seed(42)

    fig = plt.figure(figsize=(7.2, 5.4))
    subfigs = fig.subfigures(2, 2, wspace=0.04, hspace=0.04, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0])
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]

    summary, stats = [], []
    for ds, (r, c) in zip(DATASETS, order):
        df = load_ds(results[ds['key']])
        sf = subfigs[r][c]
        axs = sf.subplots(1, 2, gridspec_kw={'wspace': 0.08})
        sf.subplots_adjust(top=0.78, bottom=0.14, left=0.16 if c == 0 else 0.10, right=0.98)
        xlabels = ['Median', 'AR(1)', 'LASSO', ds['var_label']]
        bottom_row = ds['key'] in ('rowland', 'geschwind')
        panel_ylim = YLIM_BOTTOM if bottom_row else YLIM

        for j, (h_label, h_key) in enumerate(HORIZONS):
            ax = axs[j]
            ratios_by_model = {}
            for role in MODEL_ORDER:
                col = ds['var_col'] if role == 'VAR' else MODEL_COL[role]
                rat = ratios(df, h_key, col)
                ratios_by_model[role] = rat
                summary.append(dict(dataset=ds['name'], horizon=h_key, model=role,
                                    mae_col=col, n=len(rat),
                                    median_ratio=float(np.median(rat)) if len(rat) else np.nan,
                                    mean_ratio=float(np.mean(rat)) if len(rat) else np.nan))
            n, ar_lower, p = ar_vs_var(df, ds['var_col'], h_key)
            stats.append(dict(dataset=ds['name'], horizon=h_key, var_model=ds['var_col'][4:],
                              n_subjects=n, ar_lower_error=ar_lower,
                              median_ratio_ar=float(np.median(ratios_by_model['AR(1)'])),
                              median_ratio_var=float(np.median(ratios_by_model['VAR'])),
                              wilcoxon_p_two_sided=p))
            draw_subpanel(ax, ratios_by_model, xlabels, show_yticklabels=(j == 0),
                          ref_label=(ds is DATASETS[0] and j == 1), ylim=panel_ylim)
            if h_key == 'one_step':
                add_ar_var_bracket(
                    ax, [ratios_by_model['AR(1)'], ratios_by_model['LASSO'], ratios_by_model['VAR']], p,
                    y_pad_frac=0.015 if ds['key'] == 'geschwind' else 0.025,
                    h_frac=0.018 if ds['key'] == 'geschwind' else 0.025)
            ax.set_title(h_label, fontsize=8, pad=4)

        if c == 0:
            axs[0].set_ylabel('MAE ratio (model / persistence)', fontsize=8)
        sf.suptitle(ds['name'], fontsize=10, fontweight='bold', y=1.0)
        sf.text(0.5, 0.905, ds['subtitle'], ha='center', va='top', fontsize=8,
                style='italic', color='#333333')

    for ext in ('pdf', 'png'):
        path = out_dir / f'fig2.{ext}'
        fig.savefig(path, dpi=600 if ext == 'png' else None, bbox_inches='tight', facecolor='white')
        print('Saved:', path)
    plt.close(fig)

    pd.DataFrame(summary).to_csv(out_dir / 'fig2_summary.csv', index=False)
    stats = pd.DataFrame(stats)
    stats.to_csv(out_dir / 'fig2_stats.csv', index=False)
    print('Saved:', out_dir / 'fig2_summary.csv', 'and fig2_stats.csv')
    with pd.option_context('display.width', 160, 'display.float_format', '{:.3g}'.format):
        print(stats.to_string(index=False))


if __name__ == '__main__':
    main()
