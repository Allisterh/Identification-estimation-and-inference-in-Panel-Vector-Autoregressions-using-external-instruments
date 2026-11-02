"""Impulse-response grid: one row per application, one column per (specification, clustering)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .study import SPEC_LABEL, SPECS


def irf_grid(irf_df, rule='bic', specs=SPECS, clusters=('unit', 'time'), path=None, titles=None):
    apps = list(dict.fromkeys(irf_df.app))
    ncol = len(specs) * len(clusters)
    fig, axes = plt.subplots(len(apps), ncol, figsize=(3.0 * ncol, 2.7 * len(apps)), squeeze=False)
    for i, app in enumerate(apps):
        sub = irf_df[(irf_df.app == app) & (irf_df.p_rule == rule)]
        cap = 3 * np.nanmax(np.abs(sub.irf)) if len(sub) else 1
        lo_all = [max(-cap, np.nanmin(sub.loc[sub.case95 == 1, 'lo95'])) if (sub.case95 == 1).any() else -cap]
        hi_all = [min(cap, np.nanmax(sub.loc[sub.case95 == 1, 'hi95'])) if (sub.case95 == 1).any() else cap]
        ylo, yhi = min(lo_all[0], 0), max(hi_all[0], 0)
        pad = 0.06 * (yhi - ylo)
        for j, (spec, cl) in enumerate([(s, c) for s in specs for c in clusters]):
            ax = axes[i, j]
            g = sub[(sub.spec == spec) & (sub.cluster == cl)].sort_values('h')
            if g.empty:
                ax.axis('off'); continue
            h = g.h.values
            for tag, alpha in (('95', 0.18), ('90', 0.32)):
                lo = np.where(g[f'case{tag}'] == 1, g[f'lo{tag}'], np.nan)
                hi = np.where(g[f'case{tag}'] == 1, g[f'hi{tag}'], np.nan)
                ax.fill_between(h, lo, hi, color='tab:blue', alpha=alpha, lw=0)
            bad = g[g.case95 != 1]
            ax.plot(bad.h, np.full(len(bad), ylo + pad / 2), 'v', color='tab:red', ms=3)
            ax.plot(h, g.irf, color='navy', lw=1.4)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_ylim(ylo - pad, yhi + pad)
            ax.set_xlim(0, h.max())
            p = int(g.p.iloc[0])
            ax.set_title(f'{SPEC_LABEL[spec]}, {cl} clust. (p={p})', fontsize=8)
            if j == 0:
                ax.set_ylabel((titles or {}).get(app, app), fontsize=8)
            ax.tick_params(labelsize=7)
            if i == len(apps) - 1:
                ax.set_xlabel('horizon', fontsize=8)
    fig.suptitle(f'PVAR-IV response of the outcome to a unit policy shock (cumulative multiplier where indicated); 90% (dark) and '
                 f'95% (light) AR sets; lag rule: {rule}.  Red triangles: 95% set unbounded/complement', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if path:
        fig.savefig(path, dpi=150)
    return fig
