"""Main application of the paper: Chodorow-Reich, Nenov and Simsek (2021, AER).  Produces every input of Section 4.

Specification (the paper's own): bivariate PVAR in the stock-wealth shock s and 100 x log nontradable payroll y,
instrument z = S_{t-8} x R^m_t (Table 3 row 9), exogenous controls as in the paper (lagged Bartik, four interactions
of lagged wealth with aggregate returns), county and state x quarter effects, 2010 population weights, within
estimator.  Impulse = one unit of s, response of y, horizons 0..H quarters.

What is new relative to run.py:
  1. Lag order.  BIC hit the cap (pmax = 8) in run.py, and the residual check failed at p = 8.  Here pmax = 16 and the
     PVAR is estimated at p = 8, 12, 16 and at the BIC order.  The residual check reports the autocorrelation
     COEFFICIENTS at lags 1 to 4 next to their t-statistics: with 2,897 counties a tiny autocorrelation is significant,
     so the size matters for the reading.
  2. Horizon extended to H = 16 (four years), LP-IV of the paper recomputed at the same horizons.
  3. Baseline estimator: within estimator with the half-panel jackknife correction (Dhaene and Jochmans 2015).
     Every row uses it except two robustness rows: the uncorrected within estimator, and FOD-GMM (county effects
     removed by forward orthogonal deviations, lagged levels as instruments; Arellano and Bover 1995).  Other
     robustness rows: p = 12, 16 and the BIC order, unweighted, and the row-10 instrument (fixed dividends).
  4. Outputs for the paper: two figures and three LaTeX tables.
  5. Instrument and shock diagnostics (inst_diag.csv, table_shock.tex), for the assumption that the policy shock is
     serially uncorrelated:
       - autocorrelation of the identified policy shock at lags 1 to 4.  Under invertibility the shock is recovered
         up to scale as e_t = Gamma' Sigma^{-1} eta_t (Gamma = Cov(eta, z), Sigma = Cov(eta));
       - autocorrelation of the instrument z at lags 1 to 4 (after the county and state x quarter effects);
       - correlation of z_t with the VAR residuals of each equation at t-1 to t-4.
     Each is a pooled regression of the current on the lagged series (both demeaned), reported as a correlation with
     a two-way (county and quarter) clustered t-statistic.  Pairs are formed within spells of consecutive quarters.

usage:  python run_main_application.py [data dir or row9_dta.xpt] [out dir] [--reuse-lp] [--growth] [--figures-only]
        --figures-only: redraw the figures and tables from the csv files already in the output folder
        --growth: the outcome enters the PVAR as quarterly growth (first difference of 100 x log payroll) and the
        responses and AR sets are cumulated, so they refer to the payroll level as the local projections do;
        results in ../results/main_application_growth
        defaults: ../data, ../results/main_application.  --reuse-lp reads <out>/lp_iv.csv if it covers 0..H.
outputs: summary.csv, irf.csv, resid_diag.csv, inst_diag.csv, lags.csv, lp_iv.csv,
         fig_main.png, fig_robust.png, table_diagnostics.tex, table_irf.tex, table_shock.tex
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))   # project root
sys.path.insert(0, os.path.join(_ROOT, 'package'))               # the pvariv package lives in package/
sys.path.insert(0, _HERE)
_APP = os.path.abspath(os.path.join(_HERE, '..'))                 # this paper's folder
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cns
from pvariv import PVARIV
from pvariv.diagnostics import select_lags, resid_autocorr
from pvariv.study import zero_excluded


def _arg(i, default):
    return sys.argv[i] if len(sys.argv) > i and not sys.argv[i].startswith('--') else default


DATA = _arg(1, os.path.join(_APP, 'data'))
GROWTH = '--growth' in sys.argv     # outcome in quarterly growth rates; responses cumulated to the payroll level
OUT = _arg(2, os.path.join(_APP, 'results', 'main_application' + ('_growth' if GROWTH else '')))
os.makedirs(OUT, exist_ok=True)

cfg = cns.CONFIG
H = 16
PMAX = 16
P_GRID = (8, 12, 16)
LEVELS = (0.90, 0.95)
CLUSTERS = ('twoway', 'time', 'unit')
AC_LAGS = (1, 2, 3, 4)
H_TABLE = (0, 4, 7, 12, 16)
PUB_H, PUB_COEF, PUB_SE = 7, *cns.PUBLISHED['row9']

# ---------------------------------------------------------------- replication and LP-IV of the paper
FIG_ONLY = '--figures-only' in sys.argv   # redraw the figures from the stored csv files, no estimation
if FIG_ONLY:
    lp = pd.read_csv(os.path.join(_APP, 'results', 'main_application', 'lp_iv.csv') if GROWTH else os.path.join(OUT, 'lp_iv.csv'))
    summary, irf_df = pd.read_csv(f'{OUT}/summary.csv'), pd.read_csv(f'{OUT}/irf.csv')
    diag_df, idiag_df = pd.read_csv(f'{OUT}/resid_diag.csv'), pd.read_csv(f'{OUT}/inst_diag.csv')
else:
    cns.replicate(DATA)
    lp_file = os.path.join(_APP, 'results', 'main_application', 'lp_iv.csv') if GROWTH else os.path.join(OUT, 'lp_iv.csv')
    # (the local projections do not depend on the PVAR specification, so the growth run reuses the levels run's file)
    lp = None
    if '--reuse-lp' in sys.argv and os.path.exists(lp_file):
        lp = pd.read_csv(lp_file)
        if lp.h.max() < H:
            lp = None
    if lp is None:
        raw = cns.read_xpt(DATA)
        rows = []
        for h in range(H + 1):
            b, se, n, G = cns.lp_iv(raw, 9, h)
            rows.append(dict(h=h, coef=b, se=se, n=n))
            print(f'LP-IV h={h:2d}  {b:6.3f} ({se:.3f})')
        lp = pd.DataFrame(rows)
        lp.to_csv(lp_file, index=False)

    # ---------------------------------------------------------------- PVAR-IV
    summ, irfs, diag, idiag, lagtabs = [], [], [], [], []


    def _meat(sc, g):
        if g is None:
            return float(np.sum(sc ** 2))
        u, ic = np.unique(g, return_inverse=True)
        G = len(u)
        return G / (G - 1) * float(np.sum(np.bincount(ic, weights=sc) ** 2))


    def lag_regression(a, b, unit, time):
        """Slope of a on b (both demeaned, no constant), correlation, and two-way clustered t (Cameron-Gelbach-Miller;
        if the combined meat is not positive, the larger one-way meat is used)."""
        a = a - a.mean()
        b = b - b.mean()
        bb = b @ b
        beta = (a @ b) / bb
        sc = b * (a - beta * b)
        Mu, Mt = _meat(sc, unit), _meat(sc, time)
        M = Mu + Mt - _meat(sc, None)
        if M <= 0:
            M = max(Mu, Mt)
        corr = (a @ b) / np.sqrt(bb * (a @ a))
        return corr, beta / (np.sqrt(M) / bb)


    def shock_diagnostics(m, label, p):
        """Rows of inst_diag: identified shock and instrument autocorrelation, instrument vs lagged VAR residuals."""
        sp, tm, un = m.clusters['spell'], m.clusters['time'], m.clusters['unit']
        Sigma = m.eta @ m.eta.T / m.T
        shock = m.Gamma @ np.linalg.solve(Sigma, m.eta)          # e_t = Gamma' Sigma^{-1} eta_t, up to scale
        series = {'shock': shock, 'instrument': m.z}
        for l in AC_LAGS:
            idx = np.where((sp[l:] == sp[:-l]) & (tm[l:] - tm[:-l] == l))[0]
            cur, lag = idx + l, idx                               # positions of the (t, t-l) pairs
            u, t = un[cur], tm[cur]
            rows = [('shock autocorrelation', series['shock'][cur], series['shock'][lag]),
                    ('instrument autocorrelation', series['instrument'][cur], series['instrument'][lag])]
            for j, v in enumerate(cfg['endog']):
                rows.append((f'instrument vs residual {v}', m.z[cur], m.eta[j][lag]))
            for name, a, b in rows:
                c, tt = lag_regression(a, b, u, t)
                idiag.append(dict(variant=label, p=p, series=name, lag=l, corr=c, t_twoway=tt, n_pairs=len(idx)))


    def prepare_fod(df, weighted):
        """For FOD-GMM: state x quarter effects removed by (weighted) absorption, county effects left in the data (FOD
        removes them; demeaning by county first would put future shocks into the lagged-level GMM instruments), rows
        scaled by sqrt(w)."""
        cols = cfg['endog'] + [cfg['instrument']] + cfg['exog']
        w = df.w.values if weighted else np.ones(len(df))
        out = df.copy()
        out[cols] = cns.absorb(df[cols].values, [df.sq.values], w) * np.sqrt(w)[:, None]
        return out


    def fit_variant(label, df, weighted, p_list, with_bic, bias_correction=None, method='within'):
        dat = cns.prepare(df, weighted, 'twoway') if method == 'within' else prepare_fod(df, weighted)
        ps = list(p_list)
        p_bic = None
        if with_bic:
            lt = select_lags(dat, cfg['unit'], cfg['time'], cfg['endog'], PMAX, 'within')
            p_bic = int(lt.attrs['choice']['bic'])
            lagtabs.append(lt.reset_index().assign(variant=label))
            if p_bic not in ps:
                ps.append(p_bic)
        for p in sorted(ps):
            m = PVARIV(dat, cfg['unit'], cfg['time'], cfg['endog'], 'z', p=p, exog=cfg['exog'], policy=0,
                       method=method, bias_correction=bias_correction).fit()
            irf = m.irf(H)[1]
            if GROWTH:
                irf = np.cumsum(irf)
            ac, flag, maxt = resid_autocorr(m, lags=AC_LAGS)
            shock_diagnostics(m, label, p)
            maxc = 0.0
            for eq, d in ac.items():
                for l, (c, t) in d.items():
                    diag.append(dict(variant=label, p=p, equation=eq, lag=l, coef=c, t=t))
                    if np.isfinite(c):
                        maxc = max(maxc, abs(c))
            K = len(cfg['endog'])
            comp = np.zeros((K * p, K * p)); comp[:K, :] = m.AL
            if p > 1:
                comp[K:, :K * (p - 1)] = np.eye(K * (p - 1))
            max_eig = float(np.max(np.abs(np.linalg.eigvals(comp))))   # stability of the estimated PVAR
            for cl in CLUSTERS:
                fs = m.first_stage(cl)
                rec = dict(variant=label, p=p, p_is_bic=(p == p_bic), cluster=cl, n=m.T, F_robust=fs['F'],
                           resid_ac_maxcoef=maxc, resid_ac_maxt=maxt, resid_ac_flag=flag, max_eig=max_eig)
                sets = {}
                for lev in LEVELS:
                    o, c = m.ar_sets(H, cl, cumulative=GROWTH, level=lev)
                    sets[lev] = (o[1], c[1])
                for h in range(H + 1):
                    r = dict(variant=label, p=p, p_is_bic=(p == p_bic), cluster=cl, h=h, irf=irf[h])
                    for lev in LEVELS:
                        t = round(lev * 100)
                        lo, hi = sets[lev][0][h]
                        case = int(sets[lev][1][h])
                        r.update({f'lo{t}': lo, f'hi{t}': hi, f'case{t}': case, f'zero_out{t}': zero_excluded(lo, hi, case)})
                    irfs.append(r)
                summ.append(rec)
                print(f'{label:16s} p={p:2d}{" (BIC)" if p == p_bic else "      "} {cl:4s} n={m.T} F={fs["F"]:9.1f} '
                      f'h7={irf[7]:.2f}  max|ac coef|={maxc:.3f} (max|t|={maxt:.1f})  max|eig|={max_eig:.4f}')


    def to_growth(df):
        """Replace the outcome (100 x log nontradable payroll) by its quarterly change; quarters without the previous
        quarter are dropped, which splits the county into spells."""
        df = df.sort_values(['COUNTYID', 'QUARTER']).copy()
        g = df.groupby('COUNTYID')
        consecutive = (df.QUARTER - g.QUARTER.shift(1)) == 1
        df['y'] = np.where(consecutive, df.y - g.y.shift(1), np.nan)
        return df[df.y.notna()].reset_index(drop=True)


    def load(row):
        df = cns.load(DATA, row)
        return to_growth(df) if GROWTH else df


    df9 = load(9)
    fit_variant('row9 weighted', df9, True, P_GRID, with_bic=True, bias_correction='hpj')
    fit_variant('row9 unweighted', df9, False, (8,), with_bic=False, bias_correction='hpj')
    fit_variant('row10 weighted', load(10), True, (8,), with_bic=False, bias_correction='hpj')
    fit_variant('row9 weighted within', df9, True, (8,), with_bic=False)
    fit_variant('row9 weighted fod-gmm', df9, True, (8,), with_bic=False, method='fod-gmm')

    summary, irf_df, diag_df = pd.DataFrame(summ), pd.DataFrame(irfs), pd.DataFrame(diag)
    idiag_df = pd.DataFrame(idiag)
    idiag_df.to_csv(f'{OUT}/inst_diag.csv', index=False)
    summary.to_csv(f'{OUT}/summary.csv', index=False)
    irf_df.to_csv(f'{OUT}/irf.csv', index=False)
    diag_df.to_csv(f'{OUT}/resid_diag.csv', index=False)
    if lagtabs:
        pd.concat(lagtabs, ignore_index=True).to_csv(f'{OUT}/lags.csv', index=False)


def sel(variant, p, cl):
    return irf_df[(irf_df.variant == variant) & (irf_df.p == p) & (irf_df.cluster == cl)].sort_values('h')


def finite_band(g, tag):
    ok = g[f'case{tag}'] == 1
    return np.where(ok, g[f'lo{tag}'], np.nan), np.where(ok, g[f'hi{tag}'], np.nan)


# ---------------------------------------------------------------- Figure 1: main specification against LP-IV
t8, q8 = sel('row9 weighted', 8, 'twoway'), sel('row9 weighted', 8, 'time')
fig, a = plt.subplots(figsize=(8.0, 4.8))
a.axhline(0, color='k', lw=0.6)
lo, hi = finite_band(t8, '95'); a.fill_between(t8.h, lo, hi, color='#9ecae1', alpha=0.7, lw=0)
lo, hi = finite_band(t8, '90'); a.fill_between(t8.h, lo, hi, color='#3182bd', alpha=0.6, lw=0)
lo, hi = finite_band(q8, '95'); a.plot(q8.h, lo, color='navy', lw=0.9, ls='--'); a.plot(q8.h, hi, color='navy', lw=0.9, ls='--')
a.plot(t8.h, t8.irf, color='navy', lw=2, label='PVAR-IV impulse response')
a.errorbar(lp.h + 0.15, lp.coef, yerr=1.96 * lp.se, fmt='o', ms=3.5, color='darkorange', capsize=2, lw=1,
           label='LP-IV (95%, two-way clustered)')
a.errorbar([PUB_H - 0.15], [PUB_COEF], yerr=[1.96 * PUB_SE], fmt='D', ms=6, color='crimson', capsize=3, lw=1.4, zorder=6,
           label='Published estimate')
a.set_xlabel('Quarters after the shock')
a.set_ylabel('Basis points of nontradable payroll')
# legend in the upper left, above the sets: extend the vertical axis to leave room for it
top = np.nanmax([np.nanmax(np.where(np.isfinite(t8.hi95), t8.hi95, np.nan)), (lp.coef + 1.96 * lp.se).max()])
a.set_ylim(top=1.45 * top)
a.legend(fontsize=11, loc='upper left', frameon=True, framealpha=0.9, edgecolor='none')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_main.png', dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: robustness (point estimates, quarter clustering)
fig, a = plt.subplots(figsize=(8.0, 4.8))
a.axhline(0, color='k', lw=0.6)
lo, hi = finite_band(t8, '95'); a.fill_between(t8.h, lo, hi, color='#9ecae1', alpha=0.5, lw=0)
styles = [('row9 weighted', 8, 'navy', '-', 'Baseline, p = 8')]
for p in sorted(set(summary[summary.variant == 'row9 weighted'].p)):
    if p != 8:
        bic = bool(summary[(summary.variant == 'row9 weighted') & (summary.p == p)].p_is_bic.iloc[0])
        styles.append(('row9 weighted', p, None, '--', f'p = {p}' + (' (BIC)' if bic else '')))
styles += [('row9 unweighted', 8, 'teal', ':', 'Unweighted'), ('row10 weighted', 8, 'olive', '-.', 'Fixed-dividend instrument'),
           ('row9 weighted within', 8, 'crimson', '-', 'Within estimator, no correction'),
           ('row9 weighted fod-gmm', 8, 'purple', '--', 'FOD-GMM')]
for var, p, col, ls, lab in styles:
    g = sel(var, p, 'twoway')
    a.plot(g.h, g.irf, color=col, ls=ls, lw=1.8 if p == 8 and var == 'row9 weighted' else 1.3, label=lab)
a.set_xlabel('Quarters after the shock')
a.set_ylabel('Basis points of nontradable payroll')
top = np.nanmax(np.where(np.isfinite(t8.hi95), t8.hi95, np.nan))
for var, p, col, ls, lab in styles:
    top = max(top, np.nanmax(sel(var, p, 'twoway').irf))
a.set_ylim(top=1.45 * top)
a.legend(fontsize=10, loc='upper left', ncol=2, frameon=True, framealpha=0.9, edgecolor='none')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_robust.png', dpi=300); plt.close(fig)


# ---------------------------------------------------------------- LaTeX tables
def tex_set(lo, hi, case, d=2):
    if case == 4:
        return 'unbounded'
    if case == 3:
        return 'empty'
    f = f'{{:.{d}f}}'
    return f'[{f.format(lo)}, {f.format(hi)}]' if case == 1 else f'$(-\\infty,{f.format(lo)}]\\cup[{f.format(hi)},\\infty)$'


LABEL = {'row9 weighted': 'Baseline', 'row9 unweighted': 'Unweighted', 'row10 weighted': 'Fixed-dividend instrument',
         'row9 weighted within': 'Within estimator, no correction', 'row9 weighted fod-gmm': 'FOD-GMM'}
keys = summary[summary.cluster == 'twoway'][['variant', 'p', 'p_is_bic']].drop_duplicates().values.tolist()

lines = [r'\begin{table}[ht]\centering\footnotesize',
         r'\begin{tabular}{lrrrrrrr}\toprule',
         r'Specification & $p$ & $n$ & $F$, two-way & $F$, quarter & max $|\hat\rho|$ & max $|t|$ & max $|\mu|$\\\midrule']
for var, p, bic in keys:
    t = summary[(summary.variant == var) & (summary.p == p) & (summary.cluster == 'twoway')].iloc[0]
    u = summary[(summary.variant == var) & (summary.p == p) & (summary.cluster == 'time')].iloc[0]
    lines.append(f"{LABEL[var]} & {p}{' (BIC)' if bic else ''} & {int(t.n):,} & {t.F_robust:.1f} & {u.F_robust:.1f} & "
                 f"{t.resid_ac_maxcoef:.3f} & {t.resid_ac_maxt:.1f} & {t.max_eig:.3f}\\\\")
lines += [r'\bottomrule\end{tabular}',
          r'\caption{First stage and residual diagnostics. $F$ is the robust first-stage statistic, two-way clustered by '
          r'county and quarter or clustered by quarter. $\hat\rho$ is the autocorrelation of the VAR residuals at lags 1 to 4, largest absolute '
          r'value over equations and lags, with its county-clustered $t$-statistic. $|\mu|$: largest modulus of the '
          r'eigenvalues of the companion matrix of the estimated PVAR.\label{tab:cns_diag}}',
          r'\end{table}']
open(f'{OUT}/table_diagnostics.tex', 'w').write('\n'.join(lines))

lines = [r'\begin{table}[ht]\centering\footnotesize',
         r'\resizebox{\textwidth}{!}{\begin{tabular}{ll' + 'c' * len(H_TABLE) + r'}\toprule',
         r'Specification & $p$ & ' + ' & '.join(f'$h={h}$' for h in H_TABLE) + r'\\\midrule']
for var, p, bic in keys:
    g = sel(var, p, 'twoway').set_index('h')
    pt = ' & '.join(f'{g.loc[h, "irf"]:.2f}' for h in H_TABLE)
    st = ' & '.join(tex_set(g.loc[h, 'lo95'], g.loc[h, 'hi95'], g.loc[h, 'case95']) for h in H_TABLE)
    lines.append(f"{LABEL[var]} & {p}{' (BIC)' if bic else ''} & {pt}\\\\")
    lines.append(f" & & {st}\\\\[2pt]")
lpi = lp.set_index('h')
lines.append(r'\midrule Local projection IV & & ' + ' & '.join(f'{lpi.loc[h, "coef"]:.2f}' for h in H_TABLE) + r'\\')
lines.append(r' & & ' + ' & '.join(f'({lpi.loc[h, "se"]:.2f})' for h in H_TABLE) + r'\\')
lines += [r'\bottomrule\end{tabular}}',
          r'\caption{Response of nontradable payroll, in basis points, to a stock-wealth shock equal to 1\% of annual '
          r'labor income. PVAR-IV rows: point estimate and 95\% Anderson-Rubin set, two-way clustered by county and quarter; '
          r'within estimator with half-panel jackknife correction unless the row states otherwise. Local '
          r'projection IV: the estimator of the original study, with two-way clustered standard errors in '
          r'parentheses.\label{tab:cns_irf}}',
          r'\end{table}']
open(f'{OUT}/table_irf.tex', 'w').write('\n'.join(lines))

# Table 3: shock and instrument diagnostics, baseline p = 8
SERIES = [('shock autocorrelation', r'Identified policy shock, $\hat\varepsilon_t$ on $\hat\varepsilon_{t-l}$'),
          ('instrument autocorrelation', r'Instrument, $Z_t$ on $Z_{t-l}$'),
          ('instrument vs residual s', r'Instrument and wealth-shock residual, $Z_t$ on $\tilde{x}_{1,t-l}$'),
          ('instrument vs residual y', r'Instrument and payroll residual, $Z_t$ on $\tilde{x}_{2,t-l}$')]
b = idiag_df[(idiag_df.variant == 'row9 weighted') & (idiag_df.p == 8)].set_index(['series', 'lag'])
lines = [r'\begin{table}[ht]\centering\footnotesize',
         r'\begin{tabular}{l' + 'c' * len(AC_LAGS) + r'}\toprule',
         ' & ' + ' & '.join(f'$l={l}$' for l in AC_LAGS) + r'\\\midrule']
for key, lab in SERIES:
    lines.append(lab + ' & ' + ' & '.join(f'{b.loc[(key, l), "corr"]:.3f}' for l in AC_LAGS) + r'\\')
    lines.append(' & ' + ' & '.join(f'({b.loc[(key, l), "t_twoway"]:.2f})' for l in AC_LAGS) + r'\\[2pt]')
lines += [r'\bottomrule\end{tabular}',
          r'\caption{Serial correlation of the identified policy shock and of the instrument, baseline specification. '
          r'Each entry is the correlation between the current and the lagged series, pooled over counties and '
          r'quarters, with the $t$-statistic of the corresponding regression, two-way clustered by county and '
          r"quarter, in parentheses. The policy shock is recovered as $\hat\Gamma'\hat\Sigma^{-1}\tilde{x}_t$."
          r'\label{tab:cns_shock}}',
          r'\end{table}']
open(f'{OUT}/table_shock.tex', 'w').write('\n'.join(lines))

pd.set_option('display.width', 250)
print('\nShock and instrument diagnostics (correlation, two-way t), all specifications')
print(idiag_df.pivot_table(index=['variant', 'p', 'series'], columns='lag', values=['corr', 't_twoway']).round(3).to_string())
print('\nwritten to', OUT)
