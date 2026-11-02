"""Systematic study runner: the same PVAR-IV ladder for every application.

Specifications (estimation):  'none'   no fixed effects (pooled OLS VAR with a constant)
                              'unit'   unit fixed effects (within, or FOD-GMM as set by the application)
                              'twoway' unit and time fixed effects
Inference: AR sets for each of `clusters` (e.g. 'unit', 'time') at each of `levels` (e.g. 0.90, 0.95).
Lag rules, fixed before running:  'bic' = system BIC on a common sample (per specification),  'paper' = the original
paper's own lag order.  Nothing in the configuration is tuned on the second-stage results."""
import numpy as np
import pandas as pd
from .core import PVARIV
from .utils import twoway_demean, demean_by
from .diagnostics import select_lags, resid_autocorr

SPECS = ('none', 'unit', 'twoway')
SPEC_LABEL = {'none': 'No FE', 'unit': 'Unit FE', 'twoway': 'Unit + time FE',
              'unit_trend': 'Unit FE + unit trends', 'twoway_trend': 'Unit + time FE + unit trends'}


def prepare(df, cfg, spec):
    """Data and estimation method for a specification."""
    ins = [cfg['instrument']] if isinstance(cfg['instrument'], str) else list(cfg['instrument'])   # one or several instruments
    cols = list(cfg['endog']) + ins + list(cfg.get('exog', []))
    um = cfg.get('unit_method', 'within')
    if spec == 'none':
        return df, 'pooled'
    if spec == 'unit':
        return df, um
    if spec in ('unit_trend', 'twoway_trend'):          # unit effects + unit-specific linear trends (+ time effects); within only
        return trend_demean(df, cols, cfg['unit'], cfg['time'], with_time=(spec == 'twoway_trend')), 'within'
    if spec == 'twoway':
        cell = cfg.get('cell', cfg['time'])             # time-effect cell: the period, or e.g. region x period
        if um == 'within':
            return twoway_demean(df, cols, cfg['unit'], cell), 'within'
        return demean_by(df, cols, cell), um       # FOD-GMM: remove time effects only, FOD removes unit effects
    raise ValueError(spec)


def trend_demean(df, cols, unit, time, with_time):
    """Partial out unit effects, unit-specific linear trends and (optionally) time dummies from `cols` (exact projection)."""
    out = df.copy()
    cid = pd.factorize(df[unit])[0]
    n = len(df)
    t = (df[time].values - df[time].values.min()).astype(float)
    t = t - t.mean()
    D = np.zeros((n, cid.max() + 1)); D[np.arange(n), cid] = 1.0
    parts = [D, D * t[:, None]]
    if with_time:
        tt = pd.factorize(df[time])[0]
        T = np.zeros((n, tt.max() + 1)); T[np.arange(n), tt] = 1.0
        parts.append(T[:, 1:])
    Z = np.hstack(parts)
    V = df[cols].values.astype(float)
    out[cols] = V - Z @ np.linalg.lstsq(Z, V, rcond=None)[0]
    return out


def zero_excluded(lo, hi, case):
    """True if 0 is NOT in the AR set."""
    if case == 1:
        return not (lo <= 0 <= hi)
    if case == 2:                                   # complement of (lo, hi)
        return lo < 0 < hi
    if case == 3:
        return False
    return False


def run_application(app, data_path, specs=SPECS, clusters=('unit', 'time'), levels=(0.90, 0.95),
                    rules=('bic', 'paper'), verbose=True):
    cfg = app.CONFIG
    df = app.load(data_path)
    H, pol, out = cfg['H'], cfg['policy'], cfg['outcome']
    cumm = cfg.get('cumulative_multiplier', False)      # report sum_s irf_out / sum_s irf_policy instead of the IRF
    integ = cfg.get('integral_multiplier', False)      # with cumulative_multiplier: ratio of sums of level responses over 0..h
    cumr = cfg.get('cumulative_response', False)        # report sum_s irf_out (cumulative response of the outcome)
    summ, irfs, lagtabs = [], [], []
    for spec in specs:
        data, method = prepare(df, cfg, spec)
        try:
            lt = select_lags(data, cfg['unit'], cfg['time'], cfg['endog'], cfg['pmax'], method)
            p_bic = lt.attrs['choice']['bic']
            lagtabs.append(lt.reset_index().assign(app=app.NAME, spec=spec))
        except NotImplementedError:
            p_bic = None
        for rule in rules:
            p = p_bic if rule == 'bic' else cfg['p_paper'] if rule == 'paper' else int(rule)   # an int rule fixes p
            if p is None:
                continue
            ex = cfg.get('exog')
            if ex:      # controls that are identically zero once the first p periods are lost (e.g. dropped-year interactions)
                late = data[data[cfg['time']] >= data[cfg['time']].min() + p]
                ex = [c for c in ex if late[c].abs().max() > 1e-9]       # > 1e-9, not > 0: a national control is ~1e-16, not 0, after year-demeaning
            try:
                m = PVARIV(data, cfg['unit'], cfg['time'], cfg['endog'], cfg['instrument'], p=p,
                           exog=ex, policy=pol, method=method).fit()
            except Exception as e:                                   # singular design etc.: record, continue
                summ.append(dict(app=app.NAME, spec=spec, p_rule=rule, p=p, method=method, error=str(e)))
                continue
            irf = m.cumulative_multiplier(H, out, integral=integ) if cumm else (m.cumulative_irf(H)[out] if cumr else m.irf(H)[out])
            ac, flag, maxt = resid_autocorr(m)
            for cl in clusters:
                fs = m.first_stage(cl)
                rec = dict(app=app.NAME, spec=spec, cluster=cl, p_rule=rule, p=p, method=method, n=m.T,
                           units=len(np.unique(m.clusters['unit'])), internal=m.n_internal, external=m.n_external,
                           F_robust=fs['F'], F_classical=fs['F_classical'], F_eff=fs['F_eff'], t2_ar=fs['t2_ar'],
                           resid_ac_flag=flag, resid_ac_maxt=maxt, resp_H=irf[H])
                sets = {}
                for lev in levels:
                    if cumm:
                        sets[lev] = m.ar_ratio_sets(H, cl, out, level=lev, integral=integ)
                    else:
                        o, c = m.ar_sets(H, cl, cumulative=cumr, level=lev)
                        sets[lev] = (o[out], c[out])
                    tag = f'{round(lev * 100)}'
                    so, sc = sets[lev]
                    rec[f'lo{tag}'], rec[f'hi{tag}'] = so[H]
                    rec[f'case{tag}'] = int(sc[H])
                    rec[f'zero_out{tag}'] = zero_excluded(*so[H], sc[H])
                summ.append(rec)
                for h in range(H + 1):
                    r = dict(app=app.NAME, spec=spec, cluster=cl, p_rule=rule, p=p, h=h, irf=irf[h])
                    for lev in levels:
                        tag = f'{round(lev * 100)}'
                        r[f'lo{tag}'], r[f'hi{tag}'] = sets[lev][0][h]
                        r[f'case{tag}'] = int(sets[lev][1][h])
                    irfs.append(r)
                if verbose:
                    sets_txt = '  '.join(f"{round(lev * 100)}%: " + _fmt(rec[f'lo{round(lev * 100)}'], rec[f'hi{round(lev * 100)}'],
                                                                         rec[f'case{round(lev * 100)}']) for lev in levels)
                    print(f"{app.NAME:10s} {spec:6s} {cl:5s} {rule:5s} p={p} n={m.T:5d} F={fs['F']:7.1f} "
                          f"resp(h={H})={irf[H]:7.3f}  {sets_txt}")
    return dict(summary=pd.DataFrame(summ), irf=pd.DataFrame(irfs),
                lags=pd.concat(lagtabs, ignore_index=True) if lagtabs else pd.DataFrame())


def _fmt(lo, hi, case, d=2):
    if case == 4:
        return 'unbounded'
    if case == 3:
        return 'empty'
    f = f'{{:.{d}f}}'
    return f'[{f.format(lo)}, {f.format(hi)}]' if case == 1 else f'(-inf, {f.format(lo)}] U [{f.format(hi)}, inf)'


def latex_set(lo, hi, case, d=1):
    if case == 4:
        return 'unbounded'
    if case == 3:
        return 'empty'
    f = f'{{:.{d}f}}'
    s = f'[{f.format(lo)}, {f.format(hi)}]' if case == 1 else f'$(-\\infty, {f.format(lo)}]\\cup[{f.format(hi)}, \\infty)$'
    return '\\textbf{' + s + '}' if zero_excluded(lo, hi, case) else s


def longrun_table(summary, rule='bic', H=None, label_map=None):
    """LaTeX table of the long-run response with 90% and 95% AR sets (bold = 0 excluded)."""
    s = summary[(summary.p_rule == rule) & summary.cluster.notna()]
    lines = [r'\begin{table}[ht]\centering\scriptsize',
             r'\caption{Long-run response, PVAR-IV. AR sets (bold: zero excluded) at the 90\% and 95\% levels, by clustering.}',
             r'\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrllll}\hline',
             r' & & & & \multicolumn{2}{c}{$F$ (robust)} & & \multicolumn{2}{c}{AR set, unit clust.} & \multicolumn{2}{c}{AR set, time clust.}\\',
             r'Application & Spec & $p$ & $n$ & unit & time & Resp. & 90\% & 95\% & 90\% & 95\%\\\hline']
    for (app, spec), g in s.groupby(['app', 'spec'], sort=False):
        u = g[g.cluster == 'unit'].iloc[0]
        t = g[g.cluster == 'time'].iloc[0]
        lines.append(f"{app} & {SPEC_LABEL[spec]} & {int(u.p)} & {int(u.n)} & {u.F_robust:.1f} & {t.F_robust:.1f} & {u.resp_H:.1f} & "
                     f"{latex_set(u.lo90, u.hi90, u.case90)} & {latex_set(u.lo95, u.hi95, u.case95)} & "
                     f"{latex_set(t.lo90, t.hi90, t.case90)} & {latex_set(t.lo95, t.hi95, t.case95)}\\\\")
    lines += [r'\hline\end{tabular}}\end{table}']
    return '\n'.join(lines)
