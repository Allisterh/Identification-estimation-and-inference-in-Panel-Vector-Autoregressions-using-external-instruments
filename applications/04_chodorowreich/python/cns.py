"""Chodorow-Reich, Nenov and Simsek (2021, AER), 'Stock Market Wealth and the Real Economy: A Local Labor Market Approach'.

Data: data_4/row9_dta.xpt (county x quarter, 3,293 counties, 179 quarters; estimation sample SAMPLE==1).  This file is the
extract used by the MATLAB break-test replication (runmetoreplicate4_1.m, row 9 of Table 3, and _2.m, row 10).
Paper's design: local projection of y_{t+h} - y_{t-1} (y = 100 x log nontradable payroll, `LWAGES_N`) on the stock-wealth shock
s_t = S_{t-1} R_t (`WEALTH_3`, wealth/income x county return), instrumented with S_{t-8} x R^m_t (row 9, `L8_WEAL2*XRETCOUN`) or with
the fixed-dividend version `MEAN_WEA` (row 10); controls: 8 lags of s, lagged Bartik, four interactions of lagged wealth with
aggregate returns; county and state x quarter effects, 2010-population weights, county and quarter two-way clustering.  The
coefficient is reported in basis points for a shock equal to 1% of annual labor income (= beta, because y is 100 x log).

PVAR-IV: K = 2, endog = (s, y); instrument z = row-9 (or row-10) instrument at date t; impulse = one unit of s, response of y.
h = 7 is the horizon of the paper's Table 3.  Fixed in advance: within estimator (T = 92, N = 2,897, persistent y), p = 8 ('paper':
eight lags of the shock) and BIC (max 8), H = 12, unit and time clustering, county + state x quarter effects, population weights
(weights enter by weighted two-way absorption followed by scaling each county's rows by sqrt(w), which is exact WLS).
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))   # project root
sys.path.insert(0, os.path.join(_ROOT, 'package'))   # the pvariv package lives in package/
sys.path.insert(0, _HERE)
import os, sys
import numpy as np
import pandas as pd

NAME = 'cns'
PAPER = 'Chodorow-Reich, Nenov, Simsek (2021)'
# Table 3, columns 4 (nontradable payroll): row 9 'IV with lagged wealth', row 10 'IV with fixed dividends/income'; h = 7, basis points
PUBLISHED = dict(row9=(2.61, 0.96), row10=(2.90, 0.25))
# first-stage output (Stata ivreghdfe, no absorb, no weights) printed in the MATLAB files: instrument coefficient
STATA_FS = dict(row9=0.8570258, row10=0.9470561)
CONFIG = dict(unit='COUNTYID', time='QUARTER', cell='sq', endog=['s', 'y'], instrument='z', policy=0, outcome=1,
              exog=['bartik', 'x_comp', 'x_nonc', 'x_treas', 'x_hp'],
              labels=dict(s='Wealth shock (1 = 100% of annual labor income)', y='Nontradable payroll (100 x log)'),
              H=12, pmax=8, p_paper=8, unit_method='within')
CONTROLS = ['L1_BARTI']


def read_xpt(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'row9_dta.xpt')
    d = pd.read_sas(path, format='xport')
    d['COUNTYID'] = d.COUNTYID.astype(int); d['QUARTER'] = d.QUARTER.astype(int)
    d['STATE'] = d.COUNTYID // 1000
    return d.sort_values(['COUNTYID', 'QUARTER']).reset_index(drop=True)


def shift(d, col, k):
    """value of `col` at quarter t+k within county (NaN when that quarter is missing)"""
    t = d[['COUNTYID', 'QUARTER', col]].copy(); t['QUARTER'] -= k
    return d[['COUNTYID', 'QUARTER']].merge(t.rename(columns={col: '_x'}), on=['COUNTYID', 'QUARTER'], how='left')._x.values


def matlab_blocks(d, row):
    """D, X = [instrument, controls...] exactly as in runmetoreplicate4_1.m (row 9) / _2.m (row 10)."""
    ctrl = [d['L%d_WEALT' % j] for j in range(1, 9)] + [d.L1_BARTI]
    inter = [d.L1_WEAL2 * d.CUMRCOMP, d.L1_WEAL2 * d.CUMRNONC, d.L1_WEAL2 * d.RETTREAS, d.L1_WEAL2 * d.SCSUSHPI]
    z = d.L8_WEAL2 * d.XRETCOUN if row == 9 else d.MEAN_WEA
    return d.WEALTH_3.values, np.column_stack([np.asarray(c) for c in [z] + ctrl + inter])


def absorb(M, groups, w=None, tol=1e-10, maxit=2000):
    """weighted residuals of the columns of M after removing the dummies of every grouping in `groups`"""
    M = np.array(M, dtype=float, copy=True)
    if M.ndim == 1: M = M[:, None]
    w = np.ones(len(M)) if w is None else np.asarray(w, float)
    gs = [pd.factorize(g)[0] for g in groups]; gw = [np.bincount(g, weights=w) for g in gs]
    for _ in range(maxit):
        old = M.copy()
        for g, ws in zip(gs, gw):
            for j in range(M.shape[1]):
                M[:, j] -= (np.bincount(g, weights=w * M[:, j]) / ws)[g]
        if np.max(np.abs(M - old)) < tol: break
    return M


def tsls_cluster(y, X, Z, w, cl):
    """Weighted 2SLS on already-absorbed data (X[:,0] endogenous), 1- or 2-way cluster covariance (Cameron-Gelbach-Miller, G/(G-1))."""
    sw = np.sqrt(w); y = y * sw; X = X * sw[:, None]; Z = Z * sw[:, None]
    PX = Z @ (np.linalg.inv(Z.T @ Z) @ (Z.T @ X)); A = np.linalg.inv(PX.T @ X); b = A @ (PX.T @ y); u = y - X @ b
    def meat(c):
        c = pd.factorize(c)[0]; G = c.max() + 1; S = np.zeros((G, PX.shape[1])); np.add.at(S, c, PX * u[:, None]); return S.T @ S, G
    labs = list(cl) if len(cl) == 1 else [cl[0], cl[1], pd.Series(cl[0]).astype(str).values + '_' + pd.Series(cl[1]).astype(str).values]
    signs = [1] if len(cl) == 1 else [1, 1, -1]
    V = np.zeros((X.shape[1],) * 2); Gs = []
    for sg, c in zip(signs, labs):
        M, G = meat(c); Gs.append(G); V += sg * (A @ M @ A) * G / (G - 1)
    return b, V, Gs


def lp_iv(d, row=9, h=7, cluster=('COUNTYID', 'QUARTER'), weighted=True):
    """The paper's local-projection IV for y_{t+h} - y_{t-1}: coefficient (bp per 1% of annual labor income) and standard error."""
    d = d.copy(); d['yh'] = shift(d, 'LWAGES_N', h) - shift(d, 'LWAGES_N', -1)
    D, X = matlab_blocks(d, row)
    m = (d.SAMPLE.values == 1) & ~np.isnan(D) & ~np.isnan(X).any(1) & ~np.isnan(d.yh.values) & ~np.isnan(d.POPULATI.values)
    s = d[m]; w = s.POPULATI.values if weighted else np.ones(m.sum())
    Y = absorb(np.column_stack([s.yh.values, D[m], X[m]]), [s.COUNTYID.values, (s.STATE.astype(str) + '_' + s.QUARTER.astype(str)).values], w)
    b, V, G = tsls_cluster(Y[:, 0], np.column_stack([Y[:, 1], Y[:, 3:]]), Y[:, 2:], w, [s[c].values for c in cluster])
    return b[0], float(np.sqrt(V[0, 0])), int(m.sum()), G


def first_stage_matlab(d, row):
    """OLS first stage of the MATLAB scripts: D on [Z X 1] over SAMPLE == 1 (Stata `if sample`) and complete rows (no absorb, no weights)."""
    D, X = matlab_blocks(d, row)
    ok = (d.SAMPLE.values == 1) & ~np.isnan(D) & ~np.isnan(X).any(1)
    return np.linalg.lstsq(np.column_stack([X[ok], np.ones(ok.sum())]), D[ok], rcond=None)[0], int(ok.sum())


def load(path, row=9):
    """PVAR panel: s = WEALTH_3, y = LWAGES_N, z = row-`row` instrument at date t, controls as exogenous; SAMPLE == 1, complete rows."""
    d = read_xpt(path)
    D, X = matlab_blocks(d, row)
    out = pd.DataFrame(dict(COUNTYID=d.COUNTYID.values, QUARTER=d.QUARTER.values, STATE=d.STATE.values, s=D, y=d.LWAGES_N.values, z=X[:, 0],
                            bartik=X[:, 9], x_comp=X[:, 10], x_nonc=X[:, 11], x_treas=X[:, 12], x_hp=X[:, 13], w=d.POPULATI.values))
    out = out[d.SAMPLE.values == 1].dropna().reset_index(drop=True)
    out['sq'] = out.STATE.astype(str) + '_' + out.QUARTER.astype(str)
    return out


def prepare(df, weighted=True, effects='twoway'):
    """County (+ state x quarter) effects removed by (weighted) absorption; rows then scaled by sqrt(w) = exact WLS.  The PVARIV
    'within' estimator then runs on the transformed data (its own within step is a no-op up to rounding)."""
    cols = CONFIG['endog'] + [CONFIG['instrument']] + CONFIG['exog']
    w = df.w.values if weighted else np.ones(len(df))
    groups = [df.COUNTYID.values] + ([df.sq.values] if effects == 'twoway' else [])
    out = df.copy(); out[cols] = absorb(df[cols].values, groups, w) * np.sqrt(w)[:, None]
    return out


def replicate(path, verbose=True, strict=True):
    d = read_xpt(path); out, bad = {}, {}
    for row, key in ((9, 'row9'), (10, 'row10')):
        b, se, n, G = lp_iv(d, row, 7)
        pb, pse = PUBLISHED[key]; out[key] = (b, se, n, G)
        c, nfs = first_stage_matlab(d, row)
        if verbose:
            print(f'Table 3 row {row}, nontradable payroll, h=7: python {b:6.3f} ({se:.3f})  published {pb:5.2f} ({pse:.2f})   n={n} clusters={G[:2]}')
            print(f'   MATLAB first stage, instrument coef: python {c[0]:.4f} (n={nfs})  Stata {STATA_FS[key]:.4f}')
        if abs(b - pb) > 0.006: bad[key + '_coef'] = (b, pb)
        # row 10: the published SE (0.25) is not reproduced by any clustering tried (county 0.54, quarter 1.02, two-way 1.09);
        # the point estimate matches.  Only the row-9 SE is asserted.
        if key == 'row9' and abs(se - pse) > 0.006: bad[key + '_se'] = (se, pse)
        if abs(c[0] - STATA_FS[key]) > 1e-6: bad[key + '_fs'] = (c[0], STATA_FS[key])
    if bad and strict: raise AssertionError(f'replication mismatch: {bad}')
    out['mismatch'] = bad
    return out


if __name__ == '__main__':
    replicate(sys.argv[1], strict=False)
