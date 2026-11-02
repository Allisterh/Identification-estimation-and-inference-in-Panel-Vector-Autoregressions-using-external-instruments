"""Monte Carlo of Appendix D, calibrated to the main application (Chodorow-Reich, Nenov and Simsek 2021).

Three experiments, all on a balanced county x quarter panel with the size of the application:
  Sim 1  calibrated design: coverage of the 95% AR sets clustered by county, by quarter and two-way, and of the
         two-way delta-method (Wald) interval.
  Sim 2  weak instrument: the first-stage loading scaled by c = 0.4 and c = 0.2.
  Sim 3  persistent instrument: the national shift with autocorrelation 0.67 and 0.9 (no lead-lag exogeneity).

DGP, equation by equation (i = county, t = quarter):
  national shift          g_t = rho_g g_{t-1} + e_t,                 e_t ~ N(0, sigma_e^2)
  instrument              Z_it = S_i g_t                             S_i = county exposure (data)
  policy innovation       eps1_it = c gamma S_i e_t + psi_u S_i a_t + u_it,   a_t ~ N(0, 1), independent of e_t, f_t
  outcome innovation      eps2_it = psi S_i f_t + nu_it,              f_t ~ N(0, 1), independent of e_t
                          (u_it, nu_it) ~ N(0, [[s_u^2, r s_u s_nu], [r s_u s_nu, s_nu^2]]), iid over i and t
                          (r calibrated; Sim 2 adds confounding, see ENDOG_SIM2)
  PVAR                    x_it = alpha_i + sum_{l=1}^p Phi_l x_{i,t-l} + (eps1_it, beta eps1_it + eps2_it)'
  with x_it = (wealth shock, payroll).  The response of payroll at horizon h to a unit change in the policy variable
  is e_2' C_h (1, beta)', C_h the MA matrices of Phi.  The instrument moves the policy only through the innovation
  e_t of the national shift, so it is exogenous to eps2 at all dates and correlated with past policy shocks when
  rho_g > 0.  The factor term psi S_i f_t makes the outcome innovation correlated across counties within a quarter,
  more so in high-exposure counties; this is what makes clustering by county alone invalid.

Calibration (step 'calibrate', from the replication data; unweighted, county and state x quarter effects, the
paper's controls, p lags):
  Phi, beta = Gamma_2 / Gamma_1     PVAR-IV estimates
  S_i                               county mean of the lagged wealth exposure (L8_WEAL2)
  rho_g, sigma_e                    AR(1) of the quarterly mean of the return series (XRETCOUN)
  psi_u, s_u                        match var(eta_1 - pi Z) and the ratio of the quarter-clustered to the
                                    heteroskedasticity-robust variance of the first-stage moment Z (eta_1 - pi Z);
                                    without this term the first stage averages out over counties and is far
                                    stronger than in the data
  gamma = pi / (1 - rho_g^2)        pi = first-stage slope of the policy residual on the instrument
  psi, s_nu                         match var(eta_2 - beta eta_1) and the ratio of the quarter-clustered to the
                                    heteroskedasticity-robust variance of the moment Z (eta_2 - beta eta_1)
  r                                 correlation of (eta_1 - pi Z) and (eta_2 - beta eta_1)
  alpha_i                           normal, standard deviations of the county means of the two variables

First-stage strength: gamma stays at its calibrated value and the step 'tune' scales psi_u so that the median
two-way first-stage F of the calibrated design equals the F of the application (the closed-form psi_u above
understates the common first-stage noise).  Sim 2 then scales gamma by c.

Estimation in every replication: unit and quarter effects removed, within PVAR with p lags, AR sets and the Wald
interval for the payroll response at h = 0, 4, 7, 12.

usage:
  python cns_mc.py calibrate [--data DIR] [--p 8]            -> results/cns_calibration.json
  python cns_mc.py run [--reps 500] [--workers K] [--cells sim1_cal,...] [--N n_counties] [--chunk 50]
                                                             -> results/cns_mc/<cell>.csv (resumable)
  python cns_mc.py run --mode lp [...]                       -> PVAR-IV against local projection IV, results/cns_mc_growth_lp/
  python cns_mc.py run --mode clr [...]                      -> several instruments: AR(k=4), CLR(k=4), AR(k=1), results/cns_mc_growth_clr/
  python cns_mc.py run --mode gmm [...]                      -> FOD-GMM on the same panels, results/cns_mc_v5_gmm/
  python cns_mc.py tune                                      -> rescales psi_u so that the median two-way F of sim1_cal
                                                                matches the data (old cell files are moved aside)
  add --growth to any step for the specification with payroll growth (cns_calibration_growth.json, cns_mc_growth/)
  python cns_mc.py all [--growth]                           -> calibrate, tune, every design, FOD-GMM, table
  python cns_mc.py table                                     -> results/cns_mc_summary.csv, tables/table_cns_mc.tex
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')        # one thread per worker
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import json
import argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))   # project root
sys.path.insert(0, os.path.join(_ROOT, 'package'))
_CNS = os.path.join(_ROOT, 'applications', '04_chodorowreich', 'python')


def _find_cns():
    """Folder containing cns.py (the application's data module): the standard location of the replication package,
    or, if the folders were moved, the first cns.py found under the parent of simulations/."""
    if os.path.exists(os.path.join(_CNS, 'cns.py')):
        return _CNS
    for root, dirs, files in os.walk(_ROOT):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('results', 'data', '__pycache__')]
        if 'cns.py' in files:
            return root
    raise FileNotFoundError(
        f"cns.py not found. Expected {_CNS}. Keep simulations/ and applications/ in the same folder, "
        "as in the replication package.")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy import stats
import pandas as pd
from multiprocessing import Pool
from pvariv import PVARIV
from pvariv.inference import ma_matrices

GROWTH = '--growth' in sys.argv       # outcome in quarterly growth, responses cumulated to the level
CAL_FILE = os.path.join(_HERE, 'results', 'cns_calibration_growth.json' if GROWTH else 'cns_calibration.json')
MC_DIR = os.path.join(_HERE, 'results', 'cns_mc_growth' if GROWTH else 'cns_mc_v5')
HS = (0, 4, 7, 12)
H = max(HS)
BURN = 200
METHODS = ('ar_unit', 'ar_time', 'ar_twoway', 'wald_twoway', 'hpj_ar_twoway', 'hpj_wald_twoway')
GMM_METHODS = ('gmm_ar_twoway', 'gmm_wald_twoway')

# cell -> (experiment, label, c, rho_g, endogeneity); None = calibrated value / no added endogeneity.  The calibrated
# endogeneity is close to zero, and a weak instrument then leaves the Wald interval unbiased, so Sim 2 builds in a
# correlation of ENDOG_SIM2 between the outcome innovation and the part of the policy innovation the instrument does
# not move: eps2 <- rho sqrt(v2/v1) (psi_u S_i a_t + u_it) + sqrt(1 - rho^2) eps2, with v1, v2 the variances of the
# two parts, which keeps the variance of eps2 unchanged.
ENDOG_SIM2 = 0.9
CELLS = {
    'sim1_cal':   ('Sim 1', 'Calibrated', 1.0, None, None),
    'sim2_c040':  ('Sim 2', 'Weak, $c=0.4$', 0.4, None, ENDOG_SIM2),
    'sim2_c020':  ('Sim 2', 'Weak, $c=0.2$', 0.2, None, ENDOG_SIM2),
    'sim3_r067':  ('Sim 3', r'Persistent, $\rho_g=0.67$', 1.0, 0.67, None),
    'sim3_r090':  ('Sim 3', r'Persistent, $\rho_g=0.9$', 1.0, 0.90, None),
}

# Simulation with several instruments: cell -> (label, c).  CLR_EXTRA lagged-shift instruments are added to the
# shift-share instrument, so k = 1 + CLR_EXTRA.
CLR_EXTRA = 3
CLR_CELLS = {
    'sim4_cal':  ('Calibrated, $k=4$', 1.0),
    'sim4_c040': ('Weak, $c=0.4$, $k=4$', 0.4),
}

# Comparison with local projections (supplement, Section S4): cell -> (label, main design whose panels are reused or
# None for a new design, lag order used by both estimators or None for the calibrated one, news loading a).  Reusing
# the seeds of a main design means that the PVAR-IV results are those already stored for it; for a new design, or a
# different lag order, both estimators are computed.  The anticipated design adds a times the next innovation of the
# shift to the instrument, so that the instrument is correlated with the next policy shock (no lead exogeneity).
LP_CELLS = {
    'lp_cal':  ('Calibrated', 'sim1_cal', None, 0.0),
    'lp_r090': (r'Persistent, $\rho_g=0.9$', 'sim3_r090', None, 0.0),
    'lp_news': ('Anticipated, $a=1$', None, None, 1.0),
}


# ------------------------------------------------------------------------------------------------ calibration
def to_growth(df):
    """Outcome (100 x log payroll) in quarterly changes, as in run_main_application.py --growth."""
    df = df.sort_values(['COUNTYID', 'QUARTER']).copy()
    g = df.groupby('COUNTYID')
    consecutive = (df.QUARTER - g.QUARTER.shift(1)) == 1
    df['y'] = np.where(consecutive, df.y - g.y.shift(1), np.nan)
    return df[df.y.notna()].reset_index(drop=True)


def calibrate(data, p):
    cns_dir = _find_cns()
    sys.path.insert(0, cns_dir)
    import cns
    if not os.path.exists(data):                 # folders moved: data/ next to python/
        data = os.path.join(os.path.dirname(cns_dir), 'data')
    print('application module:', os.path.join(cns_dir, 'cns.py'), '| data:', data, flush=True)
    df = cns.load(data, 9)
    if GROWTH:
        df = to_growth(df)
    dat = cns.prepare(df, weighted=False, effects='twoway')
    m = PVARIV(dat, 'COUNTYID', 'QUARTER', ['s', 'y'], 'z', p=p, exog=cns.CONFIG['exog'], policy=0,
               method='within').fit()
    eta1, eta2, z = m.eta[0], m.eta[1], m.z - m.z.mean()
    tq = m.clusters['time']
    units = np.unique(m.clusters['unit'])

    raw = cns.read_xpt(data)
    raw = raw[raw.SAMPLE == 1]
    S = raw.groupby('COUNTYID').L8_WEAL2.mean().reindex(units).dropna()
    g = raw.groupby('QUARTER').XRETCOUN.mean().sort_index().dropna()
    gv = g.values
    rho_g = float(np.dot(gv[1:] - gv[1:].mean(), gv[:-1] - gv[:-1].mean()) / np.sum((gv[:-1] - gv[:-1].mean()) ** 2))
    e = (gv[1:] - gv[1:].mean()) - rho_g * (gv[:-1] - gv[:-1].mean())
    sigma_e = float(e.std(ddof=2))

    St = S.values - S.values.mean()
    mS2 = float(np.mean(St ** 2))
    pi = float(z @ eta1 / (z @ z))
    gamma = pi / (1 - rho_g ** 2)
    beta = float(m.Gamma[1] / m.Gamma[0])
    _, ic = np.unique(tq, return_inverse=True)
    A, C = float(np.sum(St ** 2)), float(np.sum(St ** 4))

    def factor_split(resid):
        """(loading psi, idiosyncratic variance, ratio R) such that psi S_i f_t + idio reproduces the variance of
        resid and the quarter-clustered / robust variance ratio of the moment z * resid."""
        mom = z * resid
        R = float(np.sum(np.bincount(ic, weights=mom) ** 2) / np.sum(mom ** 2))
        q = A * (R - 1) / (A ** 2 - R * C) if (R > 1 and A ** 2 > R * C) else 0.0
        s2 = float(np.var(resid)) / (1 + q * mS2)
        return float(np.sqrt(q * s2)), s2, R

    u_hat = eta1 - pi * z
    psi_u, s_u2, R_u = factor_split(u_hat)
    s_u2_floor = 0.05 * float(np.var(eta1))
    s_u2 = max(s_u2, s_u2_floor)
    v = eta2 - beta * eta1
    psi, s_nu2, R = factor_split(v)
    r = float(np.corrcoef(u_hat, v)[0, 1])

    sd_alpha = [float(df.groupby('COUNTYID')[c].mean().std()) for c in ('s', 'y')]

    AL = m.AL
    Ac = np.zeros((2 * p, 2 * p)); Ac[:2, :] = AL
    if p > 1:
        Ac[2:, :2 * (p - 1)] = np.eye(2 * (p - 1))
    maxeig = float(np.max(np.abs(np.linalg.eigvals(Ac))))

    fs = m.first_stage('twoway')
    cal = dict(p=p, N=int(len(S)), T=int(len(np.unique(tq))), AL=AL.tolist(), beta=beta, gamma=gamma, pi=pi,
               rho_g=rho_g, sigma_e=sigma_e, S=S.values.tolist(), s_u=float(np.sqrt(s_u2)), psi_u=psi_u, ratio_first_stage=R_u,
               s_u_floor_binding=bool(s_u2 == s_u2_floor), s_nu=float(np.sqrt(s_nu2)), psi=psi, r=r,
               ratio_time_het=R, sd_alpha=sd_alpha, max_eig_companion=maxeig,
               data_F_twoway=float(fs['F']), data_irf=[float(x) for x in m.irf(H)[1]], cumulative=GROWTH)
    os.makedirs(os.path.dirname(CAL_FILE), exist_ok=True)
    json.dump(cal, open(CAL_FILE, 'w'), indent=1)
    show = {k: v for k, v in cal.items() if k not in ('AL', 'S', 'data_irf')}
    print(json.dumps(show, indent=1))
    print('true response at h =', HS, ':', [round(truth(cal, h), 4) for h in HS])
    if maxeig >= 1:
        print('WARNING: the calibrated PVAR is not stable (max |eigenvalue| >= 1)')
    print('written', CAL_FILE)


def truth(cal, h):
    """True response of the outcome at horizon h; with cal['cumulative'], the sum over horizons 0..h."""
    p = cal['p']
    C = ma_matrices(np.array(cal['AL']), 2, p, h)
    b = np.array([1.0, cal['beta']])
    if cal.get('cumulative', False):
        return float(sum(C[k][1] @ b for k in range(h + 1)))
    return float(C[h][1] @ b)


# ------------------------------------------------------------------------------------------------ one replication
_CAL = None


def _init(cal):
    global _CAL
    _CAL = cal


def simulate(cal, c, rho_g, N, seed, endog=None, demean='twoway', extra_lags=0, news=0.0):
    rng = np.random.default_rng(seed)
    p, T = cal['p'], cal['T']
    AL = np.array(cal['AL'])                       # 2 x 2p, [Phi_1 ... Phi_p]
    S = np.array(cal['S'])
    if N < len(S):
        S = rng.choice(S, N, replace=False)
    N = len(S)
    TT = T + BURN
    e = rng.normal(0, cal['sigma_e'], TT)
    g = np.zeros(TT)
    for t in range(1, TT):
        g[t] = rho_g * g[t - 1] + e[t]
    f = rng.normal(0, 1, TT)
    a = rng.normal(0, 1, TT)
    su, sn, r = cal['s_u'], cal['s_nu'], cal['r']
    w1 = rng.normal(size=(N, TT)); w2 = rng.normal(size=(N, TT))
    u = su * w1
    nu = sn * (r * w1 + np.sqrt(1 - r ** 2) * w2)
    eps1 = c * cal.get('c_cal', 1.0) * cal['gamma'] * S[:, None] * e[None, :] + cal['psi_u'] * S[:, None] * a[None, :] + u
    eps2 = cal['psi'] * S[:, None] * f[None, :] + nu
    if endog is not None:                # Sim 2: confounding of the outcome with the non-instrumented policy part
        vS = float(np.var(S))
        v1 = cal['psi_u'] ** 2 * vS + su ** 2
        v2 = cal['psi'] ** 2 * vS + sn ** 2
        eps2 = endog * np.sqrt(v2 / v1) * (cal['psi_u'] * S[:, None] * a[None, :] + u) + np.sqrt(1 - endog ** 2) * eps2
    inn = np.stack([eps1, cal['beta'] * eps1 + eps2], axis=2)          # N x TT x 2
    alpha = np.column_stack([rng.normal(0, cal['sd_alpha'][0], N), rng.normal(0, cal['sd_alpha'][1], N)])
    Phi = [AL[:, 2 * l:2 * l + 2] for l in range(p)]
    mu = np.linalg.solve(np.eye(2) - sum(Phi), alpha.T).T                # unit means, start of the recursion
    x = np.zeros((N, TT, 2)); x[:, :p, :] = mu[:, None, :]
    for t in range(p, TT):
        acc = alpha + inn[:, t, :]
        for l in range(p):
            acc = acc + x[:, t - 1 - l, :] @ Phi[l].T
        x[:, t, :] = acc
    shift = g
    if news:                             # anticipation: the instrument also carries the next innovation of the shift
        e_next = np.r_[e[1:], rng.normal(0, cal['sigma_e'])]          # drawn last: other draws are unchanged
        shift = g + news * e_next
    x, Z = x[:, BURN:, :], (S[:, None] * shift[None, :])[:, BURN:]
    unit = np.repeat(np.arange(N), T); time = np.tile(np.arange(T), N)
    d = pd.DataFrame(dict(unit=unit, time=time, s=x[:, :, 0].ravel(), y=x[:, :, 1].ravel(), z=Z.ravel()))
    zcols = ['z']
    for l in range(1, extra_lags + 1):  # additional instruments: exposure times lagged shifts (CLR simulation)
        d[f'z{l}'] = (S[:, None] * g[None, :])[:, BURN - l:TT - l].ravel()
        zcols.append(f'z{l}')
    for col in ['s', 'y'] + zcols:          # balanced panel: exact two-way demeaning, or time effects only (FOD-GMM)
        v = d[col].values.reshape(N, T)
        v = v - v.mean(1, keepdims=True) - v.mean(0, keepdims=True) + v.mean() if demean == 'twoway' \
            else v - v.mean(0, keepdims=True)
        d[col] = v.ravel()
    return d


def covered(lo, hi, case, th):
    if case == 1:
        return lo <= th <= hi
    if case == 2:
        return th <= lo or th >= hi
    return case == 4


def gmm_rep(cell, rep, N):
    """FOD-GMM (Arellano and Bover 1995) on the same simulated panel (same seed): time effects removed, county
    effects left to FOD, lagged levels as instruments."""
    cal = _CAL
    _, _, c, rho, r_cell = CELLS[cell]
    rho_g = cal['rho_g'] if rho is None else rho
    seed = 1_000_003 * (list(CELLS).index(cell) + 1) + rep
    d = simulate(cal, c, rho_g, N, seed, r_cell, demean='time')
    m = PVARIV(d, 'unit', 'time', ['s', 'y'], 'z', p=cal['p'], policy=0, method='fod-gmm').fit()
    irf = m.irf(H)[1]
    if cal.get('cumulative', False):
        irf = np.cumsum(irf)
    out = dict(cell=cell, rep=rep, psi_u_eff=cal['psi_u'], gmm_F_twoway=float(m.first_stage('twoway')['F']))
    for cl in ('time', 'twoway'):
        sets, case = m.ar_sets(H, cl, cumulative=cal.get('cumulative', False))
        for h in HS:
            lo, hi = sets[1, h]; k = int(case[1, h]); th = truth(cal, h)
            out[f'gmm_ar_{cl}_cov_h{h}'] = bool(covered(lo, hi, k, th))
            out[f'gmm_ar_{cl}_len_h{h}'] = float(hi - lo) if k == 1 else np.inf
            out[f'gmm_ar_{cl}_case_h{h}'] = k
    w = m.plugin_intervals(H, 'twoway', cumulative=cal.get('cumulative', False))
    for h in HS:
        lo, hi = w.iloc[h]['lo95'], w.iloc[h]['hi95']; th = truth(cal, h)
        out[f'gmm_wald_twoway_cov_h{h}'] = bool(lo <= th <= hi)
        out[f'gmm_wald_twoway_len_h{h}'] = float(hi - lo)
        out[f'gmm_est_h{h}'] = float(irf[h])
    return out


def clr_rep(cell, rep, N):
    """Several instruments: the shift-share instrument and the exposure times the shift lagged 1..3 quarters.
    The lagged instruments are valid (uncorrelated with every shock other than the policy shock) and carry no
    information on the current policy innovation, the situation in which the AR test with k degrees of freedom loses
    power and the conditional likelihood ratio test should not.  Compares, for the cumulated response, AR with all
    instruments (chi2(4)), CLR with all instruments, and the AR set with the relevant instrument alone (the
    infeasible benchmark of a researcher who knows which instrument is relevant).  Within estimator with the
    half-panel jackknife, two-way clustering."""
    from pvariv.clr import reject_at
    cal = _CAL
    c = CLR_CELLS[cell][1]
    seed = 7_000_003 * (list(CLR_CELLS).index(cell) + 1) + rep
    d = simulate(cal, c, cal['rho_g'], N, seed, extra_lags=CLR_EXTRA)
    inst = ['z'] + [f'z{l}' for l in range(1, CLR_EXTRA + 1)]
    m = PVARIV(d, 'unit', 'time', ['s', 'y'], inst, p=cal['p'], policy=0, method='within', bias_correction='hpj',
               pooling='2sls').fit()
    m1 = PVARIV(d, 'unit', 'time', ['s', 'y'], 'z', p=cal['p'], policy=0, method='within',
                bias_correction='hpj').fit()
    W = m.covariance('twoway')
    sets1, case1 = m1.ar_sets(H, 'twoway', cumulative=True)
    out = dict(cell=cell, rep=rep, psi_u_eff=cal['psi_u'], clr_F_twoway=float(m.first_stage('twoway')['F']),
               clr_F1_twoway=float(m1.first_stage('twoway')['F']))
    for h in HS:
        th = truth(cal, h)
        nulls = {'true': th, 'zero': 0.0, 'half': 0.5 * th, 'double': 2.0 * th}
        rj = reject_at(np.array(list(nulls.values())), m.AL, m.GammaK, W, 2, cal['p'], m.T, 0, 1, h,
                       cumulative=True)
        lo, hi = sets1[1, h]; k1 = int(case1[1, h])
        for i, name in enumerate(nulls):
            out[f'clr_ar4_rej_{name}_h{h}'] = bool(rj['ar'][i])
            out[f'clr_clr_rej_{name}_h{h}'] = bool(rj['clr'][i])
            out[f'clr_ar1_rej_{name}_h{h}'] = not covered(lo, hi, k1, nulls[name])
        out[f'clr_ar_twoway_cov_h{h}'] = not bool(rj['ar'][0])        # AR, all instruments: coverage of the truth
    return out


def _twoway_meat(M):
    """Two-way clustered meat of a moment array M (N x Tv x q), units in rows and periods in columns."""
    M = M - M.reshape(-1, M.shape[2]).mean(0)
    U, P = M.sum(1), M.sum(0)
    F = M.reshape(-1, M.shape[2])
    W = U.T @ U + P.T @ P - F.T @ F
    val, vec = np.linalg.eigh((W + W.T) / 2)
    return (vec * np.clip(val, 0, None)) @ vec.T


def _lp_window(S, Yg, Z, p, h, ta, tb, ar=True, level=0.95):
    """Local projection IV at horizon h on the periods t = ta, ..., tb-1 (which must allow p lags and h leads):
    the change sum_{k=0}^h y_{i,t+k} (= Y_{t+h} - Y_{t-1}) regressed on s_it instrumented by z_it, with p lags of
    (s, y) as controls and unit and period effects of the window.  Returns the estimate and, if ar, the
    Anderson-Rubin set (lo, hi, case as in covered) with two-way clustering and the two-way first-stage F."""
    lhs = sum(Yg[:, ta + k:tb + k] for k in range(h + 1))
    cols = [S[:, ta:tb], lhs, Z[:, ta:tb]]
    ctrl = [S[:, ta - l:tb - l] for l in range(1, p + 1)] + [Yg[:, ta - l:tb - l] for l in range(1, p + 1)]
    dm = lambda A: A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    w, y, z = (dm(A).ravel() for A in cols)
    C = np.column_stack([dm(A).ravel() for A in ctrl])
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    w, y, z = res(w), res(y), res(z)
    est = float(z @ y / (z @ w))
    if not ar:
        return est
    Nn, Tv = lhs.shape
    pi = z @ w / (z @ z)
    u = w - pi * z
    Vpi = _twoway_meat((z * u).reshape(Nn, Tv, 1))[0, 0] / (z @ z) ** 2
    F = float(pi ** 2 / Vpi) if Vpi > 0 else np.nan
    n = len(z)
    a, b = (z * y).mean(), (z * w).mean()
    V = _twoway_meat(np.stack([z * y, z * w], -1).reshape(Nn, Tv, 2)) / n ** 2
    crit = stats.chi2.ppf(level, 1)
    A = b ** 2 - crit * V[1, 1]; B = -2 * a * b + 2 * crit * V[0, 1]; Cq = a ** 2 - crit * V[0, 0]
    disc = B ** 2 - 4 * A * Cq
    if A > 0:
        if disc < 0:
            return est, np.nan, np.nan, 3, F
        r1, r2 = sorted([(-B - np.sqrt(disc)) / (2 * A), (-B + np.sqrt(disc)) / (2 * A)])
        return est, r1, r2, 1, F
    if disc <= 0:
        return est, -np.inf, np.inf, 4, F
    r1, r2 = sorted([(-B - np.sqrt(disc)) / (2 * A), (-B + np.sqrt(disc)) / (2 * A)])
    return est, r1, r2, 2, F


def lp_iv(d, p, h, level=0.95):
    """Local projection IV of the level response at horizon h on the whole sample (estimate, AR set, F), and its
    split-panel jackknife correction (Mei, Sheng and Shi): 2 * full - (first half + second half) / 2, the halves
    splitting the periods of the projection, each with its own unit and period effects."""
    N = d.unit.nunique(); T = len(d) // N
    S = d.s.values.reshape(N, T); Yg = d.y.values.reshape(N, T); Z = d.z.values.reshape(N, T)
    t0, t1 = p, T - h
    est, lo, hi, k, F = _lp_window(S, Yg, Z, p, h, t0, t1, True, level)
    tm = t0 + (t1 - t0) // 2
    e1 = _lp_window(S, Yg, Z, p, h, t0, tm, False)
    e2 = _lp_window(S, Yg, Z, p, h, tm, t1, False)
    return est, lo, hi, k, F, 2 * est - 0.5 * (e1 + e2)


def lp_rep(cell, rep, N):
    """One replication of the comparison with local projections.  The panel is the one of the main design named in
    LP_CELLS (same seed), or a new one.  Local projections (with and without the split-panel jackknife) always;
    PVAR-IV (jackknife-corrected within estimator, two-way AR sets) only when its results are not stored already."""
    cal = _CAL
    lab, base, p_est, news = LP_CELLS[cell]
    if base is not None:
        _, _, c, rho, r_cell = CELLS[base]
        seed = 1_000_003 * (list(CELLS).index(base) + 1) + rep          # the seed of the main design
    else:
        c, rho, r_cell = 1.0, None, None
        seed = 13_000_019 * (list(LP_CELLS).index(cell) + 1) + rep
    rho_g = cal['rho_g'] if rho is None else rho
    d = simulate(cal, c, rho_g, N, seed, r_cell, news=news)
    p = cal['p'] if p_est is None else p_est
    out = dict(cell=cell, rep=rep, psi_u_eff=cal['psi_u'], p_est=p)
    for h in HS:
        est, lo, hi, k, F, spj = lp_iv(d, p, h)
        th = truth(cal, h)
        out[f'lp_est_h{h}'] = est
        out[f'lpspj_est_h{h}'] = spj
        out[f'lp_ar_twoway_cov_h{h}'] = bool(covered(lo, hi, k, th))
        out[f'lp_ar_twoway_len_h{h}'] = float(hi - lo) if k == 1 else np.inf
        if h == 0:
            out['lp_F_twoway'] = F
    if base is None or (p_est is not None and p_est != cal['p']):
        m = PVARIV(d, 'unit', 'time', ['s', 'y'], 'z', p=p, policy=0, method='within', bias_correction='hpj').fit()
        irf = np.cumsum(m.irf(H)[1])
        sets, case = m.ar_sets(H, 'twoway', cumulative=True)
        for h in HS:
            lo, hi = sets[1, h]; k = int(case[1, h]); th = truth(cal, h)
            out[f'pvar_est_h{h}'] = float(irf[h])
            out[f'pvar_ar_twoway_cov_h{h}'] = bool(covered(lo, hi, k, th))
            out[f'pvar_ar_twoway_len_h{h}'] = float(hi - lo) if k == 1 else np.inf
    return out


def one_rep(args):
    cell, rep, N, mode = args
    if mode == 'gmm':
        return gmm_rep(cell, rep, N)
    if mode == 'clr':
        return clr_rep(cell, rep, N)
    if mode == 'lp':
        return lp_rep(cell, rep, N)
    cal = _CAL
    _, _, c, rho, r_cell = CELLS[cell]
    rho_g = cal['rho_g'] if rho is None else rho
    seed = 1_000_003 * (list(CELLS).index(cell) + 1) + rep
    d = simulate(cal, c, rho_g, N, seed, r_cell)
    out = dict(cell=cell, rep=rep, c_eff=c * cal.get('c_cal', 1.0), psi_u_eff=cal['psi_u'],
               endog=np.nan if r_cell is None else r_cell)
    for prefix, bc in (('', None), ('hpj_', 'hpj')):
        m = PVARIV(d, 'unit', 'time', ['s', 'y'], 'z', p=cal['p'], policy=0, method='within',
                   bias_correction=bc).fit()
        irf = m.irf(H)[1]
        if cal.get('cumulative', False):
            irf = np.cumsum(irf)
        for cl in ('unit', 'time', 'twoway'):
            if not prefix:
                out[f'F_{cl}'] = float(m.first_stage(cl)['F'])
            sets, case = m.ar_sets(H, cl, cumulative=cal.get('cumulative', False))
            for h in HS:
                lo, hi = sets[1, h]; k = int(case[1, h]); th = truth(cal, h)
                out[f'{prefix}ar_{cl}_cov_h{h}'] = bool(covered(lo, hi, k, th))
                out[f'{prefix}ar_{cl}_len_h{h}'] = float(hi - lo) if k == 1 else np.inf
                out[f'{prefix}ar_{cl}_case_h{h}'] = k
        w = m.plugin_intervals(H, 'twoway', cumulative=cal.get('cumulative', False))
        for h in HS:
            lo, hi = w.iloc[h]['lo95'], w.iloc[h]['hi95']; th = truth(cal, h)
            out[f'{prefix}wald_twoway_cov_h{h}'] = bool(lo <= th <= hi)
            out[f'{prefix}wald_twoway_len_h{h}'] = float(hi - lo)
            out[f'{prefix}est_h{h}'] = float(irf[h])
    return out


def run(reps, workers, cells, N, chunk, mode='main'):
    """mode 'main': within estimator with and without jackknife, results in MC_DIR; mode 'gmm': FOD-GMM on the same
    panels, results in MC_DIR + '_gmm' (merged by replication in the table)."""
    cal = json.load(open(CAL_FILE))
    if not _valid_calibration(cal):
        raise RuntimeError(f'{CAL_FILE} contains missing values; delete it and run the calibrate step (or all) again')
    N = N or cal['N']
    out_dir = {'main': MC_DIR, 'gmm': MC_DIR + '_gmm', 'clr': MC_DIR + '_clr', 'lp': MC_DIR + '_lp2'}[mode]
    os.makedirs(out_dir, exist_ok=True)
    pre = {'main': '', 'gmm': 'gmm_', 'clr': 'clr_', 'lp': 'lp_'}[mode]
    with Pool(workers, initializer=_init, initargs=(cal,)) as pool:
        for cell in cells:
            f = os.path.join(out_dir, f'{cell}.csv')
            done = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
            if len(done) and ('psi_u_eff' not in done or not np.isclose(done.psi_u_eff, cal['psi_u']).all()):
                # rows produced under another calibration (e.g. stored results copied into this folder): set them
                # aside and start the design again, so that resuming never mixes calibrations
                stale = os.path.join(out_dir, 'archive_stale')
                os.makedirs(stale, exist_ok=True)
                os.replace(f, os.path.join(stale, f'{cell}.csv'))
                print(f'{cell} [{mode}]: existing file from another calibration moved to {stale}', flush=True)
                done = pd.DataFrame()
            start = int(done.rep.max()) + 1 if len(done) else 0
            while start < reps:
                stop = min(start + chunk, reps)
                rows = pool.map(one_rep, [(cell, r, N, mode) for r in range(start, stop)])
                new = pd.DataFrame(rows)
                done = pd.concat([done, new], ignore_index=True) if len(done) else new
                done.to_csv(f, index=False)
                cov = done[f'{pre}ar_twoway_cov_h0'].mean()
                print(f'{cell} [{mode}]: {len(done)}/{reps} reps, median two-way F {done[pre + "F_twoway"].median():.1f}, '
                      f'AR two-way coverage h=0 {cov:.3f}', flush=True)
                start = stop


# ------------------------------------------------------------------------------------------------ tune
def tune():
    """Match the median two-way first-stage F of sim1_cal to the application by scaling the common first-stage
    noise psi_u (the loading gamma stays at its calibrated value, so the second stage keeps the precision of the
    application).  The quarter-clustered first-stage variance is dominated by the factor term, so
    psi_u <- psi_u * sqrt(median F / F_data).  Run sim1_cal with a few replications, tune, repeat."""
    cal = json.load(open(CAL_FILE))
    if 'psi_u_calibrated' not in cal:                          # first call: switch from scaling c to scaling psi_u
        cal['psi_u_calibrated'] = cal['psi_u']
        c_old = cal.get('c_cal', 1.0)
        cal['c_cal'] = 1.0
        cal['psi_u'] = cal['psi_u'] / c_old                    # same first-stage signal-to-noise as the c-tuned design
        json.dump(cal, open(CAL_FILE, 'w'), indent=1)
        print(f'switched to noise tuning: c_cal {c_old:.4f} -> 1, psi_u {cal["psi_u_calibrated"]:.5f} -> {cal["psi_u"]:.5f}')
        print('now run sim1_cal with a few replications and tune again')
        return
    d = pd.read_csv(os.path.join(MC_DIR, 'sim1_cal.csv'))
    old = cal['psi_u']
    if 'psi_u_eff' in d:
        d = d[np.isclose(d.psi_u_eff, old)]
    if d.empty:
        raise RuntimeError('tune: no replications of sim1_cal at the current calibration; run sim1_cal first')
    medF = float(d.F_twoway.median())
    cal['psi_u'] = float(old * np.sqrt(medF / cal['data_F_twoway']))
    json.dump(cal, open(CAL_FILE, 'w'), indent=1)
    arch = os.path.join(MC_DIR, f'archive_psiu{old:.5f}')
    os.makedirs(arch, exist_ok=True)
    for fn in os.listdir(MC_DIR):
        if fn.endswith('.csv'):
            os.replace(os.path.join(MC_DIR, fn), os.path.join(arch, fn))
    print(f'median two-way F {medF:.1f} ({len(d)} reps) vs data {cal["data_F_twoway"]:.1f}: psi_u {old:.5f} -> {cal["psi_u"]:.5f}')
    print(f'cell files moved to {arch}')


# ------------------------------------------------------------------------------------------------ everything
def _valid_calibration(cal):
    """Every numerical entry of the calibration is finite."""
    def finite(v):
        if isinstance(v, bool) or v is None or isinstance(v, str):
            return True
        if isinstance(v, (list, tuple)):
            return all(finite(x) for x in v)
        return bool(np.isfinite(v))
    return all(finite(v) for v in cal.values())


def run_all(data, p, reps, workers, N, chunk, tol=0.05, max_rounds=8):
    """Calibrate, tune psi_u until the median two-way F of sim1_cal is within tol of the application, run every
    design and the FOD-GMM comparison, and write the table.  Resumable: rerunning skips finished designs."""
    if os.path.exists(CAL_FILE) and not _valid_calibration(json.load(open(CAL_FILE))):
        bad = CAL_FILE.replace('.json', '_invalid.json')
        os.replace(CAL_FILE, bad)
        print(f'calibration file contains missing values (from an interrupted run): moved to {bad}, recalibrating',
              flush=True)
    if not os.path.exists(CAL_FILE):
        calibrate(data, p)
    cal = json.load(open(CAL_FILE))
    if 'psi_u_calibrated' not in cal:
        tune()                                                   # switch to noise tuning
    for rnd in range(max_rounds):
        run(50, workers, ['sim1_cal'], N, chunk)
        d = pd.read_csv(os.path.join(MC_DIR, 'sim1_cal.csv'))
        cal = json.load(open(CAL_FILE))
        d = d[np.isclose(d.psi_u_eff, cal['psi_u'])]
        gap = abs(d.F_twoway.median() / cal['data_F_twoway'] - 1)
        print(f'tuning round {rnd + 1}: median F {d.F_twoway.median():.1f}, target {cal["data_F_twoway"]:.1f}', flush=True)
        if gap <= tol:
            break
        tune()
    else:
        print('WARNING: tuning did not converge; continuing with the last value', flush=True)
    run(reps, workers, list(CELLS), N, chunk)
    run(reps, workers, ['sim1_cal'], N, chunk, mode='gmm')
    run(reps, workers, list(CLR_CELLS), N, chunk, mode='clr')
    run(reps, workers, list(LP_CELLS), N, chunk, mode='lp')
    table()


# ------------------------------------------------------------------------------------------------ table
def table():
    cal = json.load(open(CAL_FILE))
    rows = []
    for cell, (exp, lab, c, rho, r_cell) in CELLS.items():
        f = os.path.join(MC_DIR, f'{cell}.csv')
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        fg = os.path.join(MC_DIR + '_gmm', f'{cell}.csv')
        has_gmm = os.path.exists(fg)
        if has_gmm:
            d = d.merge(pd.read_csv(fg).drop(columns='cell'), on='rep', how='inner')
        for h in HS:
            th = truth(cal, h)
            r = dict(cell=cell, experiment=exp, design=lab, reps=len(d), F_twoway=d.F_twoway.median(),
                     F_unit=d.F_unit.median(), h=h, truth=th,
                     bias=d[f'est_h{h}'].median() - th, hpj_bias=d[f'hpj_est_h{h}'].median() - th)
            for mth in METHODS + (GMM_METHODS if has_gmm else ()):
                r[f'{mth}_cov'] = 100 * d[f'{mth}_cov_h{h}'].mean()
                r[f'{mth}_medlen'] = d[f'{mth}_len_h{h}'].median()
            r['gmm_bias'] = d[f'gmm_est_h{h}'].median() - th if has_gmm else np.nan
            rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(_HERE, 'results', 'cns_mc_summary.csv'), index=False)

    gm = 'gmm_ar_twoway_cov' in s and s['gmm_ar_twoway_cov'].notna().any()

    def fmt_len(x):
        return 'unb.' if not np.isfinite(x) else f'{x:.2f}'
    def coverage_table(experiments, caption_extra, label):
        L = [r'\begin{table}[ht]\centering\footnotesize\setlength{\tabcolsep}{4pt}',
             r'\resizebox{\textwidth}{!}{\begin{tabular}{llr rrr rr rr}\toprule',
             r' & & & \multicolumn{3}{c}{Within estimator, coverage (\%)} & \multicolumn{2}{c}{Jackknife, coverage (\%)}'
             r' & \multicolumn{2}{c}{Median bias}\\',
             r'\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}',
             r'Design & $h$ & $F$ & AR, quarter & AR, two-way & Wald, two-way & AR, two-way & Wald, two-way'
             r' & Within & Jackknife\\\midrule']
        for exp in experiments:
            g = s[s.experiment == exp]
            for cell in dict.fromkeys(g.cell):
                gg = g[g.cell == cell]
                for i, r in enumerate(gg.itertuples()):
                    L.append(f"{r.design if i == 0 else ''} & {r.h} & {f'{r.F_twoway:.1f}' if i == 0 else ''} & "
                             f"{r.ar_time_cov:.1f} & {r.ar_twoway_cov:.1f} & {r.wald_twoway_cov:.1f} & "
                             f"{r.hpj_ar_twoway_cov:.1f} & {r.hpj_wald_twoway_cov:.1f} & {r.bias:.2f} & {r.hpj_bias:.2f}\\\\")
                L.append(r'\addlinespace')
        L += [r'\bottomrule\end{tabular}}',
              r'\caption{Coverage of nominal 95\% sets for the response of the outcome at horizon $h$ to a unit change in '
              r'the policy variable, and median bias of the point estimate. $F$: median first-stage statistic, two-way '
              r'clustered. AR: Anderson-Rubin set clustered by quarter or two-way. Wald: delta-method interval, '
              r'two-way clustered. Within: within estimator of the PVAR coefficients. Jackknife: half-panel jackknife '
              r'correction of the within estimator. ' + caption_extra
              + f"{int(s.reps.min()) if len(s) else 0} replications per design." + r'\label{' + label + r'}}',
              r'\end{table}']
        return '\n'.join(L)

    os.makedirs(os.path.join(_HERE, 'tables'), exist_ok=True)
    open(os.path.join(_HERE, 'tables', 'table_cns_mc.tex'), 'w').write(coverage_table(
        ('Sim 1', 'Sim 2'), r'Weak designs: correlation 0.9 between the outcome innovation and the part of the policy '
        r'innovation not moved by the instrument. ', 'tab:mc'))
    # supplement: the calibrated design with a persistent aggregate shift
    open(os.path.join(_HERE, 'tables', 'table_cns_mc_persistent.tex'), 'w').write(coverage_table(
        ('Sim 3',), r'Calibrated design with autocorrelation $\rho_g$ of the aggregate shift. ', 'tab:mc_persistent'))
    # supplement: coverage of the AR sets clustered by county alone, by quarter and two-way, every design
    K = [r'\begin{table}[ht]\centering\footnotesize',
         r'\begin{tabular}{llr rrr}\toprule',
         r' & & & \multicolumn{3}{c}{AR coverage, within estimator (\%)}\\ \cmidrule(lr){4-6}',
         r'Design & $h$ & $F$ & county & quarter & two-way\\\midrule']
    for exp in ('Sim 1', 'Sim 2', 'Sim 3'):
        g = s[s.experiment == exp]
        for cell in dict.fromkeys(g.cell):
            gg = g[g.cell == cell]
            for i, r in enumerate(gg.itertuples()):
                K.append(f"{r.design if i == 0 else ''} & {r.h} & {f'{r.F_twoway:.1f}' if i == 0 else ''} & "
                         f"{r.ar_unit_cov:.1f} & {r.ar_time_cov:.1f} & {r.ar_twoway_cov:.1f}\\\\")
            K.append(r'\addlinespace')
    K += [r'\bottomrule\end{tabular}',
          r'\caption{Coverage of nominal 95\% Anderson-Rubin sets for the cumulated response, within estimator, with '
          r'the covariance clustered by county alone, by quarter, and in both dimensions, for the designs of Section 6. '
          r'$F$: median first-stage statistic, two-way clustered. ' + f"{int(s.reps.min())} replications per design."
          + r'\label{tab:mc_county}}',
          r'\end{table}']
    open(os.path.join(_HERE, 'tables', 'table_cns_mc_county.tex'), 'w').write('\n'.join(K))
    if gm:      # supplement: FOD-GMM against the within estimator, calibrated design
        g = s[s.cell == 'sim1_cal']
        G = [r'\begin{table}[ht]\centering\footnotesize',
             r'\begin{tabular}{r rrr rr rr}\toprule',
             r' & \multicolumn{3}{c}{AR coverage, two-way (\%)} & \multicolumn{2}{c}{Median length} & \multicolumn{2}{c}{Median bias}\\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
             r'$h$ & Within & Jackknife & FOD-GMM & Jackknife & FOD-GMM & Jackknife & FOD-GMM\\\midrule']
        for r in g.itertuples():
            G.append(f"{r.h} & {r.ar_twoway_cov:.1f} & {r.hpj_ar_twoway_cov:.1f} & {r.gmm_ar_twoway_cov:.1f} & "
                     f"{fmt_len(r.hpj_ar_twoway_medlen)} & {fmt_len(r.gmm_ar_twoway_medlen)} & {r.hpj_bias:.2f} & {r.gmm_bias:.2f}\\\\")
        G += [r'\bottomrule\end{tabular}',
              r'\caption{Calibrated design of Section 6: the within estimator without and with the half-panel jackknife '
              r'correction, and GMM after forward orthogonal deviations with lagged levels as instruments. Coverage of '
              r'nominal 95\% Anderson-Rubin sets clustered by county and quarter, median length of the sets, and median '
              r'bias of the point estimate. ' + f"{int(g.reps.min())} replications." + r'\label{tab:mc_gmm}}',
              r'\end{table}']
        open(os.path.join(_HERE, 'tables', 'table_cns_mc_gmm.tex'), 'w').write('\n'.join(G))
    pd.set_option('display.width', 250)
    cols = ['design', 'h', 'F_twoway', 'bias', 'hpj_bias'] + [f'{m}_cov' for m in METHODS] \
        + ['ar_twoway_medlen', 'hpj_ar_twoway_medlen']
    if gm:
        cols += ['gmm_bias'] + [f'{m}_cov' for m in GMM_METHODS] + ['gmm_ar_twoway_medlen']
    print(s[cols].round(3).to_string())
    # several instruments: AR with k = 4, CLR with k = 4, and AR with the relevant instrument alone
    fdir = MC_DIR + '_clr'
    if os.path.isdir(fdir):
        C = [r'\begin{table}[ht]\centering\footnotesize\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{llr rrr rrr}\toprule',
             r' & & & \multicolumn{3}{c}{Coverage (\%)} & \multicolumn{3}{c}{Rejection of zero (\%)}\\',
             r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}',
             r'Design & $h$ & $F$ & AR, $k=1$ & AR, $k=4$ & CLR, $k=4$ & AR, $k=1$ & AR, $k=4$ & CLR, $k=4$\\\midrule']
        nrep = []
        for cell, (lab, c) in CLR_CELLS.items():
            f = os.path.join(fdir, f'{cell}.csv')
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f); nrep.append(len(d))
            for i, h in enumerate(HS):
                pct = lambda col: 100 * d[col].mean()
                C.append(f"{lab if i == 0 else ''} & {h} & {f'{d.clr_F_twoway.median():.1f}' if i == 0 else ''} & "
                         f"{100 - pct(f'clr_ar1_rej_true_h{h}'):.1f} & {100 - pct(f'clr_ar4_rej_true_h{h}'):.1f} & "
                         f"{100 - pct(f'clr_clr_rej_true_h{h}'):.1f} & "
                         f"{pct(f'clr_ar1_rej_zero_h{h}'):.1f} & {pct(f'clr_ar4_rej_zero_h{h}'):.1f} & "
                         f"{pct(f'clr_clr_rej_zero_h{h}'):.1f}\\\\")
            C.append(r'\addlinespace')
        C += [r'\bottomrule\end{tabular}',
              r'\caption{Several instruments: the shift-share instrument and the exposure times the shift lagged one to '
              r'three quarters ($k=4$). Coverage of nominal 95\% sets for the cumulated response at horizon $h$, and '
              r'rejection rates of the null that the response is zero. AR, $k=4$, and CLR, $k=4$: '
              r'Anderson-Rubin and conditional likelihood ratio tests with all instruments; AR, $k=1$: Anderson-Rubin '
              r'test with the relevant instrument alone. Within estimator with the half-panel jackknife correction, '
              r'two-way clustering. $F$: median first-stage statistic with all instruments, two-way clustered. '
              + f"{min(nrep) if nrep else 0} replications per design." + r'\label{tab:mc_clr}}',
              r'\end{table}']
        if nrep:
            open(os.path.join(_HERE, 'tables', 'table_cns_mc_clr.tex'), 'w').write('\n'.join(C))

    # comparison with local projections (supplement, Section S4)
    ldir = MC_DIR + '_lp2'
    if os.path.isdir(ldir):
        calj = json.load(open(CAL_FILE))
        fmt = lambda x: 'unb.' if not np.isfinite(x) else f'{x:.2f}'
        Lp = [r'\begin{table}[ht]\centering\footnotesize\setlength{\tabcolsep}{3pt}',
              r'\resizebox{\textwidth}{!}{\begin{tabular}{llr rrr rrr rr rr}\toprule',
              r' & & & \multicolumn{3}{c}{Median bias} & \multicolumn{3}{c}{Median abs.\ error}'
              r' & \multicolumn{2}{c}{AR coverage (\%)} & \multicolumn{2}{c}{Median AR length}\\',
              r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}',
              r'Design & $h$ & $F$ & PVAR & LP & LP-J & PVAR & LP & LP-J & PVAR & LP & PVAR & LP\\\midrule']
        rows, nrep = [], []
        for cell, (lab, base, p_est, news) in LP_CELLS.items():
            f = os.path.join(ldir, f'{cell}.csv')
            if not os.path.exists(f):
                continue
            dl = pd.read_csv(f)
            if base is not None and (p_est is None or p_est == calj['p']):
                dm_ = pd.read_csv(os.path.join(MC_DIR, f'{base}.csv'))
                dl = dl.merge(dm_, on='rep', suffixes=('', '_main'))
                pv = lambda q, h: dl[f'hpj_{q}_h{h}'] if q == 'est' else dl[f'hpj_ar_twoway_{q}_h{h}']
            else:
                pv = lambda q, h: dl[f'pvar_{q}_h{h}'] if q == 'est' else dl[f'pvar_ar_twoway_{q}_h{h}']
            nrep.append(len(dl))
            for i, h in enumerate(HS):
                th = truth(calj, h)
                ep, el, ej = pv('est', h) - th, dl[f'lp_est_h{h}'] - th, dl[f'lpspj_est_h{h}'] - th
                r = dict(cell=cell, design=lab, h=h, truth=th, reps=len(dl), lp_F_twoway=dl.lp_F_twoway.median(),
                         pvar_bias=ep.median(), lp_bias=el.median(), lpj_bias=ej.median(),
                         pvar_mae=ep.abs().median(), lp_mae=el.abs().median(), lpj_mae=ej.abs().median(),
                         pvar_rmse=float(np.sqrt((ep ** 2).mean())), lp_rmse=float(np.sqrt((el ** 2).mean())),
                         lpj_rmse=float(np.sqrt((ej ** 2).mean())),
                         pvar_cov=100 * pv('cov', h).mean(), lp_cov=100 * dl[f'lp_ar_twoway_cov_h{h}'].mean(),
                         pvar_len=pv('len', h).median(), lp_len=dl[f'lp_ar_twoway_len_h{h}'].median())
                rows.append(r)
                Lp.append(f"{lab if i == 0 else ''} & {h} & {f'{r['lp_F_twoway']:.1f}' if i == 0 else ''} & "
                          f"{r['pvar_bias']:.2f} & {r['lp_bias']:.2f} & {r['lpj_bias']:.2f} & "
                          f"{r['pvar_mae']:.2f} & {r['lp_mae']:.2f} & {r['lpj_mae']:.2f} & "
                          f"{r['pvar_cov']:.1f} & {r['lp_cov']:.1f} & {fmt(r['pvar_len'])} & {fmt(r['lp_len'])}\\\\")
            Lp.append(r'\addlinespace')
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(_HERE, 'results', 'cns_mc_lp_summary.csv'), index=False)
            Lp += [r'\bottomrule\end{tabular}}',
                   r'\caption{PVAR-IV and local projection IV of the cumulated response at horizon $h$, on the same '
                   r'simulated panels. PVAR: within estimator with the half-panel jackknife correction. LP: change of the '
                   r'level of the outcome from $t-1$ to $t+h$ regressed on the policy instrumented by $Z_{it}$, with $p$ '
                   r'lags of both variables as controls and county and quarter effects. LP-J: the same with the '
                   r'split-panel jackknife correction. Anticipated: the instrument also carries the next innovation of '
                   r'the aggregate shift, with loading $a$. Sets: nominal 95\% Anderson-Rubin sets with two-way '
                   r'clustering. Median absolute error: median of the absolute deviation of the estimate from the true '
                   r'response. $F$: median first-stage statistic of the local projection, two-way clustered. '
                   + f"{min(nrep)} replications per design." + r'\label{tab:mc_lp}}',
                   r'\end{table}']
            open(os.path.join(_HERE, 'tables', 'table_cns_mc_lp.tex'), 'w').write('\n'.join(Lp))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['calibrate', 'run', 'tune', 'table', 'all'])
    ap.add_argument('--data', default=os.path.join(_ROOT, 'applications', '04_chodorowreich', 'data'))
    ap.add_argument('--p', type=int, default=8)
    ap.add_argument('--reps', type=int, default=500)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--cells', default=','.join(CELLS))
    ap.add_argument('--N', type=int, default=None)
    ap.add_argument('--chunk', type=int, default=50)
    ap.add_argument('--mode', choices=['main', 'gmm', 'clr', 'lp'], default='main')
    ap.add_argument('--growth', action='store_true', help='payroll growth specification (read at import)')
    a = ap.parse_args()
    if a.step == 'calibrate':
        calibrate(a.data, a.p)
    elif a.step == 'run':
        cells = a.cells.split(',')
        if a.mode == 'clr' and a.cells == ','.join(CELLS):
            cells = list(CLR_CELLS)
        if a.mode == 'lp' and a.cells == ','.join(CELLS):
            cells = list(LP_CELLS)
        run(a.reps, a.workers, cells, a.N, a.chunk, a.mode)
    elif a.step == 'tune':
        tune()
    elif a.step == 'all':
        run_all(a.data, a.p, a.reps, a.workers, a.N, a.chunk)
    else:
        table()
