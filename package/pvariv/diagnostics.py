"""Lag selection and residual diagnostics."""
import numpy as np
import pandas as pd
from .utils import spell_ids, apply_fod


def select_lags(data, unit, time, endog, pmax=8, method='within'):
    """Information criteria for the VAR order on a COMMON estimation sample (observations with pmax lags available).

    method='within' (unit effects removed spell by spell) or 'pooled' (common constant).  The system criteria are
        BIC  = ln|Sigma_p| + K^2 p ln(n)/n,   AIC = ln|Sigma_p| + 2 K^2 p/n,   HQIC = ln|Sigma_p| + 2 K^2 p ln ln n/n,
    with Sigma_p = residual covariance of the p-lag VAR.  Returns a DataFrame indexed by p with the criteria, the
    common sample size n and the minimiser of each criterion in .attrs['choice'].

    For method='fod-gmm' the Andrews-Lu (2001) model and moment selection criteria are used instead (see
    _select_lags_gmm); the choice returned under 'bic' is then the MBIC minimiser."""
    if method == 'fod-gmm':
        return _select_lags_gmm(data, unit, time, endog, pmax)
    assert method in ('within', 'pooled')
    d = data.sort_values([unit, time]).reset_index(drop=True)
    sid = spell_ids(d, unit, time)
    K = len(endog)
    spells = []
    for s_ in np.unique(sid):
        Yn = d.loc[sid == s_, endog].values
        if len(Yn) > pmax + 1:
            spells.append(Yn)
    rows = []
    for p in range(1, pmax + 1):
        Ys, Xs = [], []
        for Yn in spells:
            T = len(Yn)
            Y = Yn[pmax:]
            X = np.hstack([Yn[pmax - l:T - l] for l in range(1, p + 1)])
            ok = ~(np.isnan(Y).any(1) | np.isnan(X).any(1))
            Y, X = Y[ok], X[ok]
            if len(Y) == 0:
                continue
            if method == 'within':
                Y, X = Y - Y.mean(0), X - X.mean(0)
            Ys.append(Y); Xs.append(X)
        Y, X = np.vstack(Ys), np.vstack(Xs)
        if method == 'pooled':
            X = np.hstack([np.ones((len(X), 1)), X])
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
        E = Y - X @ B
        n = len(Y)
        ld = np.linalg.slogdet(E.T @ E / n)[1]
        k = K * K * p
        rows.append(dict(p=p, n=n, aic=ld + 2 * k / n, bic=ld + k * np.log(n) / n, hqic=ld + 2 * k * np.log(np.log(n)) / n))
    out = pd.DataFrame(rows).set_index('p')
    out.attrs['choice'] = {c: int(out[c].idxmin()) for c in ('aic', 'bic', 'hqic')}
    return out


def resid_autocorr(model, lags=(1, 2, 3)):
    """Pooled autocorrelation of the VAR residuals of each equation at the given lags, with unit-clustered
    t-statistics (regression of e_t on e_{t-l}, no constant).  Pairs are formed within spells of consecutive dates.
    Returns {equation: {lag: (coef, t)}} and a flag that is True if any |t| > 1.96 (the VAR is dynamically
    misspecified or the residuals are serially correlated, which Assumption 5(i) rules out)."""
    sp, tm, un = model.clusters['spell'], model.clusters['time'], model.clusters['unit']
    out, flag, maxt = {}, False, 0.0
    for j, v in enumerate(model.endog):
        e = model.eta[j]
        out[v] = {}
        for l in lags:
            a, b, g = [], [], []
            same = (sp[l:] == sp[:-l]) & (tm[l:] - tm[:-l] == l)
            idx = np.where(same)[0]
            a, b, g = e[idx + l], e[idx], un[idx + l]
            if len(a) < 10:
                out[v][l] = (np.nan, np.nan); continue
            beta = (a @ b) / (b @ b)
            r = a - beta * b
            u, ic = np.unique(g, return_inverse=True)
            Gn = len(u)
            se = np.sqrt(Gn / (Gn - 1) * np.sum(np.bincount(ic, weights=b * r) ** 2)) / (b @ b)
            t = beta / se
            out[v][l] = (beta, t)
            maxt = max(maxt, abs(t))
            flag = flag or abs(t) > 1.96
    return out, flag, maxt


def _hansen_j(Y, X, Z):
    """Two-step efficient GMM Hansen J for a system of K equations with common regressors X and instruments Z, an
    observation-level weighting matrix (no cluster correction), and unrestricted coefficients per equation."""
    n, K = Y.shape
    q, m = Z.shape[1], X.shape[1]
    Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)
    E = Y - X @ np.linalg.solve(X.T @ Pz @ X, X.T @ Pz @ Y)
    G = np.stack([E[:, [k]] * Z for k in range(K)], axis=1).reshape(n, -1)
    Si = np.linalg.pinv(G.T @ G / n)
    a = np.concatenate([Z.T @ Y[:, k] / n for k in range(K)])
    M = np.zeros((K * q, K * m))
    for k in range(K):
        M[k * q:(k + 1) * q, k * m:(k + 1) * m] = Z.T @ X / n
    th = np.linalg.solve(M.T @ Si @ M, M.T @ Si @ a)
    g = a - M @ th
    return float(n * g @ Si @ g), K * (q - m), n


def _select_lags_gmm(data, unit, time, endog, pmax):
    """Andrews and Lu (2001) MMSC for the FOD-GMM estimator on a COMMON sample and a COMMON instrument set (a constant
    and the first pmax lags of the variables in levels, for every candidate p).  With J the Hansen statistic, df the
    number of overidentifying restrictions and n the number of observations:
        MBIC = J - df ln n,   MAIC = J - 2 df,   MQIC = J - 2 df ln ln n.
    A common instrument set removes the artefact of comparing models whose instrument sets differ.  Model p=pmax is just
    identified (J = df = 0).  n is the number of observations, as in the pvar package of Abrigo and Love (2016); on the
    10-region fiscal panel this reproduces the penalty terms of the paper's lag-selection table exactly.  Validation
    status: the criteria of the just-identified and under-identified rows reproduce that table; the J of the p=1 row
    does not (9.6 here against 4.1 in the old table), so treat the p=1 comparison with that caveat."""
    d = data.sort_values([unit, time]).reset_index(drop=True)
    sid = spell_ids(d, unit, time)
    K = len(endog)
    rows = []
    for p in range(1, pmax + 1):
        Ys, Xs, Zs = [], [], []
        for s_ in np.unique(sid):
            Yn = d.loc[sid == s_, endog].values
            T = len(Yn)
            if T <= pmax + 1:
                continue
            Yf = apply_fod(Yn[pmax:])
            X = apply_fod(np.hstack([Yn[pmax - l:T - l] for l in range(1, p + 1)]))
            Z = np.array([np.concatenate([Yn[pmax + t - l] for l in range(1, pmax + 1)]) for t in range(len(Yf))])
            ok = ~(np.isnan(Yf).any(1) | np.isnan(X).any(1) | np.isnan(Z).any(1))
            Ys.append(Yf[ok]); Xs.append(X[ok]); Zs.append(Z[ok])
        Y, X, Z = map(np.vstack, (Ys, Xs, Zs))
        X = np.hstack([np.ones((len(X), 1)), X]); Z = np.hstack([np.ones((len(Z), 1)), Z])
        J, df, n = _hansen_j(Y, X, Z) if Z.shape[1] > X.shape[1] else (0.0, K * (Z.shape[1] - X.shape[1]), len(Y))
        rows.append(dict(p=p, n=n, df=df, J=J, mbic=J - df * np.log(n), maic=J - 2 * df, mqic=J - 2 * df * np.log(np.log(n))))
    out = pd.DataFrame(rows).set_index('p')
    ch = {c: int(out[c].idxmin()) for c in ('mbic', 'maic', 'mqic')}
    ch['bic'] = ch['mbic']
    out.attrs['choice'] = ch
    return out
