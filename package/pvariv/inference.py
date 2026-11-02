"""Anderson-Rubin (Montiel Olea, Stock, Watson) sets and delta-method intervals for panel SVAR-IV.

Ported from the MATLAB replication package (+pvar/compute_msw_ci.m, cov_ahat_gamma.m, g_matrices.m).
"""
import numpy as np
from statistics import NormalDist

CRIT = 3.841458820694124     # chi2(1) 95% critical value


def crit_value(level=0.95):
    """chi2(1) critical value at the given confidence level (0.95 -> 3.8415, 0.90 -> 2.7055)."""
    if level == 0.95:
        return CRIT
    return NormalDist().inv_cdf((1 + level) / 2) ** 2


def vech_selector(n):
    I = np.eye(n)
    V = np.kron(I[[0], :], I)
    for i in range(1, n):
        V = np.vstack([V, np.kron(I[[i], :], I[i:, :])])
    return V


def cov_moments(p, Xs, Zext, eta, cl=None, Zinst=None):
    """Joint asymptotic covariance of (vec A_hat, Gamma_hat) * T.

    Xs: regressors (const, exog, lags) used in the VAR; Zext: external instrument (T x k);
    eta: residuals (n x T); cl: cluster labels for the moment contributions or None;
    Zinst: if given, the IV-consistent score (instruments for the VAR) is used instead of Xs.
    """
    n = eta.shape[0]
    k = Zext.shape[1]
    m = Xs.shape[1] - n * p
    T = eta.shape[1]
    iv = Zinst is not None
    score = Zinst if iv else Xs
    mat = np.hstack([score, eta.T, Zext])
    aux = (mat[:, :, None] * eta.T[:, None, :]).reshape(T, -1)   # row t = kron(mat[t], eta[:, t]), vectorized
    aux = aux - aux.mean(0)
    if cl is None:
        W = aux.T @ aux / T
    else:
        u, ic = np.unique(cl, return_inverse=True)
        S = np.zeros((len(u), aux.shape[1]))
        for j in range(aux.shape[1]):
            S[:, j] = np.bincount(ic, weights=aux[:, j])
        W = S.T @ S / T
    Q1 = (Zinst.T @ Xs / T) if iv else (Xs.T @ Xs / T)
    Q2 = Zext.T @ Xs / T
    V = vech_selector(n)
    Q1i = np.linalg.inv(Q1)
    sel = np.hstack([np.zeros((n * p, m)), np.eye(n * p)])
    r1 = np.hstack([np.kron(sel @ Q1i, np.eye(n)), np.zeros((n * n * p, n * n + k * n))])
    r2 = np.hstack([np.zeros((n * (n + 1) // 2, n * n * p + n * m)), V, np.zeros((n * (n + 1) // 2, k * n))])
    r3 = np.hstack([-np.kron(Q2 @ Q1i, np.eye(n)), np.zeros((k * n, n * n)), np.eye(k * n)])
    Sh = np.vstack([r1, r2, r3])
    Wf = Sh @ W @ Sh.T
    ii = np.r_[np.arange(n * n * p), np.arange(n * n * p + n * (n + 1) // 2, Wf.shape[0])]
    return Wf[np.ix_(ii, ii)]


def ma_matrices(AL, n, p, H):
    """MA coefficient matrices C_0..C_H of the VAR with stacked lag matrix AL = [A_1 ... A_p]."""
    Ac = np.zeros((n * p, n * p))
    Ac[:n, :] = AL
    if p > 1:
        Ac[n:, :n * (p - 1)] = np.eye(n * (p - 1))
    C, M = [], np.eye(n * p)
    for _ in range(H + 1):
        C.append(M[:n, :n].copy())
        M = Ac @ M
    return C


def _g_matrices(AL, n, p, H):
    G = []
    for h in range(H + 1):
        Gh = np.zeros((n * n, n * n * p))
        for j in range(n * n * p):
            d = np.zeros(n * n * p)
            d[j] = 1e-6
            Ap = AL + d.reshape((n, n * p), order='F')
            Am = AL - d.reshape((n, n * p), order='F')
            Gh[:, j] = (ma_matrices(Ap, n, p, H)[h] - ma_matrices(Am, n, p, H)[h]).reshape(-1, order='F') / 2e-6
        G.append(Gh)
    return G


def ar_sets(AL, Gam, W, n, p, T, H, policy=0, cumulative=False, crit=CRIT):
    """AR sets (default 95%; pass crit=crit_value(level)) for every variable j and horizon h (responses to a unit shock in the policy variable).

    Returns (sets, case): sets[j, h] = (lo, hi); case = 1 interval [lo, hi]; 2 complement of (lo, hi)
    i.e. (-inf, lo] U [hi, inf); 3 empty; 4 whole line.
    """
    W1, W12, W2 = W[:n * n * p, :n * n * p], W[:n * n * p, n * n * p:], W[n * n * p:, n * n * p:]
    C = ma_matrices(AL, n, p, H)
    G = _g_matrices(AL, n, p, H)
    if cumulative:
        C = list(np.cumsum(np.array(C), axis=0))
        G = list(np.cumsum(np.array(G), axis=0))
    e = np.eye(n)
    out = np.zeros((n, H + 1, 2))
    case = np.zeros((n, H + 1), int)
    B = Gam
    for j in range(n):
        for ih in range(H + 1):
            Ch, Gh, ej = C[ih], G[ih], e[:, j]
            kr = np.kron(B.reshape(1, -1), ej.reshape(1, -1))
            a = T * B[policy] ** 2 - crit * W2[policy, policy]
            b = (-2 * T * (ej @ Ch @ B) * B[policy] + 2 * crit * (kr @ Gh @ W12[:, policy])[0]
                 + 2 * crit * (ej @ Ch @ W2[:, policy]))
            c = (T * (ej @ Ch @ B) ** 2 - crit * (kr @ Gh @ W1 @ (kr @ Gh).T)[0, 0]
                 - 2 * crit * (kr @ Gh @ W12 @ Ch.T @ ej)[0] - crit * (ej @ Ch @ W2 @ Ch.T @ ej))
            Dl = b * b - 4 * a * c
            if a > 0 and Dl > 0:
                out[j, ih] = [(-b - np.sqrt(Dl)) / (2 * a), (-b + np.sqrt(Dl)) / (2 * a)]; case[j, ih] = 1
            elif a < 0 and Dl > 0:
                out[j, ih] = [(-b + np.sqrt(Dl)) / (2 * a), (-b - np.sqrt(Dl)) / (2 * a)]; case[j, ih] = 2
            elif a > 0 and Dl < 0:
                out[j, ih] = [np.nan, np.nan]; case[j, ih] = 3
            else:
                out[j, ih] = [-np.inf, np.inf]; case[j, ih] = 4
    return out, case


def ar_ratio_sets(AL, Gam, W, n, p, T, H, num, den, crit=CRIT, integral=False):
    """AR sets for the cumulative multiplier m_h = sum_{s<=h} irf_num(s) / sum_{s<=h} irf_den(s), e.g. the cumulative
    output response per unit of cumulative spending (Ramey-Zubairy).  The null m_h = m0 is linear in (A, Gamma),
        l_h - m0 r_h = 0,   l_h = e_num' (sum_s C_s) Gamma,   r_h = e_den' (sum_s C_s) Gamma,
    so the AR set solves one quadratic in m0 per horizon.  At h=0 it equals ar_sets for variable `num`.
    integral=True cumulates once more: m_h = sum_{s<=h} (sum_{k<=s} irf_num(k)) / sum_{s<=h} (sum_{k<=s} irf_den(k)), the ratio of
    the SUMS OF LEVEL RESPONSES over periods 0..h (an 'integral' multiplier as in Ramey-Zubairy, and the cumulated-level
    multiplier used for h+1 periods).  Default False = ratio of level responses at h (unchanged).
    Returns (sets, case) of shape (H+1, 2) and (H+1,), with the case codes of ar_sets."""
    C = np.cumsum(np.array(ma_matrices(AL, n, p, H)), axis=0)
    G = np.cumsum(np.array(_g_matrices(AL, n, p, H)), axis=0)
    if integral:
        C, G = np.cumsum(C, axis=0), np.cumsum(G, axis=0)
    C, G = list(C), list(G)
    e = np.eye(n)
    out = np.zeros((H + 1, 2))
    case = np.zeros(H + 1, int)
    for ih in range(H + 1):
        Ch, Gh = C[ih], G[ih]
        vals, grads = [], []
        for j in (num, den):
            kr = np.kron(Gam.reshape(1, -1), e[:, j].reshape(1, -1))
            vals.append(e[:, j] @ Ch @ Gam)
            grads.append(np.concatenate([(kr @ Gh)[0], Ch.T @ e[:, j]]))
        L, R = vals
        gl, gr = grads
        a = T * R ** 2 - crit * gr @ W @ gr
        b = -2 * T * L * R + 2 * crit * gl @ W @ gr
        c = T * L ** 2 - crit * gl @ W @ gl
        Dl = b * b - 4 * a * c
        if a > 0 and Dl > 0:
            out[ih] = [(-b - np.sqrt(Dl)) / (2 * a), (-b + np.sqrt(Dl)) / (2 * a)]; case[ih] = 1
        elif a < 0 and Dl > 0:
            out[ih] = [(-b + np.sqrt(Dl)) / (2 * a), (-b - np.sqrt(Dl)) / (2 * a)]; case[ih] = 2
        elif a > 0 and Dl < 0:
            out[ih] = [np.nan, np.nan]; case[ih] = 3
        else:
            out[ih] = [-np.inf, np.inf]; case[ih] = 4
    return out, case


def format_set(lo_hi, case, digits=3):
    if case == 4:
        return 'unbounded'
    if case == 3:
        return 'empty'
    lo, hi = lo_hi
    f = f'{{:.{digits}f}}'
    if case == 1:
        return f'[{f.format(lo)}, {f.format(hi)}]'
    return f'(-inf, {f.format(lo)}] U [{f.format(hi)}, inf)'
