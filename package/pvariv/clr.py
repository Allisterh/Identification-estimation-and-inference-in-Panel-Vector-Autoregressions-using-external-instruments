"""Weak-instrument-robust sets for PVAR-IV with k >= 1 external instruments: AR(chi2(k)), Kleibergen's LM (K) and a
conditional likelihood ratio (CLR) test, all inverted over a grid of null values theta0 for the response
theta_{j,h} = e_j' C_h(A) b,  b = Gamma[:, l] / Gamma[policy, l]  (the same for every instrument l under the model).

Setup.  Gamma_hat is n x k (column l = Cov(eta, z_l)).  For a null theta0 the k moments
        g_l(theta0) = e_j' C_h(A) Gamma_l - theta0 * Gamma[policy, l]            (l = 1..k)
are zero at the truth, linear in Gamma and smooth in A.  With W the joint covariance of sqrt(T)(vec A_hat, vec Gamma_hat)
(pvariv.inference.cov_moments, which already accepts k columns, stacked as [vec A, vec Gamma], Gamma column-major),
the delta method gives Sigma(theta0), the covariance of sqrt(T) g_hat, and the covariance of g_hat with
gamma_hat = Gamma_hat[policy, :], which is the (negative) derivative of g_hat in theta0.

Statistics (T-scaled):
    AR  = T g' Sigma^{-1} g                                  ~ chi2(k) under H0
    D~  = -gamma_hat + Cov(gamma_hat, g) Sigma^{-1} g        (gamma_hat purged of g: asymptotically independent of g)
    K   = T (D~' Sigma^{-1} g)^2 / (D~' Sigma^{-1} D~)       ~ chi2(1) under H0, conditionally on D~
    r   = T D~' Omega^{-1} D~,  Omega = Var(gamma_hat | g)   (rank statistic; Kleibergen / Andrews-Mikusheva)
    CLR = 0.5 [AR - r + sqrt((AR + r)^2 - 4 r (AR - K))]
Conditionally on D~, g ~ N(0, Sigma/T), so AR = z1^2 + chi2(k-1), K = z1^2, and the conditional critical value of CLR
depends on (r, k) only.  It is computed by simulation (common random numbers) and interpolated in log r.

What is and is not claimed.  Size: the test is valid in the limit experiment where (g_hat, gamma_hat) is jointly
normal with covariance Sigma, Omega built from the delta-method W, whatever the structure of W (heteroskedasticity,
clustering).  With Omega = c*Sigma (the homoskedastic IV structure) and the default rank='omega' the statistic coincides with Moreira's (2003)
CLR (r = Q_T, K = LM).  The option rank='iid' uses Sigma^{-1} instead of Omega^{-1} and equals Q_T only if c = 1; do not use it
for comparisons with Moreira's tables.  With general W the test is the Kleibergen / Andrews-Mikusheva-type conditional test;
I make no optimality claim (that is what Moreira and Moreira (2019) study).  For k = 1 the CLR set equals the AR set.
"""
import numpy as np
from scipy import stats
from .inference import ma_matrices, _g_matrices

_CACHE = {}


def _clr_crit_table(k, level, B=200000, seed=12345, ngrid=70):
    key = (k, round(level, 6), B, seed)
    if key in _CACHE:
        return _CACHE[key]
    rng = np.random.default_rng(seed)
    z1sq = rng.standard_normal(B) ** 2
    rest = rng.chisquare(k - 1, B) if k > 1 else np.zeros(B)
    AR, K = z1sq + rest, z1sq
    lr = np.log(np.logspace(-4, 5, ngrid))
    cs = []
    for r in np.exp(lr):
        LR = 0.5 * (AR - r + np.sqrt((AR + r) ** 2 - 4 * r * (AR - K)))
        cs.append(np.quantile(LR, level))
    out = (lr, np.array(cs))
    _CACHE[key] = out
    return out


def clr_crit(r, k, level=0.95, **kw):
    """Conditional critical value of the CLR statistic given the rank statistic r (array ok)."""
    r = np.asarray(r, float)
    if k == 1:
        return np.full(r.shape, stats.chi2.ppf(level, 1))
    lr, cs = _clr_crit_table(k, level, **kw)
    return np.interp(np.log(np.clip(r, 1e-4, 1e5)), lr, cs)


def _pieces(AL, Gam, W, n, p, H, policy, j, h, G, C, cumulative=False):
    k = Gam.shape[1]
    dA = n * n * p
    d = dA + n * k
    ej = np.eye(n)[j]
    Jf = np.zeros((k, d))
    Jg = np.zeros((k, d))
    if cumulative:                      # level response of an outcome in differences: sum of responses 0..h
        Ch, Gh = sum(C[s] for s in range(h + 1)), sum(G[s] for s in range(h + 1))
    else:
        Ch, Gh = C[h], G[h]
    f = np.zeros(k)
    for l in range(k):
        Jf[l, :dA] = (np.kron(Gam[:, l].reshape(1, -1), ej.reshape(1, -1)) @ Gh)[0]
        Jf[l, dA + l * n: dA + (l + 1) * n] = ej @ Ch
        Jg[l, dA + l * n + policy] = 1.0
        f[l] = ej @ Ch @ Gam[:, l]
    gam = Gam[policy, :].copy()
    Sff, Sfg, Sgg = Jf @ W @ Jf.T, Jf @ W @ Jg.T, Jg @ W @ Jg.T
    return f, gam, Sff, Sfg, Sgg


def _stats_on_grid(theta, f, gam, Sff, Sfg, Sgg, T, rank='omega'):
    """AR, K, r on a vector of null values (all vectorised)."""
    k = len(f)
    th = theta[:, None]
    g = f[None, :] - th * gam[None, :]                                           # (M,k)
    th3 = theta[:, None, None]
    Sig = Sff[None] - th3 * (Sfg + Sfg.T)[None] + th3 ** 2 * Sgg[None]            # (M,k,k)
    Cgm = Sfg.T[None] - th3 * Sgg[None]                                          # Cov(gamma_hat, g) = Jg W (Jf - th Jg)'
    Si_g = np.linalg.solve(Sig, g[..., None])[..., 0]
    AR = T * np.einsum('mk,mk->m', g, Si_g)
    Dt = -gam[None, :] + np.einsum('mkl,ml->mk', Cgm, Si_g)
    Si_D = np.linalg.solve(Sig, Dt[..., None])[..., 0]
    DSD = np.einsum('mk,mk->m', Dt, Si_D)
    K = T * np.einsum('mk,mk->m', Dt, Si_g) ** 2 / np.maximum(DSD, 1e-300)
    if rank == 'iid':
        r = T * DSD
    else:
        Si_C = np.linalg.solve(Sig, np.swapaxes(Cgm, 1, 2))                      # Sigma^{-1} Cov(g, gamma)
        Om = Sgg[None] - np.einsum('mkl,mlj->mkj', Cgm, Si_C)
        Om = 0.5 * (Om + np.swapaxes(Om, 1, 2))
        r = np.empty(len(theta))
        for i in range(len(theta)):
            w, V = np.linalg.eigh(Om[i])
            keep = w > 1e-10 * max(w.max(), 1e-300)
            x = V[:, keep].T @ Dt[i]
            r[i] = T * np.sum(x ** 2 / w[keep])
    return AR, K, r


def _intervals(theta, accept):
    out, i, M = [], 0, len(theta)
    while i < M:
        if accept[i]:
            j = i
            while j + 1 < M and accept[j + 1]:
                j += 1
            out.append((-np.inf if i == 0 else theta[i], np.inf if j == M - 1 else theta[j]))
            i = j + 1
        else:
            i += 1
    return out


def multi_sets(AL, Gam, W, n, p, T, H, policy=0, level=0.95, tests=('ar', 'clr', 'k'), outcomes=None,
               horizons=None, rank='omega', ngrid=4001, span=80.0, cumulative=False):
    """Confidence sets for theta_{j,h} for every requested (outcome j, horizon h), by test inversion on a grid.

    Returns {(j, h): {'theta_hat', 'se', test: [(lo, hi), ...], 'nonempty': ...}} with infinite endpoints marking
    sets that reach the edge of the grid (read as unbounded).  `span` is the half-width of the grid in plug-in
    standard errors (sinh-spaced, so it is dense near the centre).  Gam is n x k; AR uses chi2(k)."""
    k = Gam.shape[1]
    C = ma_matrices(AL, n, p, H)
    G = _g_matrices(AL, n, p, H)
    outcomes = [j for j in range(n) if j != policy] if outcomes is None else outcomes
    horizons = range(H + 1) if horizons is None else horizons
    res = {}
    for j in outcomes:
        for h in horizons:
            f, gam, Sff, Sfg, Sgg = _pieces(AL, Gam, W, n, p, H, policy, j, h, G, C, cumulative)
            a = gam / (gam @ gam)
            th0 = float(a @ f)
            Sig0 = Sff - th0 * (Sfg + Sfg.T) + th0 ** 2 * Sgg
            se = float(np.sqrt(max(a @ Sig0 @ a / T, 1e-300)))
            u = np.sinh(np.linspace(-np.arcsinh(span), np.arcsinh(span), ngrid))
            theta = th0 + se * u
            AR, K, r = _stats_on_grid(theta, f, gam, Sff, Sfg, Sgg, T, rank)
            ent = {'theta_hat': th0, 'se': se}
            if 'ar' in tests:
                ent['ar'] = _intervals(theta, AR <= stats.chi2.ppf(level, k))
            if 'k' in tests:
                ent['k'] = _intervals(theta, K <= stats.chi2.ppf(level, 1))
            if 'clr' in tests:
                LR = 0.5 * (AR - r + np.sqrt((AR + r) ** 2 - 4 * r * np.maximum(AR - K, 0)))
                ent['clr'] = _intervals(theta, LR <= clr_crit(r, k, level))
            res[(j, h)] = ent
    return res


def reject_at(theta0, AL, Gam, W, n, p, T, policy, j, h, level=0.95, rank='omega', cumulative=False):
    """Rejection indicators of AR(chi2(k)), K and CLR at given null value(s) theta0 for theta_{j,h}, or for the
    cumulated response sum_{s<=h} theta_{j,s} when cumulative=True (for simulations)."""
    k = Gam.shape[1]
    C = ma_matrices(AL, n, p, h)
    G = _g_matrices(AL, n, p, h)
    f, gam, Sff, Sfg, Sgg = _pieces(AL, Gam, W, n, p, h, policy, j, h, G, C, cumulative)
    th = np.atleast_1d(np.asarray(theta0, float))
    AR, K, r = _stats_on_grid(th, f, gam, Sff, Sfg, Sgg, T, rank)
    LR = 0.5 * (AR - r + np.sqrt((AR + r) ** 2 - 4 * r * np.maximum(AR - K, 0)))
    return {'ar': AR > stats.chi2.ppf(level, k), 'k': K > stats.chi2.ppf(level, 1), 'clr': LR > clr_crit(r, k, level),
            'AR': AR, 'K': K, 'r': r, 'LR': LR}
