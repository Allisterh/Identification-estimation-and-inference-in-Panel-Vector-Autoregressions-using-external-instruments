"""Panel VAR with an external instrument (PVAR-IV): FOD-GMM estimation of the VAR, IV identification of
the impact vector, impulse responses, first-stage statistics and Anderson-Rubin sets."""
import warnings
import numpy as np
import pandas as pd
from .utils import apply_fod, spell_ids, is_unit_demeaned
from scipy import stats
from .inference import cov_moments, ma_matrices, ar_sets, ar_ratio_sets, format_set, CRIT, crit_value
from .clr import multi_sets



def _psd(W):
    """Symmetrize and set negative eigenvalues to zero (two-way clustered covariances need not be PSD)."""
    W = (W + W.T) / 2
    val, vec = np.linalg.eigh(W)
    return (vec * np.clip(val, 0, None)) @ vec.T

class PVARIV:
    """
    data       : DataFrame (long format)
    unit, time : column names (time must be an integer year/period index)
    endog      : list of endogenous variables; the policy variable is endog[policy]
    instrument : column name of the external instrument, or a list of k column names.  With k == 1 everything is
        as before (closed-form AR sets).  With k > 1 the point estimate pools the instruments (see `fit`), the first
        stage is a joint Wald test, and weak-instrument-robust sets come from `robust_sets` (AR with chi2(k), Kleibergen's
        K and the conditional LR test of `pvariv.clr`).  ar_sets, ar_ratio_sets and plugin_intervals need k == 1.
    p          : VAR order;  instlags: number of lags in levels used as GMM instruments (default p)
    exog       : optional list of strictly exogenous controls (enter FOD-transformed, instrument themselves)
    cluster_cols : extra columns usable as cluster labels in inference (e.g. region-year)
    min_time   : keep only dependent observations dated >= min_time (lags may come from earlier dates)

    method='within' replaces FOD-GMM by the within transformation with OLS (consistent for large T).
    method='pooled' is OLS with a common constant and no unit effects (the 'no fixed effects' benchmark).

    demean_instrument : subtract the spell mean of the instrument (over the estimation sample) before it is used.
        Default None -> True for method='within', False for 'fod-gmm'.  With the within transformation the
        residuals sum to zero in each spell, so Gamma is unchanged, but the first stage F and every covariance
        built from z*eta need a zero-mean-per-unit instrument; with a raw instrument they are wrong (and
        time-clustered inference is badly off when z has large unit means).  For FOD residuals Gamma itself
        changes under demeaning, so the default keeps the instrument in levels as in the MATLAB package.
    allow_unit_demeaned : method='fod-gmm' needs the unit effects to be in the data (FOD removes them).  Data that
        were already demeaned by unit put the unit mean, which contains future shocks, into the lagged-level
        instruments.  By default this raises an error; pass True to override.

    pooling : how the k instruments are combined into one impact vector (point estimate only; AR/K/CLR sets do not
        depend on it).  'gamma' (default): b = Gamma a with a = gamma/|gamma|^2 (gamma = policy row of Gamma).  '2sls':
        a = S_zz^{-1} gamma / (gamma' S_zz^{-1} gamma), i.e. b_j = 2SLS of the j-th innovation on the policy innovation with
        the k instruments (the estimator of a single-equation IV paper).  Identical for k = 1.

    bias_correction : None or 'hpj' (method='within' only).  'hpj' replaces every coefficient of the within estimator
        by the half-panel jackknife 2 B - (B_1 + B_2)/2 of Dhaene and Jochmans (2015), B_1 and B_2 estimated on the
        first and second half of the dependent observations of each spell (lags carried across the split), which
        removes the O(1/T) bias of the within estimator in dynamic panels.
        Residuals, Gamma and every covariance are then computed from the corrected coefficients on the full sample;
        the jackknife leaves the asymptotic variance unchanged, so inference is otherwise as without correction.

    Units are split into spells of consecutive periods (gaps start a new spell); unit effects are removed
    by forward orthogonal deviations within each spell.  The instrument is used in levels at the date of
    the dependent observation, as in the MATLAB replication package.
    """

    def __init__(self, data, unit, time, endog, instrument, p=1, instlags=None, exog=None,
                 policy=0, cluster_cols=(), min_time=None, method='fod-gmm', demean_instrument=None,
                 allow_unit_demeaned=False, pooling='gamma', bias_correction=None):
        self.instruments = [instrument] if isinstance(instrument, str) else list(instrument)
        self.k = len(self.instruments)
        self.unit, self.time, self.endog, self.instrument = unit, time, list(endog), self.instruments[0]
        self.p, self.instlags = p, (instlags or p)
        self.exog, self.policy = list(exog or []), policy
        self.cluster_cols, self.min_time = list(cluster_cols), min_time
        assert method in ('fod-gmm', 'within', 'pooled')
        self.method = method
        self.demean_instrument = (method == 'within') if demean_instrument is None else bool(demean_instrument)
        self.allow_unit_demeaned = allow_unit_demeaned
        assert pooling in ('gamma', '2sls')
        self.pooling = pooling
        assert bias_correction in (None, 'hpj')
        if bias_correction == 'hpj' and method != 'within':
            raise ValueError("bias_correction='hpj' is implemented for method='within'")
        self.bias_correction = bias_correction
        self.data = data.sort_values([unit, time]).reset_index(drop=True)
        self.data['_spell'] = spell_ids(self.data, unit, time)
        self.K = len(self.endog)
        self._fitted = False

    # ------------------------------------------------------------------ estimation
    def fit(self):
        K, p, L = self.K, self.p, self.instlags
        if self.method == 'fod-gmm' and not self.allow_unit_demeaned and is_unit_demeaned(self.data, self.unit, self.endog):
            raise ValueError(
                "method='fod-gmm' received endogenous variables that are already demeaned by unit (every unit has "
                "mean ~0).  FOD removes the unit effects itself; demeaning first puts the unit mean, which contains "
                "future shocks, into the lagged-level GMM instruments and invalidates them.  Remove time effects "
                "with a year-only demeaning (utils.demean_by(df, cols, time)), use method='within' with "
                "twoway_demean, or pass allow_unit_demeaned=True if you really want this.")
        Yd, Xr, Zi, Ex, Zx, rows = [], [], [], [], [], {k: [] for k in ['unit', 'spell', 'time'] + self.cluster_cols}
        for s_, g in self.data.groupby('_spell'):
            Yn = g[self.endog].values
            T = len(Yn)
            if T <= p + 1:
                continue
            X = np.hstack([Yn[p - l:T - l] for l in range(1, p + 1)])
            E = g[self.exog].values[p:] if self.exog else np.zeros((T - p, 0))
            if self.method == 'fod-gmm':
                Ydep = apply_fod(Yn[p:])
                Xf = apply_fod(X)
                Ef = apply_fod(E) if self.exog else np.zeros((len(Ydep), 0))
                m = len(Ydep)
                Zlev = np.array([np.concatenate([Yn[p + t - l] if p + t - l >= 0 else np.full(K, np.nan)
                                                 for l in range(1, L + 1)]) for t in range(m)])
            elif self.method == 'pooled':            # no unit effects: levels, common constant
                Ydep = Yn[p:]
                Xf = X
                Ef = E
                m = len(Ydep)
                Zlev = np.zeros((m, 0))
            else:                                    # within transformation (unit means within the spell)
                # '_hpj_dep' (set by _half_panel_B) marks the dependent observations of one half; the lags of its
                # first observations come from the preceding periods, and the unit means are taken over the half
                sel = g['_hpj_dep'].values[p:].astype(bool) if '_hpj_dep' in g else np.ones(T - p, dtype=bool)
                if not sel.any():
                    continue
                Ydep = Yn[p:] - Yn[p:][sel].mean(0)
                Xf = X - X[sel].mean(0)
                Ef = E - E[sel].mean(0) if self.exog else np.zeros((T - p, 0))
                Ydep[~sel] = np.nan
                m = len(Ydep)
                Zlev = np.zeros((m, 0))
            zx = g[self.instruments].values[p:][:m]
            tm = g[self.time].values[p:][:m]
            ok = ~(np.isnan(Ydep).any(1) | np.isnan(Xf).any(1) | np.isnan(Zlev).any(1) | np.isnan(Ef).any(1) | np.isnan(zx).any(1))
            if self.min_time is not None:
                ok &= tm >= self.min_time
            zk = zx[ok]
            if self.demean_instrument and zk.size:
                zk = zk - zk.mean(0)
            Yd.append(Ydep[ok]); Xr.append(Xf[ok]); Zi.append(Zlev[ok]); Ex.append(Ef[ok]); Zx.append(zk)
            rows['unit'] += [g[self.unit].values[0]] * ok.sum()
            rows['spell'] += [s_] * ok.sum()
            rows['time'] += list(tm[ok])
            for c in self.cluster_cols:
                rows[c] += list(g[c].values[p:][:m][ok])
        Yd, Xr, Zi, Ex = map(np.vstack, (Yd, Xr, Zi, Ex))
        z = np.vstack(Zx)                          # n x k
        n = len(Yd)
        X1 = np.hstack([np.ones((n, 1)), Ex, Xr])
        Z1 = np.hstack([np.ones((n, 1)), Ex, Zi])
        if self.method == 'fod-gmm':
            Q, _ = np.linalg.qr(Z1)
            B = np.linalg.solve(X1.T @ Q @ Q.T @ X1, X1.T @ Q @ Q.T @ Yd)
        else:
            B = np.linalg.solve(X1.T @ X1, X1.T @ Yd)
        self.B_uncorrected = B
        if self.bias_correction == 'hpj':
            B1, B2 = self._half_panel_B()
            B = 2 * B - (B1 + B2) / 2
        self.B = B[-K * p:]                        # (K*p x K): rows lags, columns equations
        self.AL = self.B.T                         # [A_1 ... A_p]
        self.eta = (Yd - X1 @ B).T                 # K x T
        self.X, self.Zinst, self.Z, self.z = X1, Z1, z, z[:, 0]
        self.T = n
        self.clusters = {k: np.array(v) for k, v in rows.items()}
        self.GammaK = (self.eta @ z) / n           # K x k: column l = Cov(eta, z_l)
        gam = self.GammaK[self.policy]
        if self.pooling == '2sls' and self.k > 1:
            zc = z - z.mean(0)
            Sz = zc.T @ zc / n
            a = np.linalg.solve(Sz, gam)
            a = a / (gam @ a)                      # a'gam = 1, 2SLS weights
        else:
            a = gam / (gam @ gam)                  # weights with a'gam = 1; k = 1: b = Gamma / Gamma[policy]
        self.b = self.GammaK @ a                   # pooled impact vector, b[policy] = 1 (same weights as clr.multi_sets)
        self.Gamma = self.GammaK[:, 0] if self.k == 1 else self.GammaK
        self._fitted = True
        return self

    def _half_panel_B(self):
        """Coefficients of the same model estimated on the first and on the second half of the dependent observations
        of every spell (half-panel jackknife, Dhaene and Jochmans 2015).  The split is over the observations that
        enter the regression: the first observations of the second half take their lags from the end of the first
        half, so each half has half of the dependent observations and the O(1/T) bias of each half is twice that of
        the full sample.  Unit effects are removed within each half; time effects removed beforehand on the full
        sample are left as they are."""
        d = self.data.copy()
        pos = d.groupby('_spell').cumcount().values - self.p            # index among dependent observations
        size = d.groupby('_spell')[self.time].transform('size').values - self.p
        cut = (size + 1) // 2
        out = []
        for flag in ((pos >= 0) & (pos < cut), pos >= cut):
            sub = d.drop(columns='_spell').assign(_hpj_dep=flag)
            m = PVARIV(sub, self.unit, self.time, self.endog, self.instruments, p=self.p, instlags=self.instlags,
                       exog=self.exog, policy=self.policy, cluster_cols=self.cluster_cols, min_time=self.min_time,
                       method=self.method, demean_instrument=self.demean_instrument, pooling=self.pooling).fit()
            out.append(m.B_uncorrected)
        return out

    # ------------------------------------------------------------------ responses
    def irf(self, H=4):
        """K x (H+1) responses of all variables to a unit change in the policy variable."""
        return np.array([c @ self.b for c in ma_matrices(self.AL, self.K, self.p, H)]).T

    def cumulative_multiplier(self, H=4, outcome=1, integral=False):
        """sum_{s<=h} irf_outcome(s) / sum_{s<=h} irf_policy(s), h = 0..H (point estimate).
        integral=True: the ratio of the sums of the level responses over periods 0..h (cumulate once more)."""
        c = self.cumulative_irf(H)
        if integral:
            c = np.cumsum(c, axis=1)
        return c[outcome] / c[self.policy]

    def ar_ratio_sets(self, H=4, cluster=None, outcome=1, iv_consistent=None, level=0.95, integral=False):
        """AR sets for the cumulative multiplier by horizon (see inference.ar_ratio_sets)."""
        self._require_single('ar_ratio_sets')
        W = self.covariance(cluster, iv_consistent)
        return ar_ratio_sets(self.AL, self.Gamma, W, self.K, self.p, self.T, H, outcome, self.policy, crit_value(level), integral=integral)

    def cumulative_irf(self, H=4):
        return np.cumsum(self.irf(H), axis=1)

    # ------------------------------------------------------------------ inference
    def _cl(self, cluster):
        if cluster is None:
            return None
        return self.clusters[cluster] if isinstance(cluster, str) else np.asarray(cluster)

    def _iv_flag(self, iv_consistent):
        """None -> use the instruments of the VAR as the score whenever the VAR step is just identified by them
        (FOD-GMM with instlags == p).  This is the consistent sandwich; with the regressors as score (the MATLAB
        package's choice) 95% AR sets under-covered in simulation (about 87-88%)."""
        if iv_consistent is None:
            return self.method == 'fod-gmm' and self.instlags == self.p
        if iv_consistent and not (self.method == 'fod-gmm' and self.instlags == self.p):
            raise ValueError("iv_consistent=True needs method='fod-gmm' with instlags == p (just-identified VAR step)")
        return bool(iv_consistent)

    def covariance(self, cluster=None, iv_consistent=None):
        """Joint covariance of (vec A_hat, Gamma_hat), scaled by T. cluster: None, 'unit', 'time', 'spell',
        'twoway', a name in cluster_cols, or an array aligned with the estimation sample.
        'twoway' clusters by unit and by time (Cameron, Gelbach and Miller 2011): W_unit + W_time - W_het, where the
        intersection of a unit and a date is one observation; negative eigenvalues are set to zero.
        iv_consistent: None (default) = True for FOD-GMM with instlags == p, False for 'within';
        False reproduces the MATLAB package (regressors as score)."""
        iv_consistent = self._iv_flag(iv_consistent)
        Zi = self.Zinst if iv_consistent else None
        if self._is_twoway(cluster):
            Wu = cov_moments(self.p, self.X, self.Z, self.eta, self._cl('unit'), Zinst=Zi)
            Wt = cov_moments(self.p, self.X, self.Z, self.eta, self._cl('time'), Zinst=Zi)
            Wh = cov_moments(self.p, self.X, self.Z, self.eta, None, Zinst=Zi)
            return _psd(Wu + Wt - Wh)
        return cov_moments(self.p, self.X, self.Z, self.eta, self._cl(cluster), Zinst=Zi)

    @staticmethod
    def _is_twoway(cluster):
        return isinstance(cluster, str) and cluster == 'twoway'

    def _meat(self, sc, cluster):
        """Meat of a sandwich for the score matrix sc (n x k): heteroskedasticity-robust, one-way clustered with the
        G/(G-1) factor, or two-way clustered by unit and time (sum of the two one-way meats minus the robust one)."""
        sc = np.asarray(sc, dtype=float)
        if sc.ndim == 1:
            sc = sc[:, None]
        if self._is_twoway(cluster):
            return _psd(self._meat(sc, 'unit') + self._meat(sc, 'time') - self._meat(sc, None))
        cl = self._cl(cluster)
        if cl is None:
            return sc.T @ sc
        u, ic = np.unique(cl, return_inverse=True)
        G = len(u)
        B = np.column_stack([np.bincount(ic, weights=sc[:, j]) for j in range(sc.shape[1])])
        return G / (G - 1) * B.T @ B

    def first_stage(self, cluster=None, iv_consistent=None, level=0.95):
        """First stage of the policy innovation on the instrument.  Returns the coefficient, the robust F
        (heteroskedasticity-robust, or cluster-robust when `cluster` is given), the classical (homoskedastic) F,
        the effective F of Montiel Olea and Pflueger (2013) and the moment-based t^2 that governs the AR test
        (compare with the critical value at `level`).  With a single instrument the effective F equals the
        robust F, so F_eff is reported as that number."""
        if self.k > 1:
            return self._first_stage_multi(cluster, iv_consistent, level)
        z = self.z - self.z.mean()
        w = self.eta[self.policy] - self.eta[self.policy].mean()
        Szz = z @ z
        pi = (z @ w) / Szz
        r = w - pi * z
        F = pi ** 2 * Szz ** 2 / float(self._meat(z * r, cluster)[0, 0])
        Fc = pi ** 2 * Szz / (r @ r / (len(z) - 2))
        W = self.covariance(cluster, iv_consistent)
        t2 = self.T * self.Gamma[self.policy] ** 2 / W[self.K ** 2 * self.p:, self.K ** 2 * self.p:][self.policy, self.policy]
        return dict(coef=pi, F=F, F_classical=Fc, F_eff=F, t2_ar=t2, ar_informative=bool(t2 > crit_value(level)))

    @property
    def n_internal(self):
        """Number of internal (lagged-level) instruments per equation used in the VAR step (excl. constant)."""
        return self.K * self.instlags if self.method == 'fod-gmm' else 0

    @property
    def n_external(self):
        return self.k

    def _require_single(self, what):
        if self.k != 1:
            raise NotImplementedError(f'{what} needs a single instrument; with k = {self.k} use robust_sets() '
                                      f'(AR chi2(k), K, CLR).')

    def _first_stage_multi(self, cluster, iv_consistent, level):
        """Joint first stage of the policy innovation on the k instruments.  F = Wald/k with the (cluster-)robust
        covariance.  F_eff (Montiel Olea-Pflueger) is not computed for k > 1 (NaN).  t2_ar is the moment-based
        Wald statistic T gamma' V_gamma^{-1} gamma, compared with chi2(k)."""
        k = self.k
        Z = self.Z - self.Z.mean(0)
        w = self.eta[self.policy] - self.eta[self.policy].mean()
        S = Z.T @ Z
        Si = np.linalg.pinv(S)
        pi = Si @ (Z.T @ w)
        r = w - Z @ pi
        sc = Z * r[:, None]
        M = self._meat(sc, cluster)
        V = Si @ M @ Si
        F = float(pi @ np.linalg.pinv(V) @ pi / k)
        Fc = float((pi @ S @ pi / k) / (r @ r / (len(w) - k - 1)))
        W = self.covariance(cluster, iv_consistent)
        dA = self.K ** 2 * self.p
        idx = dA + np.arange(k) * self.K + self.policy
        gam = self.GammaK[self.policy]
        t2 = float(self.T * gam @ np.linalg.pinv(W[np.ix_(idx, idx)]) @ gam)
        return dict(coef=pi, F=F, F_classical=Fc, F_eff=np.nan, t2_ar=t2,
                    ar_informative=bool(t2 > stats.chi2.ppf(level, k)))

    def robust_sets(self, H=4, cluster=None, tests=('ar', 'clr', 'k'), level=0.95, outcomes=None, horizons=None,
                    iv_consistent=None, **kw):
        """Weak-instrument-robust sets for the responses theta_{j,h} (impact of a unit change in the policy variable),
        valid for any k >= 1: AR with chi2(k), Kleibergen's K and the conditional LR test, by test inversion on a grid
        (pvariv.clr.multi_sets).  Returns {(j, h): {'theta_hat', 'se', 'ar': [(lo, hi), ...], 'clr': ..., 'k': ...}};
        infinite endpoints mean the set reaches the edge of the grid (read as unbounded).  Not available for the
        cumulative multiplier.  Extra keyword arguments go to multi_sets (rank, ngrid, span)."""
        W = self.covariance(cluster, iv_consistent)
        return multi_sets(self.AL, self.GammaK, W, self.K, self.p, self.T, H, self.policy, level, tests,
                          outcomes, horizons, **kw)

    def robust_table(self, H=4, cluster=None, test='clr', level=0.95, outcome=None, digits=3, iv_consistent=None, **kw):
        """DataFrame of formatted robust sets (rows = horizons, columns = outcome variables)."""
        outs = [j for j in range(self.K) if j != self.policy] if outcome is None else [outcome]
        res = self.robust_sets(H, cluster, (test,), level, outs, None, iv_consistent, **kw)
        def fmt(iv):
            if not iv:
                return 'empty'
            return ' U '.join('[' + ', '.join(('-inf' if a == -np.inf else f'{a:.{digits}f}',
                                              'inf' if b == np.inf else f'{b:.{digits}f}'))
                              + ']' for a, b in iv)
        return pd.DataFrame({self.endog[j]: [fmt(res[(j, h)][test]) for h in range(H + 1)] for j in outs},
                            index=[f'h={h}' for h in range(H + 1)])

    def ar_sets(self, H=4, cluster=None, cumulative=False, iv_consistent=None, level=0.95):
        self._require_single('ar_sets')
        W = self.covariance(cluster, iv_consistent)
        return ar_sets(self.AL, self.Gamma, W, self.K, self.p, self.T, H, self.policy, cumulative, crit_value(level))

    def ar_table(self, H=4, cluster=None, cumulative=False, digits=3, iv_consistent=None, level=0.95):
        o, c = self.ar_sets(H, cluster, cumulative, iv_consistent, level)
        return pd.DataFrame({v: [format_set(o[j, h], c[j, h], digits) for h in range(H + 1)]
                             for j, v in enumerate(self.endog)}, index=[f'h={h}' for h in range(H + 1)])

    def plugin_intervals(self, H=4, cluster=None, outcome=1, iv_consistent=None, level=0.95, cumulative=False):
        """Standard (delta-method) intervals for the response of `outcome` at horizons 0..H (cumulative=True: the sum of
        the responses at horizons 0..h, e.g. the level response when the outcome enters in differences)."""
        self._require_single('plugin_intervals')
        K, p = self.K, self.p
        W = self.covariance(cluster, iv_consistent)
        x0 = np.r_[self.AL.reshape(-1, order='F'), self.Gamma]

        def f(x, h):
            A = x[:K * K * p].reshape((K, K * p), order='F')
            G = x[K * K * p:]
            C = ma_matrices(A, K, p, h)
            row = sum(C[k][outcome] for k in range(h + 1)) if cumulative else C[h][outcome]
            return (row @ G) / G[self.policy]
        rows = []
        for h in range(H + 1):
            g = np.zeros_like(x0)
            for j in range(len(x0)):
                d = np.zeros_like(x0); d[j] = 1e-6
                g[j] = (f(x0 + d, h) - f(x0 - d, h)) / 2e-6
            s = np.sqrt(g @ W @ g / self.T)
            est = f(x0, h)
            zc = crit_value(level) ** 0.5
            rows.append((est, s, est - zc * s, est + zc * s))
        return pd.DataFrame(rows, columns=['estimate', 'se', f'lo{round(level * 100)}', f'hi{round(level * 100)}'], index=[f'h={h}' for h in range(H + 1)])

    def summary(self, H=3, clusters=(None, 'unit')):
        print(f'PVAR-IV  n={self.T}  p={self.p}  K={self.K}  policy={self.endog[self.policy]}')
        print('impact vector b:', np.round(self.b, 4))
        print('IRF (rows = variables):\n', pd.DataFrame(self.irf(H), index=self.endog).round(4))
        for cl in clusters:
            fs = self.first_stage(cl)
            if self.k > 1:
                print(f'\ncluster={cl}: joint first stage F {fs["F"]:.1f} (k={self.k}), Wald {fs["t2_ar"]:.2f} vs chi2({self.k}) '
                      f'({"informative" if fs["ar_informative"] else "uninformative"} AR sets)')
                print(self.robust_table(H, cl))
                continue
            print(f'\ncluster={cl}: first stage coef {fs["coef"]:.4g}, F {fs["F"]:.1f}, AR t2 {fs["t2_ar"]:.2f} '
                  f'({"informative" if fs["ar_informative"] else "uninformative"} AR sets)')
            print(self.ar_table(H, cl))
