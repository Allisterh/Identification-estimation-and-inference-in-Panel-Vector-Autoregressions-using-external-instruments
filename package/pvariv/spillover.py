"""PVAR-IV with spillovers across units through a known exposure mapping (Section S5 of the supplement).

With a known matrix of weights R = [r_ik] (r_ii = 0), the exposure of unit i to the policies of the other units is
S_it = sum_k r_ik W_kt, and its exposure to their instruments is Z^S_it = sum_k r_ik Z_kt.  The system is augmented to
x_it = (W_it, S_it, Y_1it, ..., Y_Jit)' and identified with the two instruments (Z_it, Z^S_it).  With Gamma the
m x 2 matrix of covariances between the innovations and the two instruments, the impact effects are

    [theta_j, phi_j] = Gamma[Y_j, :] Gamma[(W, S), :]^{-1},

the two-stage least squares coefficients of the outcome innovation on the innovations of the policy and of the
exposure (Proposition S3).  theta_j is the direct effect of a unit change in the policy of the unit, and phi_j the
spillover effect of a unit change in the exposure.  The impact matrix of the two shocks is B = [I_2; Theta], and the
responses at horizon h are C_h B, with C_h the moving average matrices of the estimated PVAR.

What is and is not claimed.  The causal interpretation requires the conditions of Proposition S3: constant direct and
spillover effects, a policy that responds only to the instrument of its own unit, and a nonsingular 2 x 2 block
Gamma[(W, S), :], which with a shift-share instrument requires that the exposure of a unit is not proportional to the
average exposure of the units connected to it.  The intervals are delta-method intervals, valid under strong
identification; identification() reports the determinant of the 2 x 2 block and its t-statistic as a diagnostic.

Usage:
    from pvariv.spillover import SpilloverPVARIV
    m = SpilloverPVARIV(df, 'county', 'quarter', policy='W', outcomes=['dY'], instrument='Z', R=R, p=8,
                        method='within', bias_correction='hpj').fit()
    m.effects('twoway')                       # impact direct and spillover effects with standard errors
    m.intervals(H=16, cluster='twoway', cumulative=True)
    m.identification('twoway')

R can be a NumPy array ordered as the sorted unit identifiers (or as `units`), a pandas DataFrame indexed and labelled
by unit identifiers, or a dict {unit: {neighbour: weight}}.  The diagonal is ignored.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .core import PVARIV
from .inference import ma_matrices


def _weight_matrix(R, units):
    """Dense N x N weight matrix aligned with the list `units`, with zero diagonal."""
    units = list(units)
    pos = {u: i for i, u in enumerate(units)}
    N = len(units)
    if isinstance(R, pd.DataFrame):
        M = R.reindex(index=units, columns=units).fillna(0.0).to_numpy(dtype=float)
    elif isinstance(R, dict):
        M = np.zeros((N, N))
        for i, row in R.items():
            if i not in pos:
                continue
            for k, w in row.items():
                if k in pos:
                    M[pos[i], pos[k]] = float(w)
    else:
        M = np.asarray(R, dtype=float)
        if M.shape != (N, N):
            raise ValueError(f'R has shape {M.shape}, expected {(N, N)} (one row and column per unit, in the order of '
                             f'`units` or of the sorted unit identifiers)')
    M = M.copy()
    np.fill_diagonal(M, 0.0)
    return M


def exposure(data, unit, time, cols, R, units=None, renormalize=True, prefix='R_'):
    """Add the exposures sum_k r_ik x_kt of each column in `cols` as new columns prefix + col.

    renormalize: when some connected units have a missing value in a period, rescale the weights of the available
    ones so that they keep their row sum; a unit with no available connected unit gets a missing exposure.
    """
    cols = [cols] if isinstance(cols, str) else list(cols)
    units = sorted(data[unit].unique()) if units is None else list(units)
    M = _weight_matrix(R, units)
    rowsum = M.sum(1)
    out = data.copy()
    for col in cols:
        wide = data.pivot_table(index=time, columns=unit, values=col, aggfunc='first').reindex(columns=units)
        X = wide.to_numpy(dtype=float)                       # periods x units
        avail = np.isfinite(X)
        num = np.where(avail, X, 0.0) @ M.T                  # [t, i] = sum_k r_ik x_kt over available k
        if renormalize:
            den = avail.astype(float) @ M.T
            with np.errstate(invalid='ignore', divide='ignore'):
                val = np.where(den > 0, num / den * rowsum[None, :], np.nan)
        else:
            missing = (~avail).astype(float) @ (M > 0).T.astype(float)   # connected units with a missing value
            val = np.where(missing == 0, num, np.nan)
        long = pd.DataFrame(val, index=wide.index, columns=units).reset_index() \
            .melt(id_vars=time, var_name=unit, value_name=prefix + col)
        out = out.merge(long, on=[unit, time], how='left')
    return out


class SpilloverPVARIV:
    """PVAR-IV augmented with the exposure to the policies of the other units, identified with the instrument and
    the exposure to the instruments of the other units.  Keyword arguments are passed to PVARIV (method,
    bias_correction, exog, cluster_cols, ...).  The exposures are computed from the columns as given: compute them
    before removing period effects (as here) or after; with row-normalized weights and a balanced panel the two
    orders give the same estimates, because the unit and period components of the exposures are absorbed by the
    unit and period effects."""

    def __init__(self, data, unit, time, policy, outcomes, instrument, R, p=1, units=None, renormalize=True,
                 prefix='R_', **kw):
        self.unit, self.time, self.policy_name = unit, time, policy
        self.outcomes = [outcomes] if isinstance(outcomes, str) else list(outcomes)
        self.s_name, self.zs_name = prefix + policy, prefix + instrument
        df = exposure(data, unit, time, [policy, instrument], R, units=units, renormalize=renormalize, prefix=prefix)
        self.endog = [policy, self.s_name] + self.outcomes
        self.model = PVARIV(df, unit, time, self.endog, [instrument, self.zs_name], p=p, policy=0, **kw)
        self.p = p

    # -------------------------------------------------------------------------------------------- estimation
    def fit(self):
        self.model.fit()
        self.n = len(self.endog)
        self.theta = self._pack(self.model.AL, self.model.GammaK)
        self.B = self._impact(self.model.GammaK)
        return self

    @staticmethod
    def _impact(Gam):
        GP = Gam[:2, :]                                      # rows of the policy and of its exposure
        Theta = Gam[2:, :] @ np.linalg.inv(GP)
        return np.vstack([np.eye(2), Theta])                 # m x 2: columns = direct shock, spillover shock

    def _pack(self, AL, Gam):
        return np.concatenate([np.asarray(AL).reshape(-1, order='F'), np.asarray(Gam).reshape(-1, order='F')])

    def _unpack(self, th):
        n, p = self.n, self.p
        na = n * n * p
        return th[:na].reshape((n, n * p), order='F'), th[na:].reshape((n, 2), order='F')

    def _irf_from(self, th, H, cumulative):
        AL, Gam = self._unpack(th)
        C = ma_matrices(AL, self.n, self.p, H)
        B = self._impact(Gam)
        R = np.stack([C[h] @ B for h in range(H + 1)])       # (H+1) x m x 2
        return np.cumsum(R, axis=0) if cumulative else R

    def irf(self, H=4, cumulative=False):
        """Responses of every variable to a unit direct shock (policy of the unit) and to a unit spillover shock
        (exposure), as a long DataFrame."""
        R = self._irf_from(self.theta, H, cumulative)
        rows = []
        for h in range(H + 1):
            for s, shock in enumerate(('direct', 'spillover')):
                for v, var in enumerate(self.endog):
                    rows.append(dict(h=h, shock=shock, variable=var, est=R[h, v, s]))
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------------------------- inference
    def _var_theta(self, cluster):
        return self.model.covariance(cluster) / self.model.T

    def _jacobian(self, f, th, eps=1e-6):
        f0 = np.atleast_1d(f(th))
        J = np.zeros((f0.size, th.size))
        for j in range(th.size):
            d = eps * max(1.0, abs(th[j]))
            tp, tm = th.copy(), th.copy()
            tp[j] += d; tm[j] -= d
            J[:, j] = (np.atleast_1d(f(tp)) - np.atleast_1d(f(tm))) / (2 * d)
        return f0, J

    def intervals(self, H=4, cluster='twoway', level=0.95, cumulative=False):
        """Delta-method intervals for all responses to the direct and the spillover shock (strong identification)."""
        V = self._var_theta(cluster)
        f = lambda th: self._irf_from(th, H, cumulative).reshape(-1)
        est, J = self._jacobian(f, self.theta)
        se = np.sqrt(np.clip(np.einsum('ij,jk,ik->i', J, V, J), 0, None))
        q = stats.norm.ppf(0.5 + level / 2)
        out = self.irf(H, cumulative)
        # irf() loops h, shock, variable; the flattened array is h, variable, shock
        idx = [(h * self.n + v) * 2 + s for h in range(H + 1) for s in range(2) for v in range(self.n)]
        out['se'] = se[idx]
        out['lo'], out['hi'] = out.est - q * out.se, out.est + q * out.se
        return out

    def effects(self, cluster='twoway', level=0.95):
        """Impact direct effects theta_j and spillover effects phi_j of the outcomes, with delta-method intervals."""
        t = self.intervals(0, cluster, level)
        return t[t.variable.isin(self.outcomes)].reset_index(drop=True)

    def identification(self, cluster='twoway'):
        """Diagnostic for the rank condition of Proposition S3: the 2 x 2 block of Gamma for the policy and its
        exposure, its determinant, the delta-method standard error of the determinant and the t-statistic."""
        V = self._var_theta(cluster)

        def det(th):
            _, Gam = self._unpack(th)
            return np.linalg.det(Gam[:2, :])

        d, J = self._jacobian(det, self.theta)
        se = float(np.sqrt(max(float((J @ V @ J.T)[0, 0]), 0.0)))
        GP = self.model.GammaK[:2, :]
        return dict(block=pd.DataFrame(GP, index=[self.policy_name, self.s_name],
                                       columns=['instrument', 'exposure to instruments']),
                    det=float(d[0]), se=se, t=float(d[0] / se) if se > 0 else np.inf,
                    condition_number=float(np.linalg.cond(GP)))
