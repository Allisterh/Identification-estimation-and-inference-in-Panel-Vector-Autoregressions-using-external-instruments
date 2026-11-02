"""Helpers: forward orthogonal deviations, spells, demeaning, partialling out."""
import numpy as np
import pandas as pd


def apply_fod(M):
    """Forward orthogonal deviations of a (T x k) array; returns (T-1 x k)."""
    M = np.asarray(M, dtype=float)
    T = M.shape[0]
    out = np.full((T - 1, M.shape[1]), np.nan)
    for t in range(T - 1):
        n = T - t - 1
        out[t] = np.sqrt(n / (n + 1.0)) * (M[t] - M[t + 1:].mean(0))
    return out


def spell_ids(df, unit, time):
    """Integer id of each run of consecutive time periods within a unit (df must be sorted)."""
    ids = np.zeros(len(df), dtype=int)
    k = -1
    prev_u, prev_t = None, None
    for i, (u, t) in enumerate(zip(df[unit].values, df[time].values)):
        if u != prev_u or t - prev_t != 1:
            k += 1
        ids[i] = k
        prev_u, prev_t = u, t
    return ids


def demean_by(df, cols, by):
    """Subtract group means (one pass). `by` is a column name, list of names, or a Series."""
    out = df.copy()
    key = by if isinstance(by, pd.Series) else (df[by] if isinstance(by, str) else df[list(by)].astype(str).agg('_'.join, axis=1))
    for c in cols:
        out[c] = out[c] - out.groupby(key)[c].transform('mean')
    return out


def twoway_demean(df, cols, unit, cell, tol=1e-12, maxit=500):
    """Remove unit effects and `cell` effects (e.g. year, or region x year) by alternating projections.
    Missing values (NaN) are ignored when the group means are computed, column by column, and stay NaN."""
    X = df[cols].values.astype(float).copy()
    g1 = pd.factorize(df[unit])[0]
    key = df[cell] if isinstance(cell, str) else df[list(cell)].astype(str).agg('_'.join, axis=1)
    g2 = pd.factorize(key)[0]
    for _ in range(maxit):
        Xo = X.copy()
        for g in (g1, g2):
            M = ~np.isnan(X)                          # missing entries are ignored in the group means (column by column)
            n = np.zeros((g.max() + 1, X.shape[1]))
            s = np.zeros_like(n)
            np.add.at(n, g, M.astype(float))
            np.add.at(s, g, np.where(M, X, 0.0))
            X = X - (s / np.maximum(n, 1.0))[g]
        if np.abs(X - Xo).max() < tol:
            break
    out = df.copy()
    out[cols] = X
    return out


def partial_out(df, cols, controls):
    """Residualise `cols` on `controls` (OLS, no constant: demean first if needed)."""
    out = df.copy()
    C = df[controls].values.astype(float)
    V = df[cols].values.astype(float)
    out[cols] = V - C @ np.linalg.lstsq(C, V, rcond=None)[0]
    return out


def is_unit_demeaned(df, unit, cols, tol=1e-6):
    """True if every unit has (numerically) zero mean in every column of `cols`, i.e. the data were already
    demeaned by unit (one-way, or by twoway_demean).  Used to stop FOD-GMM from being run on such data."""
    g = df.groupby(unit)[list(cols)]
    means = g.mean().abs().values
    scale = df[list(cols)].std().values
    scale = np.where(scale > 0, scale, 1.0)
    return bool(np.nanmax(means / scale) < tol)
