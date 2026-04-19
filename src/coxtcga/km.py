"""Kaplan-Meier estimator and log-rank test, implemented from first principles.

Kaplan-Meier
------------
Non-parametric MLE of the survival function under right-censoring:

    S_hat(t) = prod_{t_i <= t}  (1 - d_i / n_i)

where t_i indexes the distinct event times and

    n_i = # at risk just before t_i
    d_i = # events exactly at t_i

Greenwood's formula for the variance of log(S_hat):

    Var(log S_hat(t)) = sum_{t_i <= t}  d_i / ( n_i (n_i - d_i) )

giving a pointwise (log-log) confidence band. We return the log-CI since it
enforces S in (0, 1) for any sample size.

Log-rank test
-------------
Given two groups indexed by g in {0, 1} with pooled distinct event times,

    E_i^{(g)} = d_i * n_i^{(g)} / n_i          (expected events in group g)
    V_i       = d_i (n_i - d_i) n_i^{(0)} n_i^{(1)} / ( n_i^2 (n_i - 1) )

The test statistic is

    chi2 = ( sum_i (O_i^{(1)} - E_i^{(1)}) )^2 / sum_i V_i

which under H0: lambda_0(t) = lambda_1(t) is asymptotically chi-squared(1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import numpy as np


@dataclass
class KaplanMeier:
    times_: np.ndarray | None = field(default=None, init=False)
    survival_: np.ndarray | None = field(default=None, init=False)
    at_risk_: np.ndarray | None = field(default=None, init=False)
    events_: np.ndarray | None = field(default=None, init=False)
    var_log_s_: np.ndarray | None = field(default=None, init=False)

    def fit(self, durations: np.ndarray, events: np.ndarray) -> "KaplanMeier":
        t = np.asarray(durations, dtype=np.float64)
        d = np.asarray(events, dtype=np.float64)
        if t.shape != d.shape or t.ndim != 1:
            raise ValueError("durations and events must be 1D of equal length")
        if not np.all((d == 0) | (d == 1)):
            raise ValueError("events must be 0/1")

        order = np.argsort(t, kind="stable")
        t = t[order]
        d = d[order]

        unique_t, inv = np.unique(t, return_inverse=True)
        n_unique = len(unique_t)
        events_per = np.bincount(inv, weights=d).astype(np.float64)
        at_each = np.bincount(inv).astype(np.float64)
        # n_i = #subjects still at risk just before t_i = total - #left before.
        n = len(t)
        at_risk = n - np.concatenate(([0.0], np.cumsum(at_each)))[:-1]

        # Keep only times with at least one event for the step points of S.
        mask = events_per > 0
        t_e = unique_t[mask]
        d_e = events_per[mask]
        n_e = at_risk[mask]

        hazard = d_e / n_e
        survival = np.cumprod(1.0 - hazard)
        # Greenwood (log-scale).
        incr = d_e / (n_e * (n_e - d_e + 1e-300))
        var_log_s = np.cumsum(incr)

        self.times_ = t_e
        self.survival_ = survival
        self.at_risk_ = n_e
        self.events_ = d_e
        self.var_log_s_ = var_log_s
        return self

    def confidence_interval(self, alpha: float = 0.05):
        """Two-sided (1-alpha) pointwise CI for S(t) using the log transform."""
        if self.survival_ is None:
            raise RuntimeError("fit() before confidence_interval")
        from math import erf

        # inverse standard-normal via scipy-free Newton on erf (alpha small).
        z = _inv_std_normal(1 - alpha / 2)
        se_log_s = np.sqrt(self.var_log_s_)
        log_s = np.log(self.survival_ + 1e-300)
        lo = np.exp(log_s - z * se_log_s)
        hi = np.exp(log_s + z * se_log_s)
        return np.minimum(hi, 1.0), np.maximum(lo, 0.0)

    def median_survival(self) -> float | None:
        """Smallest t such that S(t) <= 0.5; None if survival never crosses."""
        if self.survival_ is None:
            raise RuntimeError("fit() before median_survival")
        below = np.where(self.survival_ <= 0.5)[0]
        if len(below) == 0:
            return None
        return float(self.times_[below[0]])


def logrank_test(
    durations: np.ndarray,
    events: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Two-sample log-rank test.

    ``groups`` must be 0/1 (or any two-valued array). Returns a dict with the
    chi-squared statistic, the observed-minus-expected for group 1, the
    variance, and the two-sided p-value.
    """
    t = np.asarray(durations, dtype=np.float64)
    d = np.asarray(events, dtype=np.float64)
    g = np.asarray(groups)

    # Map groups to 0/1 preserving sort order of unique values.
    uniq = np.unique(g)
    if len(uniq) != 2:
        raise ValueError(f"logrank_test expects exactly 2 groups; got {uniq}")
    g01 = (g == uniq[1]).astype(np.float64)

    order = np.argsort(t, kind="stable")
    t = t[order]
    d = d[order]
    g01 = g01[order]

    unique_t = np.unique(t)
    O1_minus_E1 = 0.0
    V = 0.0
    for ti in unique_t:
        mask_at = t >= ti
        n = mask_at.sum()
        n1 = g01[mask_at].sum()
        at_t = t == ti
        d_i = d[at_t].sum()
        d1 = (d[at_t] * g01[at_t]).sum()
        if n <= 1 or d_i == 0:
            continue
        e1 = d_i * n1 / n
        O1_minus_E1 += d1 - e1
        if n > 1:
            V += d_i * (n - d_i) * n1 * (n - n1) / (n * n * (n - 1))

    chi2 = (O1_minus_E1 ** 2) / V if V > 0 else 0.0
    # chi2(1) p-value = 2 * (1 - Phi(sqrt(chi2))).
    p = 2 * (1 - 0.5 * (1 + erf(sqrt(chi2) / sqrt(2))))
    return {
        "chi2": float(chi2),
        "df": 1,
        "p_value": float(p),
        "observed_minus_expected": float(O1_minus_E1),
        "variance": float(V),
    }


# ------------------------------------------------------ tiny stats helpers

def _inv_std_normal(p: float) -> float:
    """Beasley-Springer-Moro approximation to the normal quantile function."""
    # Good to ~1e-8 across the range of interest for CI construction.
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        import math
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        import math
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
