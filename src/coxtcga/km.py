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

    def fit(self, durations: np.ndarray, events: np.ndarray) -> 'KaplanMeier':
        t = np.asarray(durations, dtype=np.float64)
        d = np.asarray(events, dtype=np.float64)
        if t.shape != d.shape or t.ndim != 1:
            raise ValueError('durations and events must be 1D of equal length')
        if not np.all((d == 0) | (d == 1)):
            raise ValueError('events must be 0/1')
        order = np.argsort(t, kind='stable')
        t = t[order]
        d = d[order]
        unique_t, inv = np.unique(t, return_inverse=True)
        n_unique = len(unique_t)
        events_per = np.bincount(inv, weights=d).astype(np.float64)
        at_each = np.bincount(inv).astype(np.float64)
        n = len(t)
        at_risk = n - np.concatenate(([0.0], np.cumsum(at_each)))[:-1]
        mask = events_per > 0
        t_e = unique_t[mask]
        d_e = events_per[mask]
        n_e = at_risk[mask]
        hazard = d_e / n_e
        survival = np.cumprod(1.0 - hazard)
        incr = d_e / (n_e * (n_e - d_e + 1e-300))
        var_log_s = np.cumsum(incr)
        self.times_ = t_e
        self.survival_ = survival
        self.at_risk_ = n_e
        self.events_ = d_e
        self.var_log_s_ = var_log_s
        return self

    def confidence_interval(self, alpha: float=0.05):
        if self.survival_ is None:
            raise RuntimeError('fit() before confidence_interval')
        from math import erf
        z = _inv_std_normal(1 - alpha / 2)
        se_log_s = np.sqrt(self.var_log_s_)
        log_s = np.log(self.survival_ + 1e-300)
        lo = np.exp(log_s - z * se_log_s)
        hi = np.exp(log_s + z * se_log_s)
        return (np.minimum(hi, 1.0), np.maximum(lo, 0.0))

    def median_survival(self) -> float | None:
        if self.survival_ is None:
            raise RuntimeError('fit() before median_survival')
        below = np.where(self.survival_ <= 0.5)[0]
        if len(below) == 0:
            return None
        return float(self.times_[below[0]])

def logrank_test(durations: np.ndarray, events: np.ndarray, groups: np.ndarray) -> dict:
    t = np.asarray(durations, dtype=np.float64)
    d = np.asarray(events, dtype=np.float64)
    g = np.asarray(groups)
    uniq = np.unique(g)
    if len(uniq) != 2:
        raise ValueError(f'logrank_test expects exactly 2 groups; got {uniq}')
    g01 = (g == uniq[1]).astype(np.float64)
    order = np.argsort(t, kind='stable')
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
    chi2 = O1_minus_E1 ** 2 / V if V > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + erf(sqrt(chi2) / sqrt(2))))
    return {'chi2': float(chi2), 'df': 1, 'p_value': float(p), 'observed_minus_expected': float(O1_minus_E1), 'variance': float(V)}

def _inv_std_normal(p: float) -> float:
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        import math
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        import math
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
