from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class CoxPH:
    max_iter: int = 100
    tol: float = 1e-07
    ridge: float = 0.0
    beta_: np.ndarray | None = field(default=None, init=False)
    log_lik_history_: list[float] = field(default_factory=list, init=False)
    n_iter_: int = field(default=0, init=False)
    fisher_information_: np.ndarray | None = field(default=None, init=False)
    baseline_cumhaz_: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, durations: np.ndarray, events: np.ndarray) -> 'CoxPH':
        X = np.asarray(X, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        events = np.asarray(events, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f'X must be 2D; got shape {X.shape}')
        n, p = X.shape
        if durations.shape != (n,) or events.shape != (n,):
            raise ValueError('durations and events must be 1D of length n')
        if not np.all((events == 0) | (events == 1)):
            raise ValueError('events must be 0/1')
        order = np.argsort(durations, kind='stable')
        X_s = X[order]
        t_s = durations[order]
        d_s = events[order]
        beta = np.zeros(p)
        self.log_lik_history_ = [self._partial_loglik(X_s, t_s, d_s, beta)]
        for it in range(self.max_iter):
            score, hess = self._score_and_info(X_s, t_s, d_s, beta)
            if self.ridge > 0:
                score = score - self.ridge * beta
                hess = hess + self.ridge * np.eye(p)
            try:
                step = np.linalg.solve(hess, score)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, score, rcond=None)[0]
            alpha = 1.0
            prev_ll = self.log_lik_history_[-1]
            for _ in range(30):
                trial = beta + alpha * step
                ll = self._partial_loglik(X_s, t_s, d_s, trial)
                if self.ridge > 0:
                    ll = ll - 0.5 * self.ridge * float(trial @ trial)
                if ll >= prev_ll - 1e-12:
                    break
                alpha *= 0.5
            beta = trial
            self.log_lik_history_.append(ll)
            rel = abs(ll - prev_ll) / (abs(prev_ll) + 1e-12)
            if rel < self.tol:
                self.n_iter_ = it + 1
                break
        else:
            self.n_iter_ = self.max_iter
        _, hess = self._score_and_info(X_s, t_s, d_s, beta)
        if self.ridge > 0:
            hess = hess + self.ridge * np.eye(p)
        self.fisher_information_ = hess
        self.beta_ = beta
        self.baseline_cumhaz_ = self._breslow_cumhaz(X_s, t_s, d_s, beta)
        return self

    def predict_log_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError('fit() before predict')
        return np.asarray(X, dtype=np.float64) @ self.beta_

    def predict_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_partial_hazard(X))

    def standard_errors(self) -> np.ndarray:
        if self.fisher_information_ is None:
            raise RuntimeError('fit() before standard_errors')
        inv = np.linalg.inv(self.fisher_information_)
        return np.sqrt(np.diag(inv))

    def summary(self, names: list[str] | None=None) -> list[dict]:
        if self.beta_ is None:
            raise RuntimeError('fit() before summary')
        se = self.standard_errors()
        z = self.beta_ / se
        from math import erf, sqrt as msqrt
        p_values = np.array([2 * (1 - 0.5 * (1 + erf(abs(zi) / msqrt(2.0)))) for zi in z])
        names = names or [f'x{i}' for i in range(len(self.beta_))]
        return [{'name': names[i], 'beta': float(self.beta_[i]), 'hr': float(np.exp(self.beta_[i])), 'se': float(se[i]), 'z': float(z[i]), 'p': float(p_values[i])} for i in range(len(self.beta_))]

    @staticmethod
    def _risk_set_reverse_cumsum(arr: np.ndarray, t_s: np.ndarray) -> np.ndarray:
        cumsum_rev = np.flip(np.cumsum(np.flip(arr)))
        n = len(t_s)
        out = cumsum_rev.copy()
        for i in range(n - 2, -1, -1):
            if t_s[i] == t_s[i + 1]:
                out[i] = out[i + 1]
        return out

    def _partial_loglik(self, X_s, t_s, d_s, beta) -> float:
        eta = X_s @ beta
        w = np.exp(eta)
        S0 = self._risk_set_reverse_cumsum(w, t_s)
        event_mask = d_s == 1
        return float(np.sum(eta[event_mask] - np.log(S0[event_mask] + 1e-300)))

    def _score_and_info(self, X_s, t_s, d_s, beta):
        n, p = X_s.shape
        eta = X_s @ beta
        w = np.exp(eta)
        S0 = self._risk_set_reverse_cumsum(w, t_s)
        wX = X_s * w[:, None]
        S1 = np.empty_like(wX)
        for k in range(p):
            S1[:, k] = self._risk_set_reverse_cumsum(wX[:, k], t_s)
        S2_all = np.empty((n, p, p))
        for a in range(p):
            for b in range(a, p):
                arr = w * X_s[:, a] * X_s[:, b]
                S2_all[:, a, b] = self._risk_set_reverse_cumsum(arr, t_s)
                if b != a:
                    S2_all[:, b, a] = S2_all[:, a, b]
        event_idx = np.where(d_s == 1)[0]
        score = np.zeros(p)
        info = np.zeros((p, p))
        for i in event_idx:
            mean = S1[i] / (S0[i] + 1e-300)
            score += X_s[i] - mean
            var = S2_all[i] / (S0[i] + 1e-300) - np.outer(mean, mean)
            info += var
        return (score, info)

    def _breslow_cumhaz(self, X_s, t_s, d_s, beta):
        w = np.exp(X_s @ beta)
        S0 = self._risk_set_reverse_cumsum(w, t_s)
        event_mask = d_s == 1
        t_ev = t_s[event_mask]
        S0_ev = S0[event_mask]
        unique_t, inv = np.unique(t_ev, return_inverse=True)
        d = np.bincount(inv)
        S0_unique = np.empty_like(unique_t)
        for k, _ in enumerate(unique_t):
            S0_unique[k] = S0_ev[np.searchsorted(t_ev, unique_t[k])]
        incr = d / (S0_unique + 1e-300)
        return (unique_t, np.cumsum(incr))
