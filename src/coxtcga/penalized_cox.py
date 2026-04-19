from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
EPS = 1e-10

def _soft_threshold(z: float, gamma: float) -> float:
    if gamma <= 0:
        return z
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0

@dataclass
class PenalizedCox:
    alpha: float = 1.0
    lam: float = 0.0
    max_outer_iter: int = 50
    max_inner_iter: int = 200
    tol: float = 1e-06
    penalty_factor: np.ndarray | None = None
    beta_: np.ndarray | None = field(default=None, init=False)
    n_outer_iter_: int = field(default=0, init=False)
    objective_history_: list[float] = field(default_factory=list, init=False)

    def fit(self, X: np.ndarray, durations: np.ndarray, events: np.ndarray) -> 'PenalizedCox':
        X = np.asarray(X, dtype=np.float64)
        t = np.asarray(durations, dtype=np.float64)
        e = np.asarray(events, dtype=np.float64)
        n, p = X.shape
        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError('alpha must be in (0, 1]')
        order = np.argsort(t, kind='stable')
        X_s = X[order]
        t_s = t[order]
        e_s = e[order]
        warm = getattr(self, '_warm_start', None)
        if warm is not None and len(warm) == p:
            beta = np.asarray(warm, dtype=np.float64).copy()
        else:
            beta = np.zeros(p)
        self.objective_history_ = []
        for it_outer in range(self.max_outer_iter):
            psi = X_s @ beta
            g, h = self._score_and_hessian_diag(psi, t_s, e_s)
            h = np.maximum(h, EPS)
            z = psi + g / h
            w = h
            beta_new = self._cd_inner(X_s, z, w, beta)
            ss = 0.5 * float((w * (z - X_s @ beta_new) ** 2).sum())
            pen = self.lam * (self.alpha * np.abs(beta_new).sum() + 0.5 * (1 - self.alpha) * float(beta_new @ beta_new))
            self.objective_history_.append(ss + pen)
            step = np.max(np.abs(beta_new - beta))
            beta = beta_new
            if step < self.tol:
                self.n_outer_iter_ = it_outer + 1
                break
        else:
            self.n_outer_iter_ = self.max_outer_iter
        self.beta_ = beta
        return self

    def predict_log_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError('fit() before predict')
        return np.asarray(X, dtype=np.float64) @ self.beta_

    @classmethod
    def lambda_max(cls, X: np.ndarray, durations: np.ndarray, events: np.ndarray, alpha: float=1.0) -> float:
        X = np.asarray(X, dtype=np.float64)
        t = np.asarray(durations, dtype=np.float64)
        e = np.asarray(events, dtype=np.float64)
        order = np.argsort(t, kind='stable')
        X_s = X[order]
        t_s = t[order]
        e_s = e[order]
        psi = np.zeros(len(t))
        g, _ = cls._score_and_hessian_diag(psi, t_s, e_s)
        score_j = X_s.T @ g
        return float(np.max(np.abs(score_j))) / max(alpha, 1e-12)

    @classmethod
    def fit_path(cls, X: np.ndarray, durations: np.ndarray, events: np.ndarray, alpha: float=1.0, n_lambdas: int=50, lambda_min_ratio: float=0.01, lambdas: np.ndarray | None=None, penalty_factor: np.ndarray | None=None, max_outer_iter: int=30, max_inner_iter: int=300, tol: float=1e-06) -> tuple[np.ndarray, np.ndarray]:
        if lambdas is None:
            lam_max = cls.lambda_max(X, durations, events, alpha=alpha)
            log_lams = np.linspace(np.log(lam_max), np.log(lam_max * lambda_min_ratio), n_lambdas)
            lams = np.exp(log_lams)
        else:
            lams = np.asarray(lambdas, dtype=np.float64)
            n_lambdas = len(lams)
        p = X.shape[1]
        betas = np.zeros((n_lambdas, p))
        beta_warm = np.zeros(p)
        for i, lam in enumerate(lams):
            model = cls(alpha=alpha, lam=float(lam), penalty_factor=penalty_factor, max_outer_iter=max_outer_iter, max_inner_iter=max_inner_iter, tol=tol)
            model._warm_start = beta_warm.copy()
            model.fit(X, durations, events)
            betas[i] = model.beta_
            beta_warm = model.beta_
        return (lams, betas)

    @staticmethod
    def _risk_set_reverse_cumsum(arr: np.ndarray, t_s: np.ndarray) -> np.ndarray:
        out = np.flip(np.cumsum(np.flip(arr)))
        for i in range(len(t_s) - 2, -1, -1):
            if t_s[i] == t_s[i + 1]:
                out[i] = out[i + 1]
        return out

    @classmethod
    def _score_and_hessian_diag(cls, psi: np.ndarray, t_s: np.ndarray, e_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pi = np.exp(psi - psi.max())
        S0 = cls._risk_set_reverse_cumsum(pi, t_s)
        event_mask = e_s == 1
        unique_t, inv = np.unique(t_s[event_mask], return_inverse=True)
        d_at = np.bincount(inv)
        first_ix = np.searchsorted(t_s, unique_t, side='left')
        S0_at = S0[first_ix]
        hazard_at = d_at / (S0_at + EPS)
        hazard_cum = np.cumsum(hazard_at)
        ix = np.searchsorted(unique_t, t_s, side='right')
        cumhaz_for_subject = np.where(ix > 0, hazard_cum[np.clip(ix - 1, 0, None)], 0.0)
        g = e_s - pi * cumhaz_for_subject
        hazard_sq_at = d_at / (S0_at ** 2 + EPS)
        hazard_sq_cum = np.cumsum(hazard_sq_at)
        cumhaz_sq_for_subject = np.where(ix > 0, hazard_sq_cum[np.clip(ix - 1, 0, None)], 0.0)
        h = pi * cumhaz_for_subject - pi ** 2 * cumhaz_sq_for_subject
        return (g, h)

    def _cd_inner(self, X: np.ndarray, z: np.ndarray, w: np.ndarray, beta_warm: np.ndarray) -> np.ndarray:
        n, p = X.shape
        beta = beta_warm.copy()
        resid = z - X @ beta
        wx2 = (w[:, None] * X ** 2).sum(axis=0)
        if self.penalty_factor is None:
            pf = np.ones(p)
        else:
            pf = np.asarray(self.penalty_factor, dtype=np.float64)
            if pf.shape != (p,):
                raise ValueError(f'penalty_factor must have shape ({p},); got {pf.shape}')
        for _ in range(self.max_inner_iter):
            max_change = 0.0
            for j in range(p):
                wjxj = w * X[:, j]
                r_no_j = resid + X[:, j] * beta[j]
                num = float((wjxj * r_no_j).sum())
                l1_j = self.lam * self.alpha * pf[j]
                l2_j = self.lam * (1 - self.alpha) * pf[j]
                den = wx2[j] + l2_j + EPS
                new_bj = _soft_threshold(num, l1_j) / den
                change = new_bj - beta[j]
                if change != 0:
                    resid = r_no_j - X[:, j] * new_bj
                    beta[j] = new_bj
                    if abs(change) > max_change:
                        max_change = abs(change)
            if max_change < self.tol:
                break
        return beta
