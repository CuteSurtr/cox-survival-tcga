"""From-scratch Cox proportional hazards regression.

Model
-----
For subject i with covariate vector x_i in R^p, event indicator delta_i in {0,1},
and observed time t_i, Cox (1972) posits

    lambda(t | x_i) = lambda_0(t) * exp(beta' x_i)

where lambda_0 is an arbitrary (nuisance) baseline hazard. The partial
likelihood eliminates lambda_0 entirely and depends only on the ranks of
observed event times:

    L(beta) = prod_{i: delta_i=1} exp(beta' x_i) / sum_{j in R(t_i)} exp(beta' x_j)

with R(t_i) = {j : t_j >= t_i} the risk set just before time t_i.

The log partial likelihood is

    l(beta) = sum_{i: delta_i=1} [ beta' x_i - log sum_{j in R(t_i)} exp(beta' x_j) ]

Its score function is

    U(beta) = sum_{i: delta_i=1} [ x_i - mean_R(x | risk set i) ]

where mean_R is the exp(beta'x)-weighted covariate mean over the risk set.
The observed information matrix is

    I(beta) = sum_{i: delta_i=1} Var_R(x | risk set i)

We maximise l(beta) by Newton-Raphson:

    beta_{k+1} = beta_k + I(beta_k)^{-1} U(beta_k)

Ties in event times are handled via the Breslow approximation (the default
in most software, including lifelines and R's survival::coxph).

References
----------
- Cox, D. R. (1972). Regression models and life-tables. JRSSB 34(2): 187-220.
- Kalbfleisch & Prentice. The Statistical Analysis of Failure Time Data, 2e.
- Therneau & Grambsch. Modeling Survival Data: Extending the Cox Model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoxPH:
    """Cox proportional hazards estimator.

    Parameters
    ----------
    max_iter : int
        Maximum Newton-Raphson iterations.
    tol : float
        Convergence tolerance on the relative change in log partial likelihood.
    ridge : float
        Ridge (L2) penalty on ``beta``. Stabilises the Hessian under
        collinearity or when p > n is approached. 0 for pure MLE.
    """

    max_iter: int = 100
    tol: float = 1e-7
    ridge: float = 0.0

    beta_: np.ndarray | None = field(default=None, init=False)
    log_lik_history_: list[float] = field(default_factory=list, init=False)
    n_iter_: int = field(default=0, init=False)
    fisher_information_: np.ndarray | None = field(default=None, init=False)
    baseline_cumhaz_: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False)

    # -------------------------------------------------------------- public API

    def fit(self, X: np.ndarray, durations: np.ndarray, events: np.ndarray) -> "CoxPH":
        """Fit beta by Newton-Raphson on the partial log-likelihood.

        Parameters
        ----------
        X : array of shape (n, p)
            Covariate matrix.
        durations : array of shape (n,)
            Observed times (event or censoring time).
        events : array of shape (n,)
            Event indicator, 1 = event observed, 0 = right-censored.
        """
        X = np.asarray(X, dtype=np.float64)
        durations = np.asarray(durations, dtype=np.float64)
        events = np.asarray(events, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        n, p = X.shape
        if durations.shape != (n,) or events.shape != (n,):
            raise ValueError("durations and events must be 1D of length n")
        if not np.all((events == 0) | (events == 1)):
            raise ValueError("events must be 0/1")

        # Sort everyone by time (ascending). This lets us compute risk-set
        # sums as *reverse* cumulative sums, since the risk set at time t_i
        # is {j : t_j >= t_i}, i.e. the tail of a time-sorted array.
        order = np.argsort(durations, kind="stable")
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

            # Newton step with Moore-Penrose fallback if singular.
            try:
                step = np.linalg.solve(hess, score)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, score, rcond=None)[0]

            # Simple backtracking line search to ensure monotone improvement.
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

        # Final information matrix at beta.
        _, hess = self._score_and_info(X_s, t_s, d_s, beta)
        if self.ridge > 0:
            hess = hess + self.ridge * np.eye(p)
        self.fisher_information_ = hess
        self.beta_ = beta
        self.baseline_cumhaz_ = self._breslow_cumhaz(X_s, t_s, d_s, beta)
        return self

    def predict_log_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError("fit() before predict")
        return np.asarray(X, dtype=np.float64) @ self.beta_

    def predict_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_partial_hazard(X))

    def standard_errors(self) -> np.ndarray:
        """sqrt of the diagonal of I(beta)^{-1}."""
        if self.fisher_information_ is None:
            raise RuntimeError("fit() before standard_errors")
        inv = np.linalg.inv(self.fisher_information_)
        return np.sqrt(np.diag(inv))

    def summary(self, names: list[str] | None = None) -> list[dict]:
        """Per-coefficient beta, HR, SE, z, and Wald p-value."""
        if self.beta_ is None:
            raise RuntimeError("fit() before summary")
        se = self.standard_errors()
        z = self.beta_ / se
        # Two-sided Wald p-value without scipy: use the standard-normal CDF.
        from math import erf, sqrt as msqrt

        p_values = np.array([2 * (1 - 0.5 * (1 + erf(abs(zi) / msqrt(2.0)))) for zi in z])
        names = names or [f"x{i}" for i in range(len(self.beta_))]
        return [
            {
                "name": names[i],
                "beta": float(self.beta_[i]),
                "hr": float(np.exp(self.beta_[i])),
                "se": float(se[i]),
                "z": float(z[i]),
                "p": float(p_values[i]),
            }
            for i in range(len(self.beta_))
        ]

    # ----------------------------------------------- partial likelihood kernels

    @staticmethod
    def _risk_set_reverse_cumsum(arr: np.ndarray, t_s: np.ndarray) -> np.ndarray:
        """Given a time-sorted array, return risk-set sums aware of ties.

        For each i, out[i] = sum of arr[j] over j such that t_s[j] >= t_s[i].
        Ties in time get the same sum (everyone at that exact time is in each
        other's risk set under Breslow).
        """
        # First compute the reverse cumulative sum (tail sums).
        # cumsum_from_end[i] = sum(arr[i:])
        cumsum_rev = np.flip(np.cumsum(np.flip(arr)))
        # For ties, propagate the sum from the earliest-index tied subject.
        # We want the first occurrence of each unique time to supply the sum.
        n = len(t_s)
        out = cumsum_rev.copy()
        # Walk backward, where consecutive equal times share the later value.
        for i in range(n - 2, -1, -1):
            if t_s[i] == t_s[i + 1]:
                out[i] = out[i + 1]
        return out

    def _partial_loglik(self, X_s, t_s, d_s, beta) -> float:
        eta = X_s @ beta  # linear predictor (n,)
        w = np.exp(eta)
        S0 = self._risk_set_reverse_cumsum(w, t_s)  # sum_{j in R(t_i)} exp(eta_j)
        # Breslow log partial likelihood over events.
        event_mask = d_s == 1
        return float(np.sum(eta[event_mask] - np.log(S0[event_mask] + 1e-300)))

    def _score_and_info(self, X_s, t_s, d_s, beta):
        n, p = X_s.shape
        eta = X_s @ beta
        w = np.exp(eta)

        # S0[i] = sum_{j in R(t_i)} w_j
        S0 = self._risk_set_reverse_cumsum(w, t_s)  # (n,)
        # S1[i] = sum_{j in R(t_i)} w_j x_j  (p,)
        wX = X_s * w[:, None]  # (n, p)
        S1 = np.empty_like(wX)
        for k in range(p):
            S1[:, k] = self._risk_set_reverse_cumsum(wX[:, k], t_s)
        # S2[i] = sum_{j in R(t_i)} w_j x_j x_j'  (p, p)
        # We compute S2 lazily: only for event rows. Precompute per-subject
        # outer products would be O(n p^2) memory; instead, accumulate via
        # reverse-cumsum on each of the p*(p+1)/2 upper-triangle entries.
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
        return score, info

    # --------------------------------------------------- baseline cum. hazard

    def _breslow_cumhaz(self, X_s, t_s, d_s, beta):
        """Breslow estimator for the baseline cumulative hazard H_0(t).

        H_0(t) = sum_{event times t_i <= t} d_i / S0(t_i)

        where d_i = number of events at t_i (ties) and S0 is as above.
        """
        w = np.exp(X_s @ beta)
        S0 = self._risk_set_reverse_cumsum(w, t_s)
        # Aggregate events at each unique event time.
        event_mask = d_s == 1
        t_ev = t_s[event_mask]
        S0_ev = S0[event_mask]
        unique_t, inv = np.unique(t_ev, return_inverse=True)
        d = np.bincount(inv)
        # Under ties, multiple events at the same time share the same S0 value.
        S0_unique = np.empty_like(unique_t)
        for k, _ in enumerate(unique_t):
            S0_unique[k] = S0_ev[np.searchsorted(t_ev, unique_t[k])]
        incr = d / (S0_unique + 1e-300)
        return unique_t, np.cumsum(incr)
