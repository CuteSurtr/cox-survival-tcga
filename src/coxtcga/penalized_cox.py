"""L1 / elastic-net penalised Cox regression via coordinate descent.

Implements the algorithm of Simon, Friedman, Hastie & Tibshirani (2011),
"Regularization paths for Cox's proportional hazards model via coordinate
descent" (JSS).

High-level idea
---------------
Around the current estimate beta_tilde the partial log-likelihood is
approximated quadratically:

    l(beta)  ~  l(beta_tilde)  -  0.5 * sum_i w_i (z_i - X_i' beta)^2

where z_i is a pseudo-response and w_i a subject weight, both derived from
the Cox score and the diagonal of the Hessian wrt the linear predictor
psi_i = X_i' beta:

    g_i = delta_i - pi_i * sum_{t <= t_i, d_t >= 1}  d_t / S0(t)
    h_i = pi_i *  sum_{t <= t_i, d_t >= 1}  d_t / S0(t) * (1 - pi_i / S0(t))
    z_i = psi_i + g_i / h_i
    w_i = h_i

With the (weighted) quadratic in hand, minimisation of

    Q(beta) = 0.5 sum_i w_i (z_i - X_i' beta)^2
              + lambda * (alpha * |beta|_1 + 0.5 * (1 - alpha) * |beta|_2^2)

reduces to cyclic coordinate descent with soft-thresholding.  We repeat the
quadratic approximation until beta converges, yielding an outer/inner loop
structure.

Only the core L1 case (alpha = 1) and the elastic-net case (0 < alpha <= 1)
are implemented.  Standardise your design matrix beforehand for comparable
coefficient magnitudes.
"""

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
    """Elastic-net-penalised Cox regression.

    Parameters
    ----------
    alpha : float in (0, 1]
        Mixing weight: alpha=1 is pure lasso, alpha<1 adds an L2 term.
    lam : float
        Regularisation strength. Must be >= 0.
    max_outer_iter : int
        Number of quadratic-approximation rounds.
    max_inner_iter : int
        Coordinate-descent passes per inner problem.
    tol : float
        Convergence tolerance on max |beta change| between outer iterations.
    """

    alpha: float = 1.0
    lam: float = 0.0
    max_outer_iter: int = 50
    max_inner_iter: int = 200
    tol: float = 1e-6

    beta_: np.ndarray | None = field(default=None, init=False)
    n_outer_iter_: int = field(default=0, init=False)
    objective_history_: list[float] = field(default_factory=list, init=False)

    # ---------------------------------------------------------------- public

    def fit(self, X: np.ndarray, durations: np.ndarray, events: np.ndarray) -> "PenalizedCox":
        X = np.asarray(X, dtype=np.float64)
        t = np.asarray(durations, dtype=np.float64)
        e = np.asarray(events, dtype=np.float64)
        n, p = X.shape
        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("alpha must be in (0, 1]")

        order = np.argsort(t, kind="stable")
        X_s = X[order]
        t_s = t[order]
        e_s = e[order]

        beta = np.zeros(p)
        self.objective_history_ = []

        for it_outer in range(self.max_outer_iter):
            psi = X_s @ beta
            g, h = self._score_and_hessian_diag(psi, t_s, e_s)
            # Avoid zero weights (cap at EPS).
            h = np.maximum(h, EPS)
            z = psi + g / h
            w = h

            # Inner cyclic coordinate descent on the weighted least-squares +
            # elastic net objective.
            beta_new = self._cd_inner(X_s, z, w, beta)

            # Track a valid objective (weighted SS + penalty) for diagnostics.
            ss = 0.5 * float((w * (z - X_s @ beta_new) ** 2).sum())
            pen = self.lam * (
                self.alpha * np.abs(beta_new).sum()
                + 0.5 * (1 - self.alpha) * float(beta_new @ beta_new)
            )
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
            raise RuntimeError("fit() before predict")
        return np.asarray(X, dtype=np.float64) @ self.beta_

    # ---------------------------------------------- Cox score + Hessian diag

    @staticmethod
    def _risk_set_reverse_cumsum(arr: np.ndarray, t_s: np.ndarray) -> np.ndarray:
        """Tail sums with tie-aware propagation (same as CoxPH)."""
        out = np.flip(np.cumsum(np.flip(arr)))
        for i in range(len(t_s) - 2, -1, -1):
            if t_s[i] == t_s[i + 1]:
                out[i] = out[i + 1]
        return out

    @classmethod
    def _score_and_hessian_diag(
        cls, psi: np.ndarray, t_s: np.ndarray, e_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-subject gradient g_i and Hessian diagonal h_i wrt psi_i."""
        pi = np.exp(psi - psi.max())  # numerical stability shift (cancels in ratios)
        S0 = cls._risk_set_reverse_cumsum(pi, t_s)

        # For each subject i and each event time t_k <= t_i, accumulate
        # d_k / S0(t_k).  We aggregate events per unique time first.
        event_mask = e_s == 1
        unique_t, inv = np.unique(t_s[event_mask], return_inverse=True)
        d_at = np.bincount(inv)

        # For each unique event time pick the S0 value (tied subjects share one).
        # t_s is sorted ascending; the first occurrence of unique_t[k] in t_s
        # gives the correct S0.
        first_ix = np.searchsorted(t_s, unique_t, side="left")
        S0_at = S0[first_ix]

        hazard_at = d_at / (S0_at + EPS)  # d_t / S0(t) per unique event time

        # For each subject i, sum hazard_at over event times <= t_i.
        # Equivalent to a cumsum of hazard_at evaluated at the index of t_i.
        hazard_cum = np.cumsum(hazard_at)
        # ix[i] = number of event times with t_event <= t_i.
        ix = np.searchsorted(unique_t, t_s, side="right")
        cumhaz_for_subject = np.where(ix > 0, hazard_cum[np.clip(ix - 1, 0, None)], 0.0)

        g = e_s - pi * cumhaz_for_subject

        # Hessian diagonal: h_i = sum_{t <= t_i} d_t pi_i / S0(t) * (1 - pi_i / S0(t))
        # Split into  A = pi_i * sum_{t<=t_i} d_t / S0(t)
        #       and   B = pi_i^2 * sum_{t<=t_i} d_t / S0(t)^2
        hazard_sq_at = d_at / (S0_at ** 2 + EPS)
        hazard_sq_cum = np.cumsum(hazard_sq_at)
        cumhaz_sq_for_subject = np.where(ix > 0, hazard_sq_cum[np.clip(ix - 1, 0, None)], 0.0)
        h = pi * cumhaz_for_subject - (pi ** 2) * cumhaz_sq_for_subject
        return g, h

    # -------------------------------------------------- coord-descent solver

    def _cd_inner(
        self,
        X: np.ndarray,
        z: np.ndarray,
        w: np.ndarray,
        beta_warm: np.ndarray,
    ) -> np.ndarray:
        """Cyclic coordinate descent for the weighted-lasso inner problem."""
        n, p = X.shape
        beta = beta_warm.copy()
        resid = z - X @ beta  # current residual

        wx2 = (w[:, None] * X ** 2).sum(axis=0)  # (p,) column norms

        for _ in range(self.max_inner_iter):
            max_change = 0.0
            for j in range(p):
                # Contribution of coordinate j to the inner product.
                wjxj = w * X[:, j]
                # Re-include the current beta_j into the residual.
                r_no_j = resid + X[:, j] * beta[j]
                num = float((wjxj * r_no_j).sum())
                den = wx2[j] + self.lam * (1 - self.alpha) + EPS
                new_bj = _soft_threshold(num, self.lam * self.alpha) / den
                change = new_bj - beta[j]
                if change != 0:
                    resid = r_no_j - X[:, j] * new_bj
                    beta[j] = new_bj
                    if abs(change) > max_change:
                        max_change = abs(change)
            if max_change < self.tol:
                break
        return beta
