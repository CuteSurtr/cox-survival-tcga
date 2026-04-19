from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter
from coxtcga.penalized_cox import PenalizedCox

def _simulate(n=500, p=10, n_active=3, beta_scale=0.7, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    active = rng.choice(p, size=n_active, replace=False)
    beta[active] = beta_scale * rng.choice([-1.0, 1.0], size=n_active)
    hazard = np.exp(X @ beta)
    T = rng.exponential(1.0 / hazard)
    C = rng.exponential(1.0 / 0.3, size=n)
    t = np.minimum(T, C)
    e = (T <= C).astype(np.float64)
    return (X, t, e, beta, active)

def test_unpenalized_matches_unpenalised_mle():
    X, t, e, _, _ = _simulate(n=800, p=4, n_active=4, seed=1)
    pc = PenalizedCox(alpha=1.0, lam=0.0, max_outer_iter=60, tol=1e-08).fit(X, t, e)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
    df['T'] = t
    df['E'] = e
    lf = CoxPHFitter().fit(df, duration_col='T', event_col='E')
    np.testing.assert_allclose(pc.beta_, lf.params_.values, atol=0.05)

def test_large_lambda_kills_all_coefficients():
    X, t, e, _, _ = _simulate(n=300, p=20, n_active=3, seed=2)
    pc = PenalizedCox(alpha=1.0, lam=100.0, max_outer_iter=30).fit(X, t, e)
    assert np.allclose(pc.beta_, 0.0), f'beta should be zero at large lambda; got {pc.beta_}'

def test_sparse_truth_recovered_in_top_active_set():
    X, t, e, beta_true, active = _simulate(n=800, p=50, n_active=3, beta_scale=0.9, seed=3)
    best_hits = 0
    best_zeros = 0
    beta_warm = np.zeros(X.shape[1])
    for lam in [40, 20, 10, 5, 2.5]:
        pc = PenalizedCox(alpha=1.0, lam=lam, max_outer_iter=40, max_inner_iter=500).fit(X, t, e)
        ranked = np.argsort(-np.abs(pc.beta_))[:5]
        hits = set(ranked) & set(active)
        zeros = int((pc.beta_ == 0).sum())
        best_hits = max(best_hits, len(hits))
        best_zeros = max(best_zeros, zeros)
    assert best_hits >= 2, f'lasso failed to recover true actives; best_hits={best_hits}'
    assert best_zeros >= 25, f'lasso failed to zero out enough irrelevant features; best_zeros={best_zeros}'

def test_regularization_path_continuity():
    X, t, e, _, _ = _simulate(n=300, p=15, n_active=3, seed=4)
    b_a = PenalizedCox(alpha=1.0, lam=2.0).fit(X, t, e).beta_
    b_b = PenalizedCox(alpha=1.0, lam=2.2).fit(X, t, e).beta_
    assert np.linalg.norm(b_a - b_b) < 0.2, 'neighbouring lambdas should give close solutions'
