"""Validate from-scratch Cox PH + KM against the lifelines reference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test as lifelines_logrank
from lifelines.utils import concordance_index as lifelines_cindex

from coxtcga.coxph import CoxPH
from coxtcga.km import KaplanMeier, logrank_test
from coxtcga.metrics import concordance_index


def _simulate(n=500, p=3, seed=0):
    """Simulate from an exponential Cox model: T ~ Exp(exp(beta'x)), censoring
    independent Exp(lambda_c). Returns a DataFrame usable by lifelines."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta_true = np.array([0.5, -0.3, 0.1][:p])
    hazard = np.exp(X @ beta_true)
    T = rng.exponential(1.0 / hazard)
    C = rng.exponential(1.0 / 0.3, size=n)
    t_obs = np.minimum(T, C)
    event = (T <= C).astype(np.float64)
    return X, t_obs, event, beta_true


# ----------------------------------------------------------------- Cox PH

def test_coxph_agrees_with_lifelines_on_simulated_data():
    X, t, e, beta_true = _simulate(n=600, p=3, seed=1)

    ours = CoxPH(max_iter=100, tol=1e-8).fit(X, t, e)

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["T"] = t
    df["E"] = e
    lf = CoxPHFitter(penalizer=0.0).fit(df, duration_col="T", event_col="E")

    lf_beta = lf.params_.values
    np.testing.assert_allclose(ours.beta_, lf_beta, atol=5e-3), (
        f"beta mismatch: ours={ours.beta_}, lifelines={lf_beta}"
    )

    # Standard errors should also match (Fisher information is the same Hessian).
    ours_se = ours.standard_errors()
    lf_se = lf.standard_errors_.values
    np.testing.assert_allclose(ours_se, lf_se, rtol=1e-2)


def test_coxph_recovers_true_parameters():
    X, t, e, beta_true = _simulate(n=3000, p=3, seed=2)
    ours = CoxPH(max_iter=200, tol=1e-9).fit(X, t, e)
    # With n=3000 the MLE should be well within 0.1 of truth.
    np.testing.assert_allclose(ours.beta_, beta_true, atol=0.1)


def test_coxph_handles_light_ties():
    """With light tying, Breslow (ours) and Efron (lifelines default) agree closely."""
    X, t, e, _ = _simulate(n=500, p=2, seed=3)
    t_rounded = np.round(t, 3)  # few ties
    ours = CoxPH(max_iter=100, tol=1e-8).fit(X, t_rounded, e)
    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["T"] = t_rounded
    df["E"] = e
    lf = CoxPHFitter().fit(df, duration_col="T", event_col="E")
    np.testing.assert_allclose(ours.beta_, lf.params_.values, atol=1e-2)


def test_coxph_heavy_ties_stable():
    """Under heavy tying, Breslow and Efron diverge by O(ties^2 / n) but both
    must stay within a sensible neighbourhood and produce finite, monotone-ish
    log-likelihood trajectories."""
    X, t, e, _ = _simulate(n=400, p=2, seed=3)
    t_heavy = np.round(t, 1)  # many ties
    ours = CoxPH(max_iter=100, tol=1e-8).fit(X, t_heavy, e)
    df = pd.DataFrame(X, columns=["x0", "x1"])
    df["T"] = t_heavy
    df["E"] = e
    lf = CoxPHFitter().fit(df, duration_col="T", event_col="E")
    # Breslow vs Efron under heavy ties: documented divergence up to ~5e-2.
    np.testing.assert_allclose(ours.beta_, lf.params_.values, atol=5e-2)
    # Log-likelihood must be finite and non-decreasing.
    assert np.isfinite(ours.log_lik_history_).all()
    diffs = np.diff(ours.log_lik_history_)
    assert (diffs >= -1e-6).all(), "log-lik should be non-decreasing under Newton-Raphson"


# ----------------------------------------------------------- Kaplan-Meier

def test_km_matches_lifelines():
    X, t, e, _ = _simulate(n=400, p=1, seed=4)
    ours = KaplanMeier().fit(t, e)

    lf = KaplanMeierFitter().fit(t, e)
    # Evaluate lifelines' S at our step-point times.
    lf_s = lf.survival_function_at_times(ours.times_).values
    np.testing.assert_allclose(ours.survival_, lf_s, atol=1e-9)


# ------------------------------------------------------------ Log-rank

def test_logrank_matches_lifelines_two_groups():
    rng = np.random.default_rng(5)
    n = 400
    g = rng.integers(0, 2, size=n)
    hazard = np.where(g == 1, 2.0, 1.0)
    T = rng.exponential(1.0 / hazard)
    C = rng.exponential(1.0 / 0.4, size=n)
    t = np.minimum(T, C)
    e = (T <= C).astype(float)

    ours = logrank_test(t, e, g)
    lf = lifelines_logrank(t[g == 0], t[g == 1], e[g == 0], e[g == 1])
    assert abs(ours["chi2"] - lf.test_statistic) < 1e-6
    assert abs(ours["p_value"] - lf.p_value) < 1e-6


# ------------------------------------------------------ Concordance index

def test_cindex_matches_lifelines():
    X, t, e, beta_true = _simulate(n=300, p=3, seed=6)
    eta = X @ beta_true
    ours = concordance_index(t, e, eta)
    # lifelines uses the opposite sign convention: lower predicted time <=> higher risk.
    # Here we pass predicted survival times = -eta, consistent with our
    # "higher eta = higher risk" convention.
    lf = lifelines_cindex(event_times=t, predicted_scores=-eta, event_observed=e)
    assert abs(ours - lf) < 1e-6
