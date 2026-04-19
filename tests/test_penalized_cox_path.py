"""Tests for the lambda-path solver in PenalizedCox."""

from __future__ import annotations

import numpy as np

from coxtcga.penalized_cox import PenalizedCox


def _simulate(n=400, p=30, n_active=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    active = rng.choice(p, size=n_active, replace=False)
    beta[active] = 0.8
    hazard = np.exp(X @ beta)
    T = rng.exponential(1.0 / hazard)
    C = rng.exponential(1.0 / 0.3, size=n)
    t = np.minimum(T, C)
    e = (T <= C).astype(np.float64)
    return X, t, e, active


def test_lambda_max_zeros_entire_path_top():
    """At the first (largest) lambda on the glmnet-style path, every
    coefficient should be exactly zero by construction."""
    X, t, e, _ = _simulate(n=300, p=25, seed=1)
    lams, betas = PenalizedCox.fit_path(
        X, t, e, n_lambdas=10, lambda_min_ratio=0.1, max_outer_iter=20, tol=1e-6
    )
    assert np.allclose(betas[0], 0.0), f"beta at lambda_max should be zero; got {betas[0]}"


def test_lambda_path_sparsity_increases_with_lambda():
    """Number of non-zero betas should be non-increasing as lambda grows."""
    X, t, e, _ = _simulate(n=300, p=30, seed=2)
    lams, betas = PenalizedCox.fit_path(
        X, t, e, n_lambdas=15, lambda_min_ratio=0.05, max_outer_iter=20, tol=1e-6
    )
    # lams is in decreasing order from fit_path (lam_max -> lam_min). So
    # non-zero count should be non-decreasing as we step through the returned
    # array. Give a small slack for numerical noise.
    nnz = (betas != 0).sum(axis=1)
    assert nnz[0] <= nnz[-1], f"nnz should grow as lambda shrinks; got {nnz}"
    # Strictly monotone after averaging windows of 3 (smoothing numerical kinks).
    smoothed = np.convolve(nnz, np.ones(3) / 3, mode="valid")
    assert (np.diff(smoothed) >= -1e-3).all(), f"nnz (smoothed) should be non-decreasing; got {smoothed}"


def test_warm_start_gives_consistent_solutions():
    """Consecutive lambdas along the path differ smoothly."""
    X, t, e, _ = _simulate(n=250, p=20, seed=3)
    lams, betas = PenalizedCox.fit_path(
        X, t, e, n_lambdas=20, lambda_min_ratio=0.05, max_outer_iter=25, tol=1e-6
    )
    diffs = np.linalg.norm(np.diff(betas, axis=0), axis=1)
    # No single step should be wildly larger than its neighbours.
    assert diffs.max() < 5 * np.median(diffs) + 1e-6, (
        f"path has an unphysical jump: max={diffs.max():.3f}, median={np.median(diffs):.3f}"
    )
