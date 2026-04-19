from __future__ import annotations
import numpy as np
import pandas as pd
from coxtcga.expression_pipeline import univariate_cox_screen

def _simulate_cohort(n_patients: int=200, n_prognostic: int=5, n_null: int=200, prog_effect: float=0.8, seed: int=7):
    rng = np.random.default_rng(seed)
    G = n_prognostic + n_null
    logexpr = rng.normal(5.0, 2.0, size=(G, n_patients))
    logexpr = np.clip(logexpr, 0.0, None)
    tpm = np.exp2(logexpr) - 1.0
    tpm = np.clip(tpm, 0.0, None)
    beta = np.zeros(G)
    beta[:n_prognostic] = prog_effect
    z = (logexpr - logexpr.mean(axis=1, keepdims=True)) / (logexpr.std(axis=1, keepdims=True) + 1e-06)
    eta = z.T @ beta
    hazard = np.exp(eta)
    T = rng.exponential(1.0 / hazard)
    C = rng.exponential(1.0 / 0.3, size=n_patients)
    t_obs = np.minimum(T, C)
    event = (T <= C).astype(np.float64)
    expr_df = pd.DataFrame(tpm, index=[f'GENE_{i}' for i in range(G)], columns=[f'P{i:04d}' for i in range(n_patients)])
    clin_df = pd.DataFrame({'submitter_id': expr_df.columns, 'os_time_days': t_obs, 'os_event': event})
    true_pos = {f'GENE_{i}' for i in range(n_prognostic)}
    return (clin_df, expr_df, true_pos)

def test_univariate_screen_ranks_true_positives_highest():
    clin, expr, truth = _simulate_cohort(n_patients=300, n_prognostic=5, n_null=200, prog_effect=0.8, seed=1)
    res = univariate_cox_screen(clin, expr, min_expression=0.0, min_expressed_fraction=0.0)
    top_genes = set(res['gene'].head(15).tolist())
    hits = truth & top_genes
    assert len(hits) >= 4, f'only {len(hits)}/{len(truth)} true genes in top 15: {top_genes}'

def test_univariate_screen_bh_fdr_control_on_null_data():
    clin, expr, _ = _simulate_cohort(n_patients=200, n_prognostic=0, n_null=500, prog_effect=0.0, seed=2)
    res = univariate_cox_screen(clin, expr, min_expression=0.0, min_expressed_fraction=0.0)
    frac_sig = (res['q'] < 0.1).mean()
    assert frac_sig < 0.2, f'{frac_sig:.3f} of null genes had q<0.1; FDR appears broken'

def test_univariate_screen_q_monotone():
    clin, expr, _ = _simulate_cohort(n_patients=150, n_prognostic=3, n_null=100, prog_effect=0.6, seed=3)
    res = univariate_cox_screen(clin, expr, min_expression=0.0, min_expressed_fraction=0.0)
    q = res['q'].values
    assert np.all(np.diff(q) >= -1e-12), 'q-values must be non-decreasing with sorted p'
