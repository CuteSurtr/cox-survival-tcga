from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coxtcga.coxph import CoxPH
from coxtcga.download import fetch_clinical_tsv, load_clinical, load_expression_matrix, query_rna_seq_files
from coxtcga.expression_pipeline import build_expression_matrix, merge_clinical_expression
from coxtcga.metrics import concordance_index
from coxtcga.penalized_cox import PenalizedCox
from coxtcga.pipeline import prepare_design_matrix

def variable_gene_filter(E: pd.DataFrame, top_k: int=2000, min_tpm: float=1.0) -> pd.DataFrame:
    log_e = np.log2(E + 1.0)
    expressed = (E >= min_tpm).mean(axis=1) >= 0.2
    log_e = log_e[expressed]
    var = log_e.var(axis=1)
    top = var.sort_values(ascending=False).head(top_k).index
    return log_e.loc[top]

def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd, mu, sd)

def lasso_cox_cv(X: np.ndarray, durations: np.ndarray, events: np.ndarray, n_lambdas: int=30, lambda_min_ratio: float=0.05, n_folds: int=5, seed: int=0, penalty_factor: np.ndarray | None=None) -> dict:
    n = len(durations)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    lam_max = PenalizedCox.lambda_max(X, durations, events, alpha=1.0)
    log_lams = np.linspace(np.log(lam_max), np.log(lam_max * lambda_min_ratio), n_lambdas)
    lams = np.exp(log_lams)
    oof = np.zeros((n_lambdas, n))
    for k in range(n_folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        print(f'  fold {k + 1}/{n_folds}: train n={len(tr)}, test n={len(te)}', flush=True)
        X_tr, mu, sd = standardise(X[tr])
        X_te = (X[te] - mu) / sd
        _, betas_fold = PenalizedCox.fit_path(X_tr, durations[tr], events[tr], alpha=1.0, lambdas=lams, penalty_factor=penalty_factor, max_outer_iter=15, max_inner_iter=200, tol=1e-05)
        for i in range(n_lambdas):
            oof[i, te] = X_te @ betas_fold[i]
    cv_cindex = np.array([concordance_index(durations, events, oof[i]) for i in range(n_lambdas)])
    best_i = int(np.nanargmax(cv_cindex))
    X_all, mu, sd = standardise(X)
    final_model = PenalizedCox(alpha=1.0, lam=float(lams[best_i]), penalty_factor=penalty_factor, max_outer_iter=30, max_inner_iter=500, tol=1e-06).fit(X_all, durations, events)
    return {'lambdas': lams, 'cv_cindex': cv_cindex, 'best_lambda_idx': best_i, 'best_lambda': float(lams[best_i]), 'best_cv_cindex': float(cv_cindex[best_i]), 'beta_at_best': final_model.beta_, 'standardise_mean': mu, 'standardise_std': sd, 'n_nonzero': int((final_model.beta_ != 0).sum())}

def run_luad_with_lasso(out_root: Path, n_files: int=300, top_k_genes: int=2000) -> dict:
    data_dir = out_root / 'data'
    fig_dir = out_root / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    clinical_path = fetch_clinical_tsv('TCGA-LUAD', data_dir / 'raw' / 'TCGA-LUAD_clinical.tsv')
    M, _ = build_expression_matrix('TCGA-LUAD', data_dir, n_files=n_files)
    print(f'LUAD expression: {M.shape}')
    clin, E = merge_clinical_expression(clinical_path, M)
    log_E = variable_gene_filter(E, top_k=top_k_genes)
    print(f'after variance filter: {log_E.shape[0]} genes')
    clin_df, clin_covs = prepare_design_matrix(clin)
    gene_block = log_E[clin_df['submitter_id']].T.values
    X_clin = clin_df[clin_covs].values
    X = np.concatenate([X_clin, gene_block], axis=1)
    all_covariate_names = clin_covs + [f'gene_{g}' for g in log_E.index]
    t = clin_df['os_time_days'].values
    e = clin_df['os_event'].values
    print(f'design: n={X.shape[0]} cases x p={X.shape[1]} covariates')
    penalty_factor = np.ones(X.shape[1])
    penalty_factor[:len(clin_covs)] = 0.0
    print('running lasso path with 5-fold CV ...')
    cv_res = lasso_cox_cv(X, t, e, n_lambdas=25, lambda_min_ratio=0.02, n_folds=5, seed=42, penalty_factor=penalty_factor)
    print(f"best lambda = {cv_res['best_lambda']:.3g}  cv c-index = {cv_res['best_cv_cindex']:.3f}  non-zero coefficients = {cv_res['n_nonzero']}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.log10(cv_res['lambdas']), cv_res['cv_cindex'], marker='o')
    ax.axvline(np.log10(cv_res['best_lambda']), color='red', linestyle='--', label=f"best lambda (c = {cv_res['best_cv_cindex']:.3f})")
    ax.set_xlabel('log10(lambda)')
    ax.set_ylabel('5-fold CV c-index')
    ax.set_title('Lasso Cox: cross-validated c-index vs lambda')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / 'TCGA-LUAD_lasso_path.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    beta_best = cv_res['beta_at_best']
    nonzero_ix = np.where(beta_best != 0)[0]
    nonzero = pd.DataFrame({'covariate': [all_covariate_names[i] for i in nonzero_ix], 'beta_std': beta_best[nonzero_ix], 'hr_std': np.exp(beta_best[nonzero_ix])}).sort_values('beta_std', key=lambda s: -s.abs()).reset_index(drop=True)
    print('\nnon-zero covariates at best lambda:')
    print(nonzero.to_string(index=False))
    proc = data_dir / 'processed'
    proc.mkdir(parents=True, exist_ok=True)
    nonzero.to_csv(proc / 'TCGA-LUAD_lasso_nonzero.tsv', sep='\t', index=False)
    return {'n_patients': int(X.shape[0]), 'n_genes_input': int(log_E.shape[0]), 'n_covariates': int(X.shape[1]), 'best_lambda': cv_res['best_lambda'], 'best_cv_cindex': cv_res['best_cv_cindex'], 'n_nonzero': cv_res['n_nonzero'], 'covariate_names': all_covariate_names, 'beta_at_best': beta_best, 'standardise_mean': cv_res['standardise_mean'], 'standardise_std': cv_res['standardise_std'], 'gene_index': list(log_E.index), 'clinical_covs': clin_covs}

def cross_project_eval(luad_fit: dict, target_project: str, out_root: Path, n_files: int=200) -> dict:
    data_dir = out_root / 'data'
    clinical_path = fetch_clinical_tsv(target_project, data_dir / 'raw' / f'{target_project}_clinical.tsv')
    M, _ = build_expression_matrix(target_project, data_dir, n_files=n_files)
    clin, E = merge_clinical_expression(clinical_path, M)
    luad_genes = [g for g in luad_fit['gene_index'] if g in E.index]
    missing = len(luad_fit['gene_index']) - len(luad_genes)
    print(f"{target_project}: {missing} of {len(luad_fit['gene_index'])} training genes missing")
    log_E = np.log2(E.loc[luad_genes] + 1.0)
    n_clin = len(luad_fit['clinical_covs'])
    training_gene_means = luad_fit['standardise_mean'][n_clin:]
    full_gene_df = pd.DataFrame(index=luad_fit['gene_index'], columns=clin['submitter_id'], dtype=float)
    for i, g in enumerate(luad_fit['gene_index']):
        full_gene_df.loc[g] = float(training_gene_means[i])
    full_gene_df.loc[luad_genes] = log_E[clin['submitter_id']].values
    clin_df, clin_covs = prepare_design_matrix(clin)
    gene_block = full_gene_df[clin_df['submitter_id']].T.values
    X_clin = clin_df[clin_covs].values
    X_test = np.concatenate([X_clin, gene_block], axis=1)
    X_test_std = (X_test - luad_fit['standardise_mean']) / luad_fit['standardise_std']
    eta = X_test_std @ luad_fit['beta_at_best']
    c = concordance_index(clin_df['os_time_days'].values, clin_df['os_event'].values, eta)
    return {'target_project': target_project, 'n_test_patients': int(len(clin_df)), 'c_index_transfer': float(c)}
