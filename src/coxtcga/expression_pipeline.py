from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coxtcga.coxph import CoxPH
from coxtcga.download import download_rna_seq, load_clinical, load_expression_matrix, query_rna_seq_files
from coxtcga.km import KaplanMeier, logrank_test
from coxtcga.metrics import concordance_index
from coxtcga.pipeline import prepare_design_matrix

def build_expression_matrix(project_id: str, data_dir: Path, n_files: int) -> tuple[pd.DataFrame, list[dict]]:
    rna_dir = data_dir / 'raw' / f'{project_id}_rna'
    rna_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = rna_dir / f'{project_id}_rna_manifest.json'
    if manifest_path.exists():
        hits = json.loads(manifest_path.read_text(encoding='utf-8'))
    else:
        hits = query_rna_seq_files(project_id, size=n_files)
        manifest_path.write_text(json.dumps(hits, indent=2), encoding='utf-8')
    paths = download_rna_seq(project_id, rna_dir, limit=n_files)
    print(f'[{project_id}] parsed {len(paths)} RNA-seq files')
    M = load_expression_matrix(paths, hits[:len(paths)], value_col='tpm_unstranded')
    return (M, hits[:len(paths)])

def merge_clinical_expression(clinical_path: Path, M: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clin = load_clinical(clinical_path)
    shared = sorted(set(clin['submitter_id']) & set(M.columns))
    if not shared:
        raise RuntimeError('no submitter_id overlap between clinical + expression')
    clin = clin[clin['submitter_id'].isin(shared)].drop_duplicates('submitter_id')
    clin = clin.set_index('submitter_id').loc[shared].reset_index()
    E = M[shared].copy()
    print(f'overlap: {len(shared)} patients')
    return (clin, E)

def univariate_cox_screen(clin: pd.DataFrame, E: pd.DataFrame, min_expression: float=1.0, min_expressed_fraction: float=0.25) -> pd.DataFrame:
    t = clin['os_time_days'].values.astype(np.float64)
    e = clin['os_event'].values.astype(np.float64)
    expressed = (E >= min_expression).mean(axis=1) >= min_expressed_fraction
    E_f = E[expressed]
    x_mat = np.log2(E_f.values + 1.0)
    n = x_mat.shape[1]
    print(f'screening {E_f.shape[0]} genes (of {E.shape[0]} total) on {n} samples')
    records: list[dict] = []
    var = x_mat.var(axis=1)
    keep = np.where(var > 0.01)[0]
    for idx in keep:
        x = x_mat[idx].reshape(-1, 1)
        try:
            m = CoxPH(max_iter=30, tol=1e-06).fit(x, t, e)
            s = m.summary(names=['expr'])[0]
            records.append({'gene': E_f.index[idx], 'beta': s['beta'], 'hr': s['hr'], 'se': s['se'], 'z': s['z'], 'p': s['p']})
        except Exception:
            continue
    res = pd.DataFrame(records)
    if res.empty:
        res['q'] = []
        return res
    res = res.sort_values('p').reset_index(drop=True)
    m = len(res)
    q = np.minimum(res['p'].values * m / (np.arange(m) + 1), 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    res['q'] = q
    return res

def nested_cv_cindex(clin: pd.DataFrame, E: pd.DataFrame, top_k: int=10, n_folds: int=5, ridge: float=0.001, seed: int=0) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(clin))
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    oof_eta = np.zeros(len(clin))
    for k in range(n_folds):
        test_ix = folds[k]
        train_ix = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        clin_tr = clin.iloc[train_ix].reset_index(drop=True)
        clin_te = clin.iloc[test_ix].reset_index(drop=True)
        E_tr = E[clin_tr['submitter_id']]
        E_te = E[clin_te['submitter_id']]
        screen = univariate_cox_screen(clin_tr, E_tr)
        fold_top = screen['gene'].head(top_k).tolist()
        X_tr_df, clin_covs = prepare_design_matrix(clin_tr)
        gene_tr = np.log2(E_tr.loc[fold_top, X_tr_df['submitter_id']].T + 1.0)
        gene_tr.columns = [f'gene_{g}' for g in fold_top]
        tr_mean = gene_tr.mean()
        tr_std = gene_tr.std(ddof=0).replace(0, 1)
        gene_tr = (gene_tr - tr_mean) / tr_std
        full_tr = pd.concat([X_tr_df.reset_index(drop=True), gene_tr.reset_index(drop=True)], axis=1)
        covs = clin_covs + list(gene_tr.columns)
        model = CoxPH(max_iter=300, tol=1e-09, ridge=ridge).fit(full_tr[covs].values, full_tr['os_time_days'].values, full_tr['os_event'].values)
        X_te_df, _ = prepare_design_matrix(clin_te)
        gene_te = np.log2(E_te.loc[fold_top, X_te_df['submitter_id']].T + 1.0)
        gene_te.columns = [f'gene_{g}' for g in fold_top]
        gene_te = (gene_te - tr_mean) / tr_std
        full_te = pd.concat([X_te_df.reset_index(drop=True), gene_te.reset_index(drop=True)], axis=1)
        eta_te = model.predict_log_partial_hazard(full_te[covs].values)
        order_map = {sid: i for i, sid in enumerate(clin.iloc[test_ix]['submitter_id'].values)}
        for sid, eta_val in zip(X_te_df['submitter_id'].values, eta_te):
            oof_eta[test_ix[order_map[sid]]] = eta_val
    c = concordance_index(clin['os_time_days'].values, clin['os_event'].values, oof_eta)
    return {'cv_c_index': float(c), 'n_folds': n_folds}

def cross_val_cindex(full: pd.DataFrame, covariates: list[str], n_folds: int=5, ridge: float=0.001, seed: int=0) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(full))
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    oof_eta = np.zeros(len(full))
    for k in range(n_folds):
        test_ix = folds[k]
        train_ix = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        train = full.iloc[train_ix]
        test = full.iloc[test_ix]
        X_tr = train[covariates].values
        t_tr = train['os_time_days'].values
        e_tr = train['os_event'].values
        try:
            m = CoxPH(max_iter=300, tol=1e-08, ridge=ridge).fit(X_tr, t_tr, e_tr)
        except Exception:
            m = CoxPH(max_iter=300, tol=1e-08, ridge=max(ridge, 0.1)).fit(X_tr, t_tr, e_tr)
        oof_eta[test_ix] = m.predict_log_partial_hazard(test[covariates].values)
    c = concordance_index(full['os_time_days'].values, full['os_event'].values, oof_eta)
    return {'cv_c_index': float(c), 'n_folds': n_folds, 'oof_eta': oof_eta}

def fit_multivariate(clin: pd.DataFrame, E: pd.DataFrame, top_genes: list[str]) -> tuple[CoxPH, pd.DataFrame, pd.DataFrame, list[str]]:
    X_df, clin_covs = prepare_design_matrix(clin)
    gene_df = np.log2(E.loc[top_genes, X_df['submitter_id']].T + 1.0)
    gene_df.columns = [f'gene_{g}' for g in top_genes]
    gene_df = (gene_df - gene_df.mean()) / gene_df.std(ddof=0).replace(0, 1)
    full = pd.concat([X_df.reset_index(drop=True), gene_df.reset_index(drop=True)], axis=1)
    covariates = clin_covs + list(gene_df.columns)
    t = full['os_time_days'].values
    e = full['os_event'].values
    X = full[covariates].values
    model = CoxPH(max_iter=200, tol=1e-09, ridge=0.0001).fit(X, t, e)
    summary = pd.DataFrame(model.summary(names=covariates))
    return (model, summary, full, covariates)

def plot_top_gene_km(clin: pd.DataFrame, E: pd.DataFrame, gene: str, out_path: Path):
    x = np.log2(E.loc[gene, clin['submitter_id']].values + 1.0)
    med = np.median(x)
    high = x >= med
    t = clin['os_time_days'].values
    e = clin['os_event'].values
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, mask in [(f'{gene} high (n={high.sum()})', high), (f'{gene} low  (n={(~high).sum()})', ~high)]:
        km = KaplanMeier().fit(t[mask], e[mask])
        ax.step(km.times_, km.survival_, where='post', label=label)
    lr = logrank_test(t, e, high.astype(int))
    ax.set_xlabel('Days')
    ax.set_ylabel('Overall survival probability')
    ax.set_title(f"{gene}: KM stratified at median (log-rank p = {lr['p_value']:.2e})")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return lr

def run(project_id: str, out_root: Path, n_files: int=100, top_k: int=10) -> dict:
    data_dir = out_root / 'data'
    fig_dir = out_root / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    from coxtcga.download import fetch_clinical_tsv
    clinical_path = fetch_clinical_tsv(project_id, data_dir / 'raw' / f'{project_id}_clinical.tsv')
    M, _hits = build_expression_matrix(project_id, data_dir, n_files)
    print(f'expression matrix: {M.shape[0]} genes x {M.shape[1]} samples')
    clin, E = merge_clinical_expression(clinical_path, M)
    screen = univariate_cox_screen(clin, E)
    screen_path = data_dir / 'processed' / f'{project_id}_univariate_screen.tsv'
    screen_path.parent.mkdir(parents=True, exist_ok=True)
    screen.to_csv(screen_path, sep='\t', index=False)
    n_sig = int((screen['q'] < 0.05).sum())
    print(f'univariate screen: {n_sig} genes with BH q < 0.05 / {len(screen)} tested')
    top_genes = screen['gene'].head(top_k).tolist()
    print(f'top-{top_k} by Wald p: {top_genes}')
    model, summary, full, covs = fit_multivariate(clin, E, top_genes)
    print('\nmultivariate Cox (clinical + top genes):')
    print(summary.to_string(index=False))
    eta = model.predict_log_partial_hazard(full[covs].values)
    c_in = concordance_index(full['os_time_days'].values, full['os_event'].values, eta)
    print(f'in-sample c-index = {c_in:.3f}  (optimistic; reflects training-set fit)')
    cv_clin_only = cross_val_cindex(full, covs[:5], n_folds=5, ridge=0.001)
    cv_full_shared = cross_val_cindex(full, covs, n_folds=5, ridge=0.001)
    print(f"5-fold CV c-index (clinical only):         {cv_clin_only['cv_c_index']:.3f}")
    print(f"5-fold CV c-index (clin+top, shared genes):{cv_full_shared['cv_c_index']:.3f}  [selection bias: genes picked on all data]")
    print('\nrunning nested-CV with per-fold gene re-selection...')
    nested = nested_cv_cindex(clin, E, top_k=len(top_genes), n_folds=5, ridge=0.001)
    print(f"5-fold NESTED CV c-index (clin+genes):     {nested['cv_c_index']:.3f}  [honest out-of-sample]")
    delta = nested['cv_c_index'] - cv_clin_only['cv_c_index']
    print(f'  delta from adding expression (nested):   {delta:+.3f}')
    lr = plot_top_gene_km(clin, E, top_genes[0], fig_dir / f'{project_id}_top_gene_km.png')
    print(f"top gene {top_genes[0]} KM log-rank p = {lr['p_value']:.2e}")
    summary.to_csv(data_dir / 'processed' / f'{project_id}_multivariate_cox.tsv', sep='\t', index=False)
    return {'n_patients': int(len(clin)), 'n_genes_tested': int(len(screen)), 'n_fdr_significant': n_sig, 'top_genes': top_genes, 'c_index_in_sample': float(c_in), 'c_index_cv_clinical_only': float(cv_clin_only['cv_c_index']), 'c_index_cv_clinical_plus_genes_leaky': float(cv_full_shared['cv_c_index']), 'c_index_nested_cv': float(nested['cv_c_index']), 'top_gene_logrank_p': float(lr['p_value'])}
