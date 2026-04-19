"""Phase 3: penalised Cox on large TCGA cohorts + cross-project validation.

Contents:
  * variable_gene_filter -- reduce G to the top-K most variable log-TPM genes.
  * lasso_cox_cv         -- nested CV over the lambda path; return the best
                            lambda and the corresponding coefficients.
  * run_luad_with_lasso  -- end-to-end LUAD pipeline with lasso feature sel.
  * cross_project_eval   -- train on one project, evaluate c-index on another.

Design notes
------------
Our Python-level coordinate descent is O(n p) per inner sweep and thousands
of passes on p = 20 000 genes would take minutes per lambda. For a
portfolio-friendly runtime we *first* apply a variance filter to retain the
~K most variable log2(TPM+1) genes (default K = 2000), *then* run lasso
Cox. This is the standard glmnet-for-genomics preprocessing step.
"""

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


# ------------------------------------------------------------- preprocessing

def variable_gene_filter(E: pd.DataFrame, top_k: int = 2000, min_tpm: float = 1.0) -> pd.DataFrame:
    """Keep the top-k most variable genes by log2(TPM+1) variance."""
    log_e = np.log2(E + 1.0)
    expressed = (E >= min_tpm).mean(axis=1) >= 0.2
    log_e = log_e[expressed]
    var = log_e.var(axis=1)
    top = var.sort_values(ascending=False).head(top_k).index
    return log_e.loc[top]


def standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score columns and return (Z, mean, std) for later test-set application."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


# ---------------------------------------------------------- penalised Cox CV

def lasso_cox_cv(
    X: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    n_lambdas: int = 30,
    lambda_min_ratio: float = 0.05,
    n_folds: int = 5,
    seed: int = 0,
    penalty_factor: np.ndarray | None = None,
) -> dict:
    """Nested-style CV over the lasso lambda path.

    Fit the full regularisation path on each training fold; score held-out
    patients by c-index at every lambda; average over folds.

    Returns a dict with ``lambdas``, ``cv_cindex`` (shape n_lambdas), the
    best lambda and its CV c-index, and the final betas refit on all data.
    """
    n = len(durations)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)

    # Lambda grid from all-data lambda_max.
    lam_max = PenalizedCox.lambda_max(X, durations, events, alpha=1.0)
    log_lams = np.linspace(np.log(lam_max), np.log(lam_max * lambda_min_ratio), n_lambdas)
    lams = np.exp(log_lams)

    # Per-fold out-of-fold eta at each lambda.
    oof = np.zeros((n_lambdas, n))
    for k in range(n_folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        print(f"  fold {k + 1}/{n_folds}: train n={len(tr)}, test n={len(te)}", flush=True)

        # Standardise using training-fold stats.
        X_tr, mu, sd = standardise(X[tr])
        X_te = (X[te] - mu) / sd

        # Fit along the shared outer lambda grid (warm-started down the path).
        _, betas_fold = PenalizedCox.fit_path(
            X_tr, durations[tr], events[tr],
            alpha=1.0, lambdas=lams,
            penalty_factor=penalty_factor,
            max_outer_iter=15, max_inner_iter=200, tol=1e-5,
        )
        # Score each lambda on the held-out fold.
        for i in range(n_lambdas):
            oof[i, te] = X_te @ betas_fold[i]

    cv_cindex = np.array([
        concordance_index(durations, events, oof[i]) for i in range(n_lambdas)
    ])
    best_i = int(np.nanargmax(cv_cindex))

    # Refit at best lambda on all data.
    X_all, mu, sd = standardise(X)
    final_model = PenalizedCox(
        alpha=1.0, lam=float(lams[best_i]),
        penalty_factor=penalty_factor,
        max_outer_iter=30, max_inner_iter=500, tol=1e-6,
    ).fit(X_all, durations, events)

    return {
        "lambdas": lams,
        "cv_cindex": cv_cindex,
        "best_lambda_idx": best_i,
        "best_lambda": float(lams[best_i]),
        "best_cv_cindex": float(cv_cindex[best_i]),
        "beta_at_best": final_model.beta_,
        "standardise_mean": mu,
        "standardise_std": sd,
        "n_nonzero": int((final_model.beta_ != 0).sum()),
    }


# --------------------------------------------------- main LUAD + LUSC driver

def run_luad_with_lasso(
    out_root: Path,
    n_files: int = 300,
    top_k_genes: int = 2000,
) -> dict:
    data_dir = out_root / "data"
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    clinical_path = fetch_clinical_tsv("TCGA-LUAD", data_dir / "raw" / "TCGA-LUAD_clinical.tsv")
    M, _ = build_expression_matrix("TCGA-LUAD", data_dir, n_files=n_files)
    print(f"LUAD expression: {M.shape}")

    clin, E = merge_clinical_expression(clinical_path, M)

    # Variable-gene filter.
    log_E = variable_gene_filter(E, top_k=top_k_genes)
    print(f"after variance filter: {log_E.shape[0]} genes")

    # Build design with clinical covariates + genes.
    clin_df, clin_covs = prepare_design_matrix(clin)
    gene_block = log_E[clin_df["submitter_id"]].T.values  # (n, G)
    X_clin = clin_df[clin_covs].values
    X = np.concatenate([X_clin, gene_block], axis=1)
    all_covariate_names = clin_covs + [f"gene_{g}" for g in log_E.index]
    t = clin_df["os_time_days"].values
    e = clin_df["os_event"].values
    print(f"design: n={X.shape[0]} cases x p={X.shape[1]} covariates")

    # Clinical covariates are unpenalised; only gene coefficients get L1 shrinkage.
    # This is the standard glmnet-for-genomics setup: the model always has
    # age/sex/stage as baseline, and lasso only decides which genes to add.
    penalty_factor = np.ones(X.shape[1])
    penalty_factor[:len(clin_covs)] = 0.0  # clinical block

    # Nested CV for lambda selection. Wider grid (0.02*lam_max down to lam_max)
    # to give the lasso a chance to find a useful mid-path lambda.
    print("running lasso path with 5-fold CV ...")
    cv_res = lasso_cox_cv(
        X, t, e,
        n_lambdas=25, lambda_min_ratio=0.02, n_folds=5, seed=42,
        penalty_factor=penalty_factor,
    )
    print(f"best lambda = {cv_res['best_lambda']:.3g}  "
          f"cv c-index = {cv_res['best_cv_cindex']:.3f}  "
          f"non-zero coefficients = {cv_res['n_nonzero']}")

    # Plot CV curve.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.log10(cv_res["lambdas"]), cv_res["cv_cindex"], marker="o")
    ax.axvline(np.log10(cv_res["best_lambda"]), color="red", linestyle="--",
               label=f"best lambda (c = {cv_res['best_cv_cindex']:.3f})")
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("5-fold CV c-index")
    ax.set_title("Lasso Cox: cross-validated c-index vs lambda")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "TCGA-LUAD_lasso_path.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # List the non-zero genes at the best lambda.
    beta_best = cv_res["beta_at_best"]
    nonzero_ix = np.where(beta_best != 0)[0]
    nonzero = pd.DataFrame({
        "covariate": [all_covariate_names[i] for i in nonzero_ix],
        "beta_std": beta_best[nonzero_ix],
        "hr_std": np.exp(beta_best[nonzero_ix]),
    }).sort_values("beta_std", key=lambda s: -s.abs()).reset_index(drop=True)
    print("\nnon-zero covariates at best lambda:")
    print(nonzero.to_string(index=False))

    proc = data_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    nonzero.to_csv(proc / "TCGA-LUAD_lasso_nonzero.tsv", sep="\t", index=False)

    return {
        "n_patients": int(X.shape[0]),
        "n_genes_input": int(log_E.shape[0]),
        "n_covariates": int(X.shape[1]),
        "best_lambda": cv_res["best_lambda"],
        "best_cv_cindex": cv_res["best_cv_cindex"],
        "n_nonzero": cv_res["n_nonzero"],
        "covariate_names": all_covariate_names,
        "beta_at_best": beta_best,
        "standardise_mean": cv_res["standardise_mean"],
        "standardise_std": cv_res["standardise_std"],
        "gene_index": list(log_E.index),
        "clinical_covs": clin_covs,
    }


def cross_project_eval(
    luad_fit: dict,
    target_project: str,
    out_root: Path,
    n_files: int = 200,
) -> dict:
    """Apply the LUAD-trained lasso-Cox model to a different TCGA project."""
    data_dir = out_root / "data"
    clinical_path = fetch_clinical_tsv(target_project, data_dir / "raw" / f"{target_project}_clinical.tsv")
    M, _ = build_expression_matrix(target_project, data_dir, n_files=n_files)
    clin, E = merge_clinical_expression(clinical_path, M)

    # Reuse LUAD's filtered gene set (intersecting with LUSC's genes).
    luad_genes = [g for g in luad_fit["gene_index"] if g in E.index]
    missing = len(luad_fit["gene_index"]) - len(luad_genes)
    print(f"{target_project}: {missing} of {len(luad_fit['gene_index'])} training genes missing")

    log_E = np.log2(E.loc[luad_genes] + 1.0)
    # Fill missing genes with the *training* mean so that after applying
    # LUAD's standardisation (x - mu)/sd, the contribution is exactly zero
    # instead of -mu/sd (the default-fill-with-0 bug).
    n_clin = len(luad_fit["clinical_covs"])
    training_gene_means = luad_fit["standardise_mean"][n_clin:]
    full_gene_df = pd.DataFrame(
        index=luad_fit["gene_index"],
        columns=clin["submitter_id"],
        dtype=float,
    )
    for i, g in enumerate(luad_fit["gene_index"]):
        full_gene_df.loc[g] = float(training_gene_means[i])
    full_gene_df.loc[luad_genes] = log_E[clin["submitter_id"]].values

    clin_df, clin_covs = prepare_design_matrix(clin)
    gene_block = full_gene_df[clin_df["submitter_id"]].T.values
    X_clin = clin_df[clin_covs].values
    X_test = np.concatenate([X_clin, gene_block], axis=1)
    # Apply LUAD's standardisation parameters.
    X_test_std = (X_test - luad_fit["standardise_mean"]) / luad_fit["standardise_std"]

    eta = X_test_std @ luad_fit["beta_at_best"]
    c = concordance_index(
        clin_df["os_time_days"].values,
        clin_df["os_event"].values,
        eta,
    )
    return {
        "target_project": target_project,
        "n_test_patients": int(len(clin_df)),
        "c_index_transfer": float(c),
    }
