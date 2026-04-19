"""End-to-end pipeline: TCGA project -> clinical Cox analysis.

Phase 1 (this module): clinical-only Cox + KM + log-rank on age/sex/stage.
Phase 2 (to extend): integrate RNA-seq expression for prognostic-gene
screens.

The clinical-only path is fast (one API call, <1 MB download) and already
demonstrates the full mathematical stack on real cancer data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coxtcga.coxph import CoxPH
from coxtcga.download import fetch_clinical_tsv, load_clinical
from coxtcga.km import KaplanMeier, logrank_test
from coxtcga.metrics import concordance_index


def harmonise_stage(s: str) -> str | None:
    """Collapse AJCC sub-stages into I/II/III/IV."""
    if not isinstance(s, str):
        return None
    s = s.upper().replace("STAGE ", "").strip()
    if s.startswith("IV"):
        return "IV"
    if s.startswith("III"):
        return "III"
    if s.startswith("II"):
        return "II"
    if s.startswith("I"):
        return "I"
    return None


def prepare_design_matrix(clin: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (design_df, covariate_names).

    Covariates: age, is_male, stage_II, stage_III, stage_IV (dummies with I as
    reference). Drops rows with any missing covariate.
    """
    df = clin.copy()
    df["stage_clean"] = df["stage"].map(harmonise_stage)
    df = df.dropna(subset=["age", "stage_clean"])
    df["is_male"] = (df["sex"].str.lower() == "male").astype(float)
    for cat in ["II", "III", "IV"]:
        df[f"stage_{cat}"] = (df["stage_clean"] == cat).astype(float)
    covariates = ["age", "is_male", "stage_II", "stage_III", "stage_IV"]
    X = df[["submitter_id", "os_time_days", "os_event"] + covariates].copy()
    return X.reset_index(drop=True), covariates


def plot_km_by_stage(clin: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for stage in ["I", "II", "III", "IV"]:
        sub = clin[clin["stage_clean"] == stage]
        if len(sub) < 5:
            continue
        km = KaplanMeier().fit(sub["os_time_days"].values, sub["os_event"].values)
        ax.step(km.times_, km.survival_, where="post", label=f"Stage {stage} (n={len(sub)})")
    ax.set_xlabel("Days")
    ax.set_ylabel("Overall survival probability")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run(project_id: str, out_root: Path) -> dict:
    data_dir = out_root / "data"
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch + load clinical.
    raw = fetch_clinical_tsv(project_id, data_dir / "raw" / f"{project_id}_clinical.tsv")
    clin_raw = load_clinical(raw)
    print(f"[{project_id}] {len(clin_raw)} cases with usable survival times")

    # 2. Build design matrix.
    X_df, covariates = prepare_design_matrix(clin_raw)
    X_df["stage_clean"] = (
        clin_raw.set_index("submitter_id").loc[X_df["submitter_id"], "stage"].map(harmonise_stage).values
    )
    print(f"[{project_id}] {len(X_df)} cases retained after dropping missing covariates")

    # 3. Cox PH.
    X = X_df[covariates].values
    t = X_df["os_time_days"].values
    e = X_df["os_event"].values

    model = CoxPH(max_iter=200, tol=1e-9).fit(X, t, e)
    print("\nCox PH coefficients:")
    summary = model.summary(names=covariates)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    # 4. C-index on risk scores.
    eta = model.predict_log_partial_hazard(X)
    c = concordance_index(t, e, eta)
    print(f"\nConcordance index: {c:.3f}")

    # 5. KM + log-rank.
    plot_km_by_stage(X_df.assign(os_time_days=t, os_event=e),
                     fig_dir / f"{project_id}_km_by_stage.png",
                     title=f"{project_id}: Kaplan-Meier by AJCC stage")
    print(f"saved figures/{project_id}_km_by_stage.png")

    # Two-group log-rank: early (I+II) vs late (III+IV).
    g_late = X_df["stage_clean"].isin(["III", "IV"]).astype(int).values
    lr = logrank_test(t, e, g_late)
    print(f"Log-rank (early I/II vs late III/IV): chi2 = {lr['chi2']:.3f}, p = {lr['p_value']:.3e}")

    # 6. Forest plot of hazard ratios with 95% CI.
    plot_hazard_ratios(summary_df, fig_dir / f"{project_id}_forest.png",
                        title=f"{project_id}: Cox PH hazard ratios")

    # 7. Save processed design matrix and results.
    proc_dir = data_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    X_df.to_csv(proc_dir / f"{project_id}_design.tsv", sep="\t", index=False)
    summary_df.to_csv(proc_dir / f"{project_id}_cox_summary.tsv", sep="\t", index=False)

    return {
        "n_cases": int(len(X_df)),
        "cox_summary": summary_df,
        "c_index": float(c),
        "logrank_chi2": float(lr["chi2"]),
        "logrank_p": float(lr["p_value"]),
    }


def plot_hazard_ratios(summary_df: pd.DataFrame, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(summary_df) + 1.5))
    ys = np.arange(len(summary_df))[::-1]
    hrs = summary_df["hr"].values
    # 95% CI for HR via exp(beta +/- 1.96 * se).
    lo = np.exp(summary_df["beta"].values - 1.96 * summary_df["se"].values)
    hi = np.exp(summary_df["beta"].values + 1.96 * summary_df["se"].values)
    ax.errorbar(hrs, ys, xerr=[hrs - lo, hi - hrs], fmt="o", capsize=3, color="black")
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(summary_df["name"].values)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
