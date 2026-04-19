"""Download TCGA clinical + expression data via the GDC API.

Two datasets are pulled per project:
  1. Clinical TSV  -- time-to-event fields (days_to_death, days_to_last_follow_up,
     vital_status, age, sex, stage).
  2. Gene-level RNA-seq STAR counts (open-access "stranded_second" column)
     aggregated across the project.

GDC API docs:  https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/
"""

from __future__ import annotations

import gzip
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data/"
GDC_CLINICAL_BIOSPECIMEN = "https://api.gdc.cancer.gov/clinical_analysis/TSV"
HEADERS = {"User-Agent": "cox-survival-tcga/0.1"}
REQUEST_DELAY = 0.3


# ----------------------------------------------------------------- clinical

def fetch_clinical_tsv(project_id: str, out_path: Path) -> Path:
    """Download the clinical TSV for a TCGA project via the GDC clinical endpoint.

    Returns a DataFrame indexed by submitter_id (TCGA-XX-XXXX). Each row has
    survival fields usable directly in Cox / KM analyses.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 1024:
        return out_path

    # Use the GDC bulk-clinical TSV endpoint rather than per-case queries.
    url = "https://api.gdc.cancer.gov/clinical_analysis/TSV"
    payload = {
        "filters": {
            "op": "in",
            "content": {"field": "cases.project.project_id", "value": [project_id]},
        },
    }
    r = requests.post(url, json=payload, headers=HEADERS, timeout=120)
    if r.status_code != 200:
        # Fallback: /files search with clinical supplement filter, then download each.
        return _fallback_clinical(project_id, out_path)
    out_path.write_bytes(r.content)
    time.sleep(REQUEST_DELAY)
    return out_path


def _fallback_clinical(project_id: str, out_path: Path) -> Path:
    """Case-by-case clinical fetch via the /cases endpoint."""
    url = "https://api.gdc.cancer.gov/cases"
    fields = [
        "submitter_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "demographic.age_at_index",
        "demographic.gender",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.tumor_stage",
        "diagnoses.primary_diagnosis",
    ]
    rows: list[dict] = []
    size = 500
    frm = 0
    while True:
        payload = {
            "filters": {
                "op": "in",
                "content": {"field": "project.project_id", "value": [project_id]},
            },
            "fields": ",".join(fields),
            "format": "JSON",
            "size": str(size),
            "from": str(frm),
        }
        r = requests.post(url, json=payload, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()["data"]
        hits = data["hits"]
        for h in hits:
            demo = (h.get("demographic") or {})
            diags = (h.get("diagnoses") or [{}])[0]
            rows.append({
                "submitter_id": h.get("submitter_id"),
                "vital_status": demo.get("vital_status"),
                "days_to_death": demo.get("days_to_death"),
                "days_to_last_follow_up": diags.get("days_to_last_follow_up"),
                "age_at_index": demo.get("age_at_index"),
                "gender": demo.get("gender"),
                "stage": diags.get("ajcc_pathologic_stage") or diags.get("tumor_stage"),
                "primary_diagnosis": diags.get("primary_diagnosis"),
            })
        if len(hits) < size:
            break
        frm += size
        time.sleep(REQUEST_DELAY)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def load_clinical(path: Path) -> pd.DataFrame:
    """Load a clinical TSV and harmonise the survival columns.

    Produces columns:
      submitter_id, os_time_days, os_event, age, sex, stage
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)

    # Different GDC endpoints name the survival columns differently; try both.
    vital_col = _first_present(df, ["vital_status", "demographic.vital_status"])
    death_col = _first_present(df, ["days_to_death", "demographic.days_to_death"])
    fup_col = _first_present(df, ["days_to_last_follow_up",
                                   "diagnoses.days_to_last_follow_up"])
    age_col = _first_present(df, ["age_at_index", "demographic.age_at_index"])
    sex_col = _first_present(df, ["gender", "demographic.gender"])
    stage_col = _first_present(df, ["stage", "ajcc_pathologic_stage",
                                     "diagnoses.ajcc_pathologic_stage"])
    sid_col = _first_present(df, ["submitter_id", "case_submitter_id"])

    if None in (vital_col, death_col, fup_col, sid_col):
        raise ValueError(f"clinical TSV missing expected columns. Present: {list(df.columns)}")

    vital = df[vital_col].astype(str).str.lower()
    event = (vital == "dead").astype(float)
    time_days = pd.to_numeric(df[death_col], errors="coerce").where(
        event == 1, pd.to_numeric(df[fup_col], errors="coerce")
    )
    out = pd.DataFrame({
        "submitter_id": df[sid_col].astype(str),
        "os_time_days": time_days,
        "os_event": event,
        "age": pd.to_numeric(df[age_col], errors="coerce") if age_col else np.nan,
        "sex": df[sex_col].astype(str) if sex_col else "",
        "stage": df[stage_col].astype(str) if stage_col else "",
    })
    out = out.dropna(subset=["os_time_days"])
    out = out[out["os_time_days"] > 0]
    return out.drop_duplicates(subset="submitter_id").reset_index(drop=True)


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ----------------------------------------------------------------- expression

def query_rna_seq_files(project_id: str, size: int) -> list[dict]:
    """List open-access STAR-Counts RNA-seq files for a project."""
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id",
                                          "value": [project_id]}},
                {"op": "in", "content": {"field": "data_type",
                                          "value": ["Gene Expression Quantification"]}},
                {"op": "in", "content": {"field": "analysis.workflow_type",
                                          "value": ["STAR - Counts"]}},
                {"op": "in", "content": {"field": "access", "value": ["open"]}},
            ],
        },
        "fields": "file_id,file_name,cases.submitter_id",
        "size": str(size),
        "format": "JSON",
    }
    r = requests.post(GDC_FILES, json=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def download_rna_seq(project_id: str, out_dir: Path, limit: int) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{project_id}_rna_manifest.json"
    if meta_path.exists():
        hits = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        hits = query_rna_seq_files(project_id, size=limit)
        meta_path.write_text(json.dumps(hits, indent=2), encoding="utf-8")

    paths: list[Path] = []
    for hit in hits[:limit]:
        fname = hit["file_name"]
        local = out_dir / fname
        if not local.exists() or local.stat().st_size == 0:
            r = requests.get(GDC_DATA + hit["file_id"], headers=HEADERS, timeout=120)
            if r.status_code != 200:
                continue
            local.write_bytes(r.content)
            time.sleep(REQUEST_DELAY)
        paths.append(local)
    return paths


def load_expression_matrix(
    paths: list[Path],
    hits_meta: list[dict],
    value_col: str = "tpm_unstranded",
) -> pd.DataFrame:
    """Parse STAR-Counts TSVs and assemble a (gene x sample) matrix.

    GDC STAR-Counts file layout (verified against a downloaded file):

        # gene-model: GENCODE v36
        gene_id<TAB>gene_name<TAB>gene_type<TAB>unstranded<TAB>...
        N_unmapped<TAB><TAB><TAB>1784008<TAB>...
        N_multimapping<TAB>...
        N_noFeature<TAB>...
        N_ambiguous<TAB>...
        ENSG00000000003.15<TAB>TSPAN6<TAB>protein_coding<TAB>2497<TAB>...

    pd.read_csv with ``comment="#"`` drops line 1 so the header is read
    correctly. We then filter to rows whose gene_id begins with "ENSG".
    Duplicate gene_name entries (multiple ENSG ids mapping to the same
    HGNC symbol) are collapsed by summing counts / averaging expression.
    """
    hits_by_name = {h["file_name"]: h for h in hits_meta}
    cols: dict[str, pd.Series] = {}
    for p in paths:
        df = pd.read_csv(p, sep="\t", comment="#", low_memory=False)
        df = df[df["gene_id"].astype(str).str.startswith("ENSG")].copy()
        if value_col not in df.columns:
            raise ValueError(
                f"{p.name}: expected column {value_col!r} not in {list(df.columns)}"
            )
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        gene_label = "gene_name" if "gene_name" in df.columns else "gene_id"
        values = df.set_index(gene_label)[value_col]
        # For TPM / FPKM: average across duplicate symbols (preserves scale).
        # For raw counts: sum.
        if value_col in ("tpm_unstranded", "fpkm_unstranded", "fpkm_uq_unstranded"):
            values = values.groupby(level=0).mean()
        else:
            values = values.groupby(level=0).sum()
        meta = hits_by_name.get(p.name)
        sid = meta["cases"][0]["submitter_id"] if meta else p.stem
        cols[sid] = values
    return pd.DataFrame(cols).dropna(how="all")
