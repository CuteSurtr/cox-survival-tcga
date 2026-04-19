from __future__ import annotations
import gzip
import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
GDC_FILES = 'https://api.gdc.cancer.gov/files'
GDC_DATA = 'https://api.gdc.cancer.gov/data/'
GDC_CLINICAL_BIOSPECIMEN = 'https://api.gdc.cancer.gov/clinical_analysis/TSV'
HEADERS = {'User-Agent': 'cox-survival-tcga/0.1'}
REQUEST_DELAY = 0.3

def fetch_clinical_tsv(project_id: str, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 1024:
        try:
            df_probe = pd.read_csv(out_path, sep='\t', nrows=5, low_memory=False)
            has_any = any((c in df_probe.columns for c in ('submitter_id', 'case_submitter_id', 'vital_status')))
            if has_any:
                return out_path
        except Exception:
            pass
        out_path.unlink(missing_ok=True)
    url = 'https://api.gdc.cancer.gov/clinical_analysis/TSV'
    payload = {'filters': {'op': 'in', 'content': {'field': 'cases.project.project_id', 'value': [project_id]}}}
    try:
        r = requests.post(url, json=payload, headers=HEADERS, timeout=120)
        ok = r.status_code == 200 and b'submitter_id' in r.content[:2048]
    except requests.RequestException:
        ok = False
    if not ok:
        return _fallback_clinical(project_id, out_path)
    out_path.write_bytes(r.content)
    time.sleep(REQUEST_DELAY)
    return out_path

def _fallback_clinical(project_id: str, out_path: Path) -> Path:
    url = 'https://api.gdc.cancer.gov/cases'
    fields = ['submitter_id', 'demographic.vital_status', 'demographic.days_to_death', 'demographic.age_at_index', 'demographic.gender', 'diagnoses.days_to_last_follow_up', 'diagnoses.ajcc_pathologic_stage', 'diagnoses.tumor_stage', 'diagnoses.primary_diagnosis']
    rows: list[dict] = []
    size = 500
    frm = 0
    while True:
        payload = {'filters': {'op': 'in', 'content': {'field': 'project.project_id', 'value': [project_id]}}, 'fields': ','.join(fields), 'format': 'JSON', 'size': str(size), 'from': str(frm)}
        r = requests.post(url, json=payload, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()['data']
        hits = data['hits']
        for h in hits:
            demo = h.get('demographic') or {}
            diags = (h.get('diagnoses') or [{}])[0]
            rows.append({'submitter_id': h.get('submitter_id'), 'vital_status': demo.get('vital_status'), 'days_to_death': demo.get('days_to_death'), 'days_to_last_follow_up': diags.get('days_to_last_follow_up'), 'age_at_index': demo.get('age_at_index'), 'gender': demo.get('gender'), 'stage': diags.get('ajcc_pathologic_stage') or diags.get('tumor_stage'), 'primary_diagnosis': diags.get('primary_diagnosis')})
        if len(hits) < size:
            break
        frm += size
        time.sleep(REQUEST_DELAY)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep='\t', index=False)
    return out_path

def load_clinical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', low_memory=False)
    vital_col = _first_present(df, ['vital_status', 'demographic.vital_status'])
    death_col = _first_present(df, ['days_to_death', 'demographic.days_to_death'])
    fup_col = _first_present(df, ['days_to_last_follow_up', 'diagnoses.days_to_last_follow_up'])
    age_col = _first_present(df, ['age_at_index', 'demographic.age_at_index'])
    sex_col = _first_present(df, ['gender', 'demographic.gender'])
    stage_col = _first_present(df, ['stage', 'ajcc_pathologic_stage', 'diagnoses.ajcc_pathologic_stage'])
    sid_col = _first_present(df, ['submitter_id', 'case_submitter_id'])
    if None in (vital_col, death_col, fup_col, sid_col):
        raise ValueError(f'clinical TSV missing expected columns. Present: {list(df.columns)}')
    vital = df[vital_col].astype(str).str.lower()
    event = (vital == 'dead').astype(float)
    time_days = pd.to_numeric(df[death_col], errors='coerce').where(event == 1, pd.to_numeric(df[fup_col], errors='coerce'))
    out = pd.DataFrame({'submitter_id': df[sid_col].astype(str), 'os_time_days': time_days, 'os_event': event, 'age': pd.to_numeric(df[age_col], errors='coerce') if age_col else np.nan, 'sex': df[sex_col].astype(str) if sex_col else '', 'stage': df[stage_col].astype(str) if stage_col else ''})
    out = out.dropna(subset=['os_time_days'])
    out = out[out['os_time_days'] > 0]
    return out.drop_duplicates(subset='submitter_id').reset_index(drop=True)

def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def query_rna_seq_files(project_id: str, size: int) -> list[dict]:
    payload = {'filters': {'op': 'and', 'content': [{'op': 'in', 'content': {'field': 'cases.project.project_id', 'value': [project_id]}}, {'op': 'in', 'content': {'field': 'data_type', 'value': ['Gene Expression Quantification']}}, {'op': 'in', 'content': {'field': 'analysis.workflow_type', 'value': ['STAR - Counts']}}, {'op': 'in', 'content': {'field': 'access', 'value': ['open']}}]}, 'fields': 'file_id,file_name,cases.submitter_id', 'size': str(size), 'format': 'JSON'}
    r = requests.post(GDC_FILES, json=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()['data']['hits']

def download_rna_seq(project_id: str, out_dir: Path, limit: int) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f'{project_id}_rna_manifest.json'
    if meta_path.exists():
        hits = json.loads(meta_path.read_text(encoding='utf-8'))
    else:
        hits = query_rna_seq_files(project_id, size=limit)
        meta_path.write_text(json.dumps(hits, indent=2), encoding='utf-8')
    paths: list[Path] = []
    for hit in hits[:limit]:
        fname = hit['file_name']
        local = out_dir / fname
        if not local.exists() or local.stat().st_size == 0:
            ok = False
            for attempt in range(5):
                try:
                    r = requests.get(GDC_DATA + hit['file_id'], headers=HEADERS, timeout=120)
                    if r.status_code == 200:
                        local.write_bytes(r.content)
                        time.sleep(REQUEST_DELAY)
                        ok = True
                        break
                except requests.RequestException as ex:
                    backoff = min(60, 2 ** attempt)
                    print(f'  ! {type(ex).__name__} on {fname}; retry {attempt + 1}/5 in {backoff}s', flush=True)
                    time.sleep(backoff)
            if not ok:
                print(f'  ! giving up on {fname}', flush=True)
                continue
        paths.append(local)
    return paths

def load_expression_matrix(paths: list[Path], hits_meta: list[dict], value_col: str='tpm_unstranded') -> pd.DataFrame:
    hits_by_name = {h['file_name']: h for h in hits_meta}
    cols: dict[str, pd.Series] = {}
    for p in paths:
        df = pd.read_csv(p, sep='\t', comment='#', low_memory=False)
        df = df[df['gene_id'].astype(str).str.startswith('ENSG')].copy()
        if value_col not in df.columns:
            raise ValueError(f'{p.name}: expected column {value_col!r} not in {list(df.columns)}')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        gene_label = 'gene_name' if 'gene_name' in df.columns else 'gene_id'
        values = df.set_index(gene_label)[value_col]
        if value_col in ('tpm_unstranded', 'fpkm_unstranded', 'fpkm_uq_unstranded'):
            values = values.groupby(level=0).mean()
        else:
            values = values.groupby(level=0).sum()
        meta = hits_by_name.get(p.name)
        sid = meta['cases'][0]['submitter_id'] if meta else p.stem
        cols[sid] = values
    return pd.DataFrame(cols).dropna(how='all')
