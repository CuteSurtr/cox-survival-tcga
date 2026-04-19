"""Run the clinical Cox PH + KM + log-rank pipeline on one TCGA project.

    python scripts/run_survival.py --project TCGA-LUAD
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coxtcga.pipeline import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="TCGA-LUAD")
    args = ap.parse_args()
    out_root = Path(__file__).resolve().parent.parent
    summary = run(args.project, out_root)
    print("\n=== Summary ===")
    for k, v in summary.items():
        if k == "cox_summary":
            continue
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
