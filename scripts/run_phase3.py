"""Phase 3: lasso Cox on a large TCGA-LUAD cohort + LUSC cross-validation.

    python scripts/run_phase3.py --n-luad 300 --n-lusc 200 --top-k-genes 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coxtcga.phase3 import cross_project_eval, run_luad_with_lasso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-luad", type=int, default=300)
    ap.add_argument("--n-lusc", type=int, default=200, help="0 to skip LUSC transfer eval")
    ap.add_argument("--top-k-genes", type=int, default=2000)
    args = ap.parse_args()

    out_root = Path(__file__).resolve().parent.parent
    luad_fit = run_luad_with_lasso(
        out_root, n_files=args.n_luad, top_k_genes=args.top_k_genes
    )
    print("\n=== LUAD lasso Cox summary ===")
    for k, v in luad_fit.items():
        if isinstance(v, (int, float, str)):
            print(f"{k}: {v}")

    if args.n_lusc > 0:
        try:
            lusc = cross_project_eval(luad_fit, "TCGA-LUSC", out_root, n_files=args.n_lusc)
            print("\n=== LUSC transfer ===")
            for k, v in lusc.items():
                print(f"{k}: {v}")
        except Exception as exc:
            print(f"\n[LUSC transfer skipped: {type(exc).__name__}: {exc}]")
    else:
        print("\n[LUSC transfer skipped by --n-lusc 0]")


if __name__ == "__main__":
    main()
