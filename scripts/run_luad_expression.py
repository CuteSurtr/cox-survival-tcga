"""Phase 2: run the expression + survival pipeline on TCGA-LUAD.

    python scripts/run_luad_expression.py --n-files 100 --top-k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coxtcga.expression_pipeline import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="TCGA-LUAD")
    ap.add_argument("--n-files", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    out_root = Path(__file__).resolve().parent.parent
    summary = run(args.project, out_root, n_files=args.n_files, top_k=args.top_k)
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
