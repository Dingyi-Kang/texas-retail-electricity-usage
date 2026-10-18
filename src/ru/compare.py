"""Paired comparison of two A1 runs: WAPE difference (pp) with 7-day block
bootstrap CI, folds won. Usage: python -m src.ru.compare EXP_A EXP_B [--holdout]"""
import glob, sys
import pandas as pd
from .metrics import daily_err, block_bootstrap_diff


def load(e, holdout=False):
    fs = [f for f in sorted(glob.glob(f"experiments/runs/*_{e}{'_HOLDOUT' if holdout else ''}/predictions.parquet"))
          if holdout or "HOLDOUT" not in f]
    return pd.read_parquet(fs[-1])


def main():
    a, b = sys.argv[1], sys.argv[2]; h = "--holdout" in sys.argv
    A, B = load(a, h), load(b, h)
    d, ci = block_bootstrap_diff(daily_err(A), daily_err(B))
    fw = sum(1 for f in A["fold"].unique()
             if ((A[A.fold == f].pred - A[A.fold == f].y).abs() * A[A.fold == f].w).sum()
             < ((B[B.fold == f].pred - B[B.fold == f].y).abs() * B[B.fold == f].w).sum())
    print(f"{a} - {b}: {d:+.2f} pp, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], folds won {fw}/{A['fold'].nunique()}")


if __name__ == "__main__":
    main()
