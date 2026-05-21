"""Стекинг OOF предсказаний нескольких моделей через Ridge в log-space.
Пример: см. scripts/run_pipeline.py stage stack."""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import Ridge
from src.validation.metrics import mape, mape_to_score


def load_oof(paths):
    frames, names = [], []
    for p in paths:
        df = pd.read_parquet(p)
        col = Path(p).stem.replace("_oof", "")
        df = df.rename(columns={"oof_pred": col})
        frames.append(df[["store_id", "t", "year", "month", "rto", col]])
        names.append(col)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["store_id", "t", "year", "month", "rto"], how="inner")
    return out, names


def load_test_predictions(paths, names_expected):
    dfs = []
    for p, n in zip(paths, names_expected):
        d = pd.read_csv(p).rename(columns={"rto": n})
        dfs.append(d[["new_id", n]])
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="new_id")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True)
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--out", default="data/submissions/stack.csv")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    oof_df, names = load_oof(args.oof)
    pred_df = load_test_predictions(args.predictions, names)
    valid = oof_df[names + ["rto"]].dropna()
    y_true = valid["rto"].values
    X = np.log(np.clip(valid[names].values, 1.0, None))
    y = np.log(np.clip(y_true, 1.0, None))
    meta = Ridge(alpha=args.alpha, positive=True)
    meta.fit(X, y)
    print("Meta weights:")
    for n, w in zip(names, meta.coef_):
        print(f"  {n:30s}: {w:+.4f}")
    print(f"  intercept: {meta.intercept_:+.4f}")

    pred_oof = np.exp(meta.predict(X))
    m = mape(y_true, pred_oof)
    print(f"OOF stacked MAPE: {m:.4f}  Score: {mape_to_score(m):.3f}")
    for n in names:
        print(f"  single [{n}] MAPE: {mape(y_true, valid[n].values):.4f}")

    Xt = np.log(np.clip(pred_df[names].values, 1.0, None))
    final = np.exp(meta.predict(Xt))
    sub = pd.DataFrame({"new_id": pred_df["new_id"].astype(int).values, "rto": final})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
