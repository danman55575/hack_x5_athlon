"""Загружает все OOF из experiments/oof, обучает Ridge-метамодель на (store_id, target_month),
делает финальный prediction для March 2025 как взвешенный blend (через коэффициенты Ridge).

Использование:
    python -m scripts.stack_oof \
        --oof experiments/oof/lgbm_*_oof.parquet experiments/oof/xgb_*_oof.parquet \
        --predictions experiments/predictions/lgbm_*.csv experiments/predictions/xgb_*.csv \
        --out data/submissions/stack_v1.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import Ridge
from src.validation.metrics import mape, mape_to_score


def load_oof(paths):
    """Возвращает df (store_id, t, year, month, rto) + по одной колонке на каждый файл."""
    frames = []
    names = []
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
    """Берёт прогнозы март-2025 из обычных submission CSV."""
    dfs = []
    for p, name in zip(paths, names_expected):
        d = pd.read_csv(p).rename(columns={"rto": name})
        dfs.append(d[["new_id", name]])
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
    ap.add_argument("--positive", action="store_true", default=True)
    ap.add_argument("--log", action="store_true", default=True,
                    help="Стекать в log-space (рекомендуется для MAPE)")
    args = ap.parse_args()

    assert len(args.oof) == len(args.predictions), "oof and predictions count must match"
    oof_df, names = load_oof(args.oof)
    pred_df = load_test_predictions(args.predictions, names)

    # маска валидных OOF (без NaN)
    valid = oof_df[names + ["rto"]].dropna()
    y_true = valid["rto"].values

    if args.log:
        X = np.log(np.clip(valid[names].values, 1.0, None))
        y = np.log(np.clip(y_true, 1.0, None))
    else:
        X = valid[names].values
        y = y_true

    meta = Ridge(alpha=args.alpha, positive=args.positive)
    meta.fit(X, y)
    print("Meta weights:")
    for n, w in zip(names, meta.coef_):
        print(f"  {n:30s}: {w: .4f}")
    print(f"  intercept: {meta.intercept_:.4f}")

    # in-sample MAPE
    pred_oof = meta.predict(X)
    if args.log:
        pred_oof = np.exp(pred_oof)
    m = mape(y_true, pred_oof)
    print(f"OOF stacked MAPE: {m:.4f}  Score: {mape_to_score(m):.3f}")

    # сравним с лучшим одиночным
    for n in names:
        m_n = mape(y_true, valid[n].values)
        print(f"  single [{n}] MAPE: {m_n:.4f}")

    # финальный прогноз
    if args.log:
        Xt = np.log(np.clip(pred_df[names].values, 1.0, None))
        final = np.exp(meta.predict(Xt))
    else:
        final = meta.predict(pred_df[names].values)

    sub = pd.DataFrame({"new_id": pred_df["new_id"].astype(int).values, "rto": final})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"Saved stacked submission: {args.out}")


if __name__ == "__main__":
    main()
