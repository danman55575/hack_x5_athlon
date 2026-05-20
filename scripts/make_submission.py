"""Берёт predictions из нескольких экспериментов и делает blend.
Пример:
    python -m scripts.make_submission \
        --predictions experiments/predictions/lgbm_*.csv experiments/predictions/xgb_*.csv \
        --weights 0.6 0.4 \
        --out data/submissions/blend_v1.csv
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", nargs="+", required=True,
                   help="Список csv-файлов с колонками new_id, rto")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Веса для блендинга. Если не задано — равномерно.")
    p.add_argument("--mode", choices=["mean", "gmean"], default="gmean",
                   help="mean — арифм. среднее, gmean — геом. среднее (рекомендуется для MAPE).")
    p.add_argument("--out", default="data/submissions/blend.csv")
    args = p.parse_args()

    paths = [Path(p) for p in args.predictions]
    dfs = [pd.read_csv(p) for p in paths]
    for d, p in zip(dfs, paths):
        assert {"new_id", "rto"} <= set(d.columns), f"{p} missing cols"
        d.sort_values("new_id", inplace=True)
        d.reset_index(drop=True, inplace=True)

    base_ids = dfs[0]["new_id"].values
    for d in dfs[1:]:
        assert np.array_equal(d["new_id"].values, base_ids), "new_id mismatch between files"

    n = len(dfs)
    weights = np.array(args.weights if args.weights else [1.0] * n, dtype=np.float64)
    assert len(weights) == n, "weights count must match predictions count"
    weights = weights / weights.sum()

    preds = np.stack([d["rto"].values for d in dfs], axis=0)  # (n, N)
    if args.mode == "mean":
        blended = (preds * weights[:, None]).sum(axis=0)
    else:  # geometric mean — устойчивее для MAPE
        log_preds = np.log(np.clip(preds, 1.0, None))
        blended = np.exp((log_preds * weights[:, None]).sum(axis=0))

    out = pd.DataFrame({"new_id": base_ids.astype(int), "rto": blended})
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved blend: {out_path}  ({len(out)} rows)")
    print(f"Sources: {[p.name for p in paths]}")
    print(f"Weights: {weights.tolist()}  Mode: {args.mode}")


if __name__ == "__main__":
    main()