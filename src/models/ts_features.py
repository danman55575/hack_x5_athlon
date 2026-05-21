"""ETS / Holt-Winters прогноз на каждый таргетный месяц для каждого магазина.
Кэшируется в parquet, считается параллельно."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings("ignore")


def _ets_one(series: np.ndarray) -> float:
    s = pd.Series(series).dropna().astype(float).values
    if len(s) < 4:
        return float(s[-1]) if len(s) else 0.0
    try:
        if len(s) >= 24:
            model = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=12,
                                         initialization_method="estimated")
        elif len(s) >= 12:
            model = ExponentialSmoothing(s, trend="add", seasonal=None,
                                         initialization_method="estimated")
        else:
            return float(s[-1])
        fit = model.fit(optimized=True)
        return float(fit.forecast(1)[0])
    except Exception:
        return float(s[-1])


def compute_ets_features(df: pd.DataFrame, target: str = "rto",
                          n_jobs: int = -1, min_t_for_prediction: int = 6) -> pd.DataFrame:
    """Для каждой пары (store_id, t) считаем ETS-прогноз на основе данных store_id < t.
    Возвращает df с колонкой 'ets_pred'."""
    df = df.sort_values(["store_id", "t"]).copy()
    rows = []

    def _process_store(sid, sub):
        sub = sub.sort_values("t")
        ys = sub[target].values
        ts = sub["t"].values
        preds = []
        for i, t in enumerate(ts):
            if i < min_t_for_prediction:
                preds.append(np.nan)
            else:
                preds.append(_ets_one(ys[:i]))
        return pd.DataFrame({"store_id": sid, "t": ts, "ets_pred": preds})

    grouped = df.groupby("store_id")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(_process_store)(sid, sub) for sid, sub in grouped
    )
    out = pd.concat(results, ignore_index=True)
    out["ets_pred"] = out["ets_pred"].astype(np.float32)
    return out
