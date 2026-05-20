"""Простой baseline по ARIMA на отдельных рядах (медленный, опционально для стекинга)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings; warnings.filterwarnings("ignore")


def ets_forecast_single(series: np.ndarray, periods: int = 1) -> float:
    """Holt-Winters / ETS на один шаг. Возвращает прогноз."""
    s = pd.Series(series).dropna().astype(float).values
    if len(s) < 6:
        return float(s[-1]) if len(s) else 0.0
    try:
        if len(s) >= 24:
            model = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=12,
                                         initialization_method="estimated")
        else:
            model = ExponentialSmoothing(s, trend="add", seasonal=None,
                                         initialization_method="estimated")
        fit = model.fit(optimized=True)
        return float(fit.forecast(periods)[-1])
    except Exception:
        return float(s[-1])


def ets_predict_all(df: pd.DataFrame, target_t: int, target_col: str = "rto") -> pd.Series:
    """Для каждого магазина возвращает ETS-прогноз на target_t."""
    preds = {}
    for sid, g in df[df["t"] < target_t].sort_values(["store_id", "t"]).groupby("store_id"):
        preds[sid] = ets_forecast_single(g[target_col].values)
    return pd.Series(preds)
