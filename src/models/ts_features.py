"""Robust ETS / Holt-Winters прогноз для каждого (store, t).

Ключевые улучшения vs предыдущей версии:
1. Работаем на log-scale (retail-РТО имеет multiplicative dynamics — рост в %).
2. Defensive winsorization исходного ряда (clip 2% и 98% перцентили).
3. Перебор конфигов: damped/non-damped trend × additive seasonal / no seasonal /
   level-only. Выбор лучшего по AIC.
4. Робастный fallback: для коротких рядов или после всех неудач — среднее (last, lag12).
5. Прогноз всегда положительный и финитный (защита от NaN/inf/<=0).

Эти изменения должны существенно повысить качество ets_pred и снять "шумность",
о которой говорилось в анализе.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings("ignore")


def _safe_fit_forecast(log_s: np.ndarray, kwargs: dict, periods: int = 1):
    """Возвращает (pred_in_original_scale, aic) или (None, +inf) при ошибке."""
    try:
        model = ExponentialSmoothing(log_s, initialization_method="estimated", **kwargs)
        fit = model.fit(optimized=True, use_brute=False)
        aic = getattr(fit, "aic", None)
        if aic is None or not np.isfinite(aic):
            resid = getattr(fit, "resid", None)
            aic = float(np.sum(np.asarray(resid) ** 2)) if resid is not None else np.inf
        fc = fit.forecast(periods)
        pred = float(np.exp(fc[-1]))
        if not np.isfinite(pred) or pred <= 0:
            return None, np.inf
        return pred, float(aic)
    except Exception:
        return None, np.inf


def _ets_one(series: np.ndarray, periods: int = 1) -> float:
    s = pd.Series(series).dropna().astype(float).values
    n = len(s)
    if n == 0:
        return 0.0
    if n < 3:
        return float(s[-1])

    # 1) Defensive winsorization (clip 2%/98%).
    if n >= 8:
        lo, hi = np.quantile(s, [0.01, 0.99])
        if hi > lo > 0:
            s_clip = np.clip(s, lo, hi)
        else:
            s_clip = s.copy()
    else:
        s_clip = s.copy()

    # 2) Log-transform: retail-РТО ведёт себя multiplicative.
    log_s = np.log(np.maximum(s_clip, 1.0))

    # 3) Перебор конфигов и выбор лучшего по AIC.
    configs = []
    if n >= 25:
        configs.append(dict(trend="add", seasonal="add", seasonal_periods=12, damped_trend=True))
        configs.append(dict(trend="add", seasonal="add", seasonal_periods=12, damped_trend=False))
    if n >= 14:
        configs.append(dict(trend="add", damped_trend=True))
        configs.append(dict(trend="add", damped_trend=False))
    configs.append(dict(trend=None))  # SES-fallback всегда

    best_pred, best_aic = None, np.inf
    for cfg in configs:
        pred, aic = _safe_fit_forecast(log_s, cfg, periods=periods)
        if pred is None:
            continue
        if aic < best_aic:
            best_aic = aic
            best_pred = pred

    if best_pred is not None and np.isfinite(best_pred) and best_pred > 0:
        return float(best_pred)

    # 4) Robust fallback.
    if n >= 12:
        return float(0.5 * (s[-1] + s[-12]))
    if n >= 3:
        return float(np.median(s[-3:]))
    return float(s[-1])


def compute_ets_features(df: pd.DataFrame, target: str = "rto",
                          n_jobs: int = -1, min_t_for_prediction: int = 6) -> pd.DataFrame:
    """Для каждой пары (store_id, t) считаем ETS-прогноз по данным store_id с t' < t."""
    df = df.sort_values(["store_id", "t"]).copy()

    def _process_store(sid, sub):
        sub = sub.sort_values("t")
        ys = sub[target].values
        ts = sub["t"].values
        preds = []
        for i, _ in enumerate(ts):
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