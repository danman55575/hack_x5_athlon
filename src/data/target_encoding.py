"""KFold target encoding по time-aware схеме: для каждой строки кодируем категорию
средним таргета по строкам с t < текущего t (expanding mean) — без утечки."""
from __future__ import annotations
import numpy as np
import pandas as pd


def expanding_target_encode(df: pd.DataFrame, cat_col: str, target: str = "rto",
                            min_samples: int = 30, smoothing: float = 50.0,
                            time_col: str = "t") -> np.ndarray:
    """Для каждой строки возвращает (среднее по cat_col по записям с t' < t),
    сглаженное к глобальному среднему. Полностью без утечки.

    ВАЖНО: возвращаемый массив строго соответствует ИСХОДНОМУ порядку строк df,
    переданному вызывающим кодом.
    """
    # 1. Сохраняем исходный порядок ДО любых сортировок.
    df = df.copy()
    df["_orig_idx"] = np.arange(len(df), dtype=np.int64)

    # 2. Глобальное среднее (используется для smoothing и для fallback).
    global_mean = float(df[target].mean())

    # 3. Делаем sum/count по (cat, t), затем cumsum по cat, ПРЕДВАРИТЕЛЬНО
    # отсортировав по времени, чтобы cumsum шёл хронологически.
    grouped = (df.groupby([cat_col, time_col], dropna=False)[target]
                 .agg(["sum", "count"])
                 .reset_index()
                 .sort_values([cat_col, time_col]))
    grouped["sum"] = grouped["sum"].fillna(0.0)
    grouped["count"] = grouped["count"].fillna(0).astype(np.int64)
    grouped["cum_sum"] = grouped.groupby(cat_col)["sum"].cumsum() - grouped["sum"]
    grouped["cum_cnt"] = grouped.groupby(cat_col)["count"].cumsum() - grouped["count"]
    grouped["te"] = (grouped["cum_sum"] + smoothing * global_mean) / (grouped["cum_cnt"] + smoothing)
    grouped.loc[grouped["cum_cnt"] < min_samples, "te"] = global_mean

    # 4. Merge и восстановление исходного порядка.
    out = df.merge(grouped[[cat_col, time_col, "te"]], on=[cat_col, time_col], how="left")
    out = out.sort_values("_orig_idx", kind="stable")
    te = out["te"].astype(np.float32).values
    # Если по какой-то причине осталась NaN (например в (cat,t), которой нет в grouped) — глобальное среднее
    te = np.where(np.isnan(te), np.float32(global_mean), te)
    return te


def add_target_encodings(df: pd.DataFrame, target: str = "rto",
                         cat_cols=("region", "locality", "open_date_cat", "area_cat")) -> pd.DataFrame:
    """Добавляет TE-фичи. Только row-level, time-aware. Train + test строки покрыты."""
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[f"te_{col}"] = expanding_target_encode(df, col, target=target)
    if {"region", "month"}.issubset(df.columns):
        df["_region_month"] = df["region"].astype(str) + "_" + df["month"].astype(str)
        df["te_region_month"] = expanding_target_encode(df, "_region_month", target=target)
        df = df.drop(columns=["_region_month"])
    if {"area_cat", "open_date_cat"}.issubset(df.columns):
        df["_area_open"] = df["area_cat"].astype(str) + "_" + df["open_date_cat"].astype(str)
        df["te_area_open"] = expanding_target_encode(df, "_area_open", target=target)
        df = df.drop(columns=["_area_open"])
    # Дополнительный полезный сигнал: TE по region+area_cat
    if {"region", "area_cat"}.issubset(df.columns):
        df["_region_area"] = df["region"].astype(str) + "_" + df["area_cat"].astype(str)
        df["te_region_area"] = expanding_target_encode(df, "_region_area", target=target)
        df = df.drop(columns=["_region_area"])
    return df
