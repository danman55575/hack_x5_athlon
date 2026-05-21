"""KFold target encoding по time-aware схеме: для каждой строки кодируем категорию
средним таргета по строкам с t < текущего t (expanding mean) — без утечки."""
from __future__ import annotations
import numpy as np
import pandas as pd


def expanding_target_encode(df: pd.DataFrame, cat_col: str, target: str = "rto",
                            min_samples: int = 30, smoothing: float = 50.0,
                            time_col: str = "t") -> pd.Series:
    """Для каждой строки возвращает (среднее по cat_col по записям с t' < t),
    сглаженное к глобальному среднему. Полностью без утечки."""
    df = df.sort_values([time_col]).copy()
    df["_orig_idx"] = np.arange(len(df))

    global_mean = df[target].mean()
    # сумма и количество по cat_col, накопительно по времени
    # делаем по уникальным t, чтобы избежать leakage внутри одного t
    grouped = df.groupby([cat_col, time_col])[target].agg(["sum", "count"]).reset_index()
    grouped = grouped.sort_values([cat_col, time_col])
    grouped["cum_sum"] = grouped.groupby(cat_col)["sum"].cumsum() - grouped["sum"]
    grouped["cum_cnt"] = grouped.groupby(cat_col)["count"].cumsum() - grouped["count"]
    grouped["te"] = (grouped["cum_sum"] + smoothing * global_mean) / (grouped["cum_cnt"] + smoothing)
    grouped.loc[grouped["cum_cnt"] < min_samples, "te"] = global_mean

    out = df.merge(grouped[[cat_col, time_col, "te"]],
                   on=[cat_col, time_col], how="left")
    out = out.sort_values("_orig_idx")
    return out["te"].astype(np.float32).values


def add_target_encodings(df: pd.DataFrame, target: str = "rto",
                         cat_cols=("region", "locality", "open_date_cat", "area_cat")) -> pd.DataFrame:
    """Добавляет TE-фичи. Только row-level, time-aware. Train + test строки покрыты."""
    df = df.copy()
    # для записи с NaN таргета (test) expanding encode возьмёт прошлое - это норм
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[f"te_{col}"] = expanding_target_encode(df, col, target=target)
    # парный TE
    if {"region", "month"}.issubset(df.columns):
        df["_region_month"] = df["region"].astype(str) + "_" + df["month"].astype(str)
        df["te_region_month"] = expanding_target_encode(df, "_region_month", target=target)
        df = df.drop(columns=["_region_month"])
    if {"area_cat", "open_date_cat"}.issubset(df.columns):
        df["_area_open"] = df["area_cat"].astype(str) + "_" + df["open_date_cat"].astype(str)
        df["te_area_open"] = expanding_target_encode(df, "_area_open", target=target)
        df = df.drop(columns=["_area_open"])
    return df
