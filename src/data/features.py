"""Feature engineering. Все фичи — без утечек таргета: лаги/роллинги считаются по shift(1)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .loader import CAT_COLS, STORE_STATIC_COLS, DYNAMIC_COLS


def add_lag_features(df: pd.DataFrame, target: str = "rto",
                     lags=(1, 2, 3, 6, 12), group="store_id") -> pd.DataFrame:
    df = df.sort_values([group, "t"]).copy()
    g = df.groupby(group)[target]
    for L in lags:
        df[f"{target}_lag_{L}"] = g.shift(L)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "rto",
                         windows=(3, 6, 12), group="store_id") -> pd.DataFrame:
    df = df.sort_values([group, "t"]).copy()
    shifted = df.groupby(group)[target].shift(1)
    for w in windows:
        df[f"{target}_rmean_{w}"] = (shifted.groupby(df[group])
                                     .rolling(window=w, min_periods=max(1, w // 2))
                                     .mean().reset_index(level=0, drop=True))
        df[f"{target}_rstd_{w}"] = (shifted.groupby(df[group])
                                    .rolling(window=w, min_periods=max(2, w // 2))
                                    .std().reset_index(level=0, drop=True))
        df[f"{target}_rmin_{w}"] = (shifted.groupby(df[group])
                                    .rolling(window=w, min_periods=max(1, w // 2))
                                    .min().reset_index(level=0, drop=True))
        df[f"{target}_rmax_{w}"] = (shifted.groupby(df[group])
                                    .rolling(window=w, min_periods=max(1, w // 2))
                                    .max().reset_index(level=0, drop=True))
    return df


def add_diff_features(df: pd.DataFrame, target: str = "rto", group="store_id") -> pd.DataFrame:
    df = df.sort_values([group, "t"]).copy()
    g = df.groupby(group)[target]
    lag1 = g.shift(1); lag2 = g.shift(2); lag3 = g.shift(3); lag12 = g.shift(12)
    df[f"{target}_diff_1"] = lag1 - lag2
    df[f"{target}_diff_2"] = lag2 - lag3
    df[f"{target}_pct_1"] = (lag1 - lag2) / lag2.replace(0, np.nan)
    # year-over-year
    df[f"{target}_yoy_diff"] = lag1 - lag12
    df[f"{target}_yoy_ratio"] = lag1 / lag12.replace(0, np.nan)
    return df


def add_same_month_history(df: pd.DataFrame, target: str = "rto", group="store_id") -> pd.DataFrame:
    """Сезонные лаги: РТО того же месяца годом ранее (и за 2 года, если есть)."""
    df = df.sort_values([group, "t"]).copy()
    # rto тот же месяц год назад (lag 12) и два года назад (lag 24)
    g = df.groupby(group)[target]
    df[f"{target}_same_month_1y"] = g.shift(12)
    df[f"{target}_same_month_2y"] = g.shift(24)
    df[f"{target}_same_month_mean"] = df[[f"{target}_same_month_1y", f"{target}_same_month_2y"]].mean(axis=1)
    # Сравнение с тем же месяцем
    lag1 = g.shift(1)
    df[f"{target}_ratio_lag1_to_same_month"] = lag1 / df[f"{target}_same_month_1y"].replace(0, np.nan)
    return df


def add_dynamic_lags(df: pd.DataFrame, group="store_id") -> pd.DataFrame:
    """Лаги динамических признаков (промо, чеки, отмены, часы)."""
    df = df.sort_values([group, "t"]).copy()
    for col in DYNAMIC_COLS:
        if col not in df.columns:
            continue
        g = df.groupby(group)[col]
        df[f"{col}_lag_1"] = g.shift(1)
        df[f"{col}_lag_12"] = g.shift(12)
        df[f"{col}_rmean_3"] = g.shift(1).groupby(df[group]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_jan"] = (df["month"] == 1).astype(np.int8)
    df["is_mar"] = (df["month"] == 3).astype(np.int8)
    df["is_dec"] = (df["month"] == 12).astype(np.int8)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(np.int8)
    return df


def add_store_stats(df: pd.DataFrame, target: str = "rto", group="store_id") -> pd.DataFrame:
    """Глобальные статистики по магазину, но только по прошлым (через cumulative shift(1))."""
    df = df.sort_values([group, "t"]).copy()
    shifted = df.groupby(group)[target].shift(1)
    df[f"{target}_cummean"] = shifted.groupby(df[group]).expanding().mean().reset_index(level=0, drop=True)
    df[f"{target}_cumstd"]  = shifted.groupby(df[group]).expanding().std().reset_index(level=0, drop=True)
    df[f"{target}_cummax"]  = shifted.groupby(df[group]).expanding().max().reset_index(level=0, drop=True)
    df[f"{target}_cummin"]  = shifted.groupby(df[group]).expanding().min().reset_index(level=0, drop=True)
    return df


def encode_categoricals_ordinal(df: pd.DataFrame, cat_cols=CAT_COLS) -> tuple[pd.DataFrame, dict]:
    """Заменяем категории на целочисленные коды. Возвращаем mapping."""
    df = df.copy()
    mappings = {}
    for c in cat_cols:
        if c not in df.columns:
            continue
        df[c] = df[c].astype("category")
        mappings[c] = list(df[c].cat.categories)
        df[c] = df[c].cat.codes.astype(np.int32)
    return df, mappings


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include="float64").columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include="int64").columns:
        if c == "store_id":
            df[c] = df[c].astype(np.int32)
        else:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def build_features(df: pd.DataFrame, target: str = "rto") -> tuple[pd.DataFrame, list[str], list[str]]:
    """Главный pipeline. Возвращает df, feature_columns, categorical_columns."""
    df = add_lag_features(df, target=target)
    df = add_rolling_features(df, target=target)
    df = add_diff_features(df, target=target)
    df = add_same_month_history(df, target=target)
    df = add_dynamic_lags(df)
    df = add_calendar_features(df)
    df = add_store_stats(df, target=target)
    df, _ = encode_categoricals_ordinal(df)
    df = downcast(df)

    drop_cols = {target, "store_id", "year"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    cat_features = [c for c in CAT_COLS if c in feature_cols]
    return df, feature_cols, cat_features
