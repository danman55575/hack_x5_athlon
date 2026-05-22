"""Безутечковая фичегенерация. Все лаги/роллинги/тренды со сдвигом shift(1).
Target encoding делается отдельно out-of-fold в pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .loader import CAT_COLS, DYNAMIC_COLS


def add_lag_features(df, target="rto", lags=(1, 2, 3, 6, 7, 9, 10, 12, 13, 15, 16, 24, 25),
                     group="store_id"):
    # Assume df is already sorted by (group, t) and copied
    g = df.groupby(group)[target]
    for L in lags:
        df[f"{target}_lag_{L}"] = g.shift(L)
    return df


def add_rolling_features(df, target="rto", windows=(3, 6, 12, 24), group="store_id"):
    # Assume df is already sorted by (group, t) and copied
    shifted = df.groupby(group)[target].shift(1)
    for w in windows:
        roll = shifted.groupby(df[group]).rolling(window=w, min_periods=max(2, w // 3))
        df[f"{target}_rmean_{w}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{target}_rstd_{w}"] = roll.std().reset_index(level=0, drop=True)
        df[f"{target}_rmin_{w}"] = roll.min().reset_index(level=0, drop=True)
        df[f"{target}_rmax_{w}"] = roll.max().reset_index(level=0, drop=True)
        df[f"{target}_rmedian_{w}"] = roll.median().reset_index(level=0, drop=True)
    return df


def add_diff_features(df, target="rto", group="store_id"):
    # Assume df is already sorted by (group, t) and copied
    g = df.groupby(group)[target]
    lag1, lag2, lag3, lag6, lag12 = g.shift(1), g.shift(2), g.shift(3), g.shift(6), g.shift(12)
    df[f"{target}_diff_1"] = lag1 - lag2
    df[f"{target}_diff_2"] = lag2 - lag3
    df[f"{target}_pct_1"] = (lag1 - lag2) / lag2.replace(0, np.nan)
    df[f"{target}_pct_2"] = (lag2 - lag3) / lag3.replace(0, np.nan)
    df[f"{target}_diff_6"] = lag1 - lag6
    df[f"{target}_pct_6"] = (lag1 - lag6) / lag6.replace(0, np.nan)
    df[f"{target}_yoy_diff"] = lag1 - lag12
    df[f"{target}_yoy_ratio"] = lag1 / lag12.replace(0, np.nan)
    df[f"{target}_log_ratio_1_12"] = np.log1p(lag1) - np.log1p(lag12)
    return df


def add_seasonal_features(df, target="rto", group="store_id"):
    # Assume df is already sorted by (group, t) and copied
    g = df.groupby(group)[target]
    df[f"{target}_same_month_1y"] = g.shift(12)
    df[f"{target}_same_month_2y"] = g.shift(24)
    df[f"{target}_same_month_mean"] = df[[f"{target}_same_month_1y",
                                           f"{target}_same_month_2y"]].mean(axis=1)
    lag1 = g.shift(1)
    lag12 = g.shift(12)
    lag24 = g.shift(24)
    season_trend = lag12 / lag24.replace(0, np.nan)
    df[f"{target}_naive_seasonal"] = df[f"{target}_same_month_1y"] * season_trend
    df[f"{target}_ratio_lag1_sm1y"] = lag1 / df[f"{target}_same_month_1y"].replace(0, np.nan)
    return df


def add_trend_features(df, target="rto", group="store_id"):
    # Assume df is already sorted by (group, t) and copied
    shifted = df.groupby(group)[target].shift(1)

    def _slope_numba(arr):
        """Compute linear regression slope - optimized for small windows."""
        a = np.asarray(arr, dtype=np.float64)
        valid = ~np.isnan(a)
        a = a[valid]
        n = len(a)
        if n < 3:
            return np.nan
        # Fast numpy computation without polyfit overhead
        x = np.arange(n, dtype=np.float64)
        x_mean = x.mean()
        y_mean = a.mean()
        numerator = ((x - x_mean) * (a - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        return numerator / denominator if denominator != 0 else np.nan

    for w in (3, 6, 12):
        # Use rolling apply but with optimized slope function
        s = (shifted.groupby(df[group])
             .rolling(window=w, min_periods=3)
             .apply(_slope_numba, raw=False)
             .reset_index(level=0, drop=True))
        df[f"{target}_slope_{w}"] = s
    
    return df


def add_expanding_stats(df, target="rto", group="store_id"):
    # Assume df is already sorted by (group, t) and copied
    shifted = df.groupby(group)[target].shift(1)
    grp = shifted.groupby(df[group]).expanding()
    df[f"{target}_cummean"] = grp.mean().reset_index(level=0, drop=True)
    df[f"{target}_cumstd"]  = grp.std().reset_index(level=0, drop=True)
    df[f"{target}_cummax"]  = grp.max().reset_index(level=0, drop=True)
    df[f"{target}_cummin"]  = grp.min().reset_index(level=0, drop=True)
    df[f"{target}_lag1_to_cummean"] = (df.groupby(group)[target].shift(1) /
                                       df[f"{target}_cummean"].replace(0, np.nan))
    return df


def add_dynamic_lags(df, group="store_id"):
    # Assume df is already sorted by (group, t) and copied
    for col in DYNAMIC_COLS:
        if col not in df.columns:
            continue
        g = df.groupby(group)[col]
        df[f"{col}_lag_1"]  = g.shift(1)
        df[f"{col}_lag_2"]  = g.shift(2)
        df[f"{col}_lag_12"] = g.shift(12)
        df[f"{col}_rmean_3"] = (g.shift(1).groupby(df[group])
                                .rolling(3, min_periods=1).mean()
                                .reset_index(level=0, drop=True))
        df[f"{col}_rmean_12"] = (g.shift(1).groupby(df[group])
                                 .rolling(12, min_periods=3).mean()
                                 .reset_index(level=0, drop=True))
    return df


def add_calendar_features(df):
    # Assume df is already sorted and copied
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["is_jan"]    = (df["month"] == 1).astype(np.int8)
    df["is_feb"]    = (df["month"] == 2).astype(np.int8)
    df["is_mar"]    = (df["month"] == 3).astype(np.int8)
    df["is_dec"]    = (df["month"] == 12).astype(np.int8)
    df["quarter"]   = ((df["month"] - 1) // 3 + 1).astype(np.int8)
    return df


def add_group_aggregations(df: pd.DataFrame, target: str = "rto"):
    """Глобальные signals: средний РТО / средний YoY по region / area_cat / open_date_cat.
    Всё считается по shifted (только прошлое) с помощью expanding (без утечек).
    """
    df = df.copy()
    original_index = df.index

    # Лаги
    df["_rto_lag1"] = df.groupby("store_id")[target].shift(1)
    df["_rto_lag12"] = df.groupby("store_id")[target].shift(12)
    df["_rto_lag13"] = df.groupby("store_id")[target].shift(13)

    # Сортируем по времени для expanding
    df = df.sort_values("t").reset_index(drop=True)

    for key in ["region", "area_cat", "open_date_cat"]:
        if key not in df.columns:
            continue

        # Expanding mean для lag1
        df[f"grp_{key}_lag1_mean"] = df.groupby(key)["_rto_lag1"].expanding().mean().shift(1).reset_index(level=0, drop=True)
        # Expanding median для lag1
        df[f"grp_{key}_lag1_median"] = df.groupby(key)["_rto_lag1"].expanding().quantile(0.5).shift(1).reset_index(level=0, drop=True)

        # YoY: _rto_lag1 / _rto_lag12
        yoy = df["_rto_lag1"] / df["_rto_lag12"].replace(0, np.nan)
        df[f"grp_{key}_yoy_mean"] = yoy.groupby(df[key]).expanding().mean().shift(1).reset_index(level=0, drop=True)
        df[f"grp_{key}_yoy_median"] = yoy.groupby(df[key]).expanding().quantile(0.5).shift(1).reset_index(level=0, drop=True)

    # ----- Глобальные макро-сигналы (все магазины) -----
    df["grp_all_lag1_mean"] = df["_rto_lag1"].expanding().mean().shift(1)
    df["grp_all_lag12_mean"] = df["_rto_lag12"].expanding().mean().shift(1)

    yoy_macro = df["_rto_lag1"] / df["_rto_lag13"].replace(0, np.nan)
    df["grp_all_yoy_macro_mean"] = yoy_macro.expanding().mean().shift(1)
    df["grp_all_yoy_macro_median"] = yoy_macro.expanding().quantile(0.5).shift(1)

    # Удаляем временные колонки
    df = df.drop(columns=["_rto_lag1", "_rto_lag12", "_rto_lag13"])

    # Восстанавливаем исходный порядок строк
    df = df.set_index(original_index)
    return df


def encode_categoricals_ordinal(df, cat_cols=None):
    # df is already copied in build_features, no need to copy again
    mappings = {}
    if cat_cols is None:
        cat_cols=CAT_COLS
        
    for c in cat_cols:
        if c not in df.columns:
            continue
        df[c] = df[c].astype("category")
        mappings[c] = list(df[c].cat.categories)
        df[c] = df[c].cat.codes.astype(np.int32)
    return df, mappings


def downcast(df):
    # Assume df is already copied if needed
    for c in df.select_dtypes(include="float64").columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include="int64").columns:
        if c == "store_id":
            df[c] = df[c].astype(np.int32)
        else:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df



def build_features(df: pd.DataFrame, target: str = "rto"):
    # Sort and copy ONCE at the start, then reuse for all feature functions
    df = df.sort_values(["store_id", "t"]).copy()
    
    df = add_lag_features(df, target=target)
    df = add_rolling_features(df, target=target)
    df = add_diff_features(df, target=target)
    df = add_seasonal_features(df, target=target)
    df = add_trend_features(df, target=target)
    df = add_expanding_stats(df, target=target)
    df = add_dynamic_lags(df)
    df = add_calendar_features(df)
    df = add_group_aggregations(df, target=target)
    df, _ = encode_categoricals_ordinal(df)
    df = downcast(df)

    # 't' исключаем из фичей. Сезонность кодируется month_sin/cos + лагами.
    # Иначе при предсказании марта 2025 (t=26) деревья экстраполируют в неизвестность —
    # сплиты типа "t > 24" уводят в опасные регионы пространства.
    drop_cols = {target, "store_id", "year", "t", "month"} | set(DYNAMIC_COLS)
    feature_cols = [c for c in df.columns if c not in drop_cols]
    cat_features = [c for c in CAT_COLS if c in feature_cols]
    return df, feature_cols, cat_features
