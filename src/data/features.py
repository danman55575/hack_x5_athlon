"""Безутечковая фичегенерация для помесячного прогноза РТО.

Все временные фичи строятся только из прошлого через ``shift(1)`` или более
дальний сдвиг. Target encoding добавляется отдельно в pipeline.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import *


def add_region_spendings(
    df: pd.DataFrame,
    spendings_file: str = "data/processed/region_spendings.csv",
) -> pd.DataFrame:
    """Добавляет региональные расходы в ценах марта 2025 года."""
    spendings_df = pd.read_csv(spendings_file)
    return df.merge(
        spendings_df[["year", "month", "region", "region_spendings_inflated"]],
        on=["year", "month", "region"],
        how="left",
    )


def add_external_macro_features(
    df: pd.DataFrame,
    spendings_file: str = "data/processed/region_spendings.csv",
) -> pd.DataFrame:
    """Adds external macro features from inflation and regional spendings tables.

    The features are designed to remain available for the forecast month even when
    current-month spendings are absent: lagged and rolling statistics are computed
    on a full `(region, year, month)` panel derived from the modeling frame.
    """
    df = df.copy()

    spendings_path = Path(spendings_file)
    if not spendings_path.exists():
        return df

    spend = pd.read_csv(spendings_path)
    spend = spend.sort_values(["region", "year", "month"]).reset_index(drop=True)

    region_panel = (
        df[["year", "month", "t", "region"]]
        .drop_duplicates()
        .sort_values(["region", "t"])
        .reset_index(drop=True)
    )
    region_panel = region_panel.merge(
        spend[["year", "month", "region", "region_spendings_inflated"]],
        on=["year", "month", "region"],
        how="left",
    )

    region_g = region_panel.groupby("region")["region_spendings_inflated"]
    region_panel["region_spendings_lag_1"] = region_g.shift(1)
    region_panel["region_spendings_lag_2"] = region_g.shift(2)
    region_panel["region_spendings_lag_3"] = region_g.shift(3)
    region_panel["region_spendings_lag_12"] = region_g.shift(12)

    shifted_region_spend = region_g.shift(1)
    region_panel["region_spendings_rmean_3"] = (
        shifted_region_spend
        .groupby(region_panel["region"])
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    region_panel["region_spendings_rmean_12"] = (
        shifted_region_spend
        .groupby(region_panel["region"])
        .rolling(window=12, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    region_panel["region_spendings_mom_ratio"] = safe_divide(
        region_panel["region_spendings_lag_1"],
        region_panel["region_spendings_lag_2"],
    )
    region_panel["region_spendings_yoy_ratio"] = safe_divide(
        region_panel["region_spendings_lag_1"],
        region_panel["region_spendings_lag_12"],
    )
    region_panel["region_spendings_lag1_to_rmean3"] = safe_divide(
        region_panel["region_spendings_lag_1"],
        region_panel["region_spendings_rmean_3"],
    )

    macro_cols = [
        "year",
        "month",
        "region",
        "region_spendings_inflated",
        "region_spendings_lag_1",
        "region_spendings_lag_2",
        "region_spendings_lag_3",
        "region_spendings_lag_12",
        "region_spendings_rmean_3",
        "region_spendings_rmean_12",
        "region_spendings_mom_ratio",
        "region_spendings_yoy_ratio",
        "region_spendings_lag1_to_rmean3",
    ]
    return df.merge(region_panel[macro_cols], on=["year", "month", "region"], how="left")


def add_lag_features(
    df: pd.DataFrame,
    target: str = "rto",
    lags: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
    group: str = "store_id",
) -> pd.DataFrame:
    g = df.groupby(group)[target]
    return with_columns(df, {f"{target}_lag_{lag}": g.shift(lag) for lag in lags})


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "rto",
    windows: tuple[int, ...] = (2, 3, 4, 6, 12, 24),
    group: str = "store_id",
) -> pd.DataFrame:
    shifted = df.groupby(group)[target].shift(1)
    new_cols: dict[str, pd.Series | np.ndarray] = {}
    for window in windows:
        roll = shifted.groupby(df[group]).rolling(window=window, min_periods=max(1, window // 3))
        mean_col = f"{target}_rmean_{window}"
        std_col = f"{target}_rstd_{window}"
        min_col = f"{target}_rmin_{window}"
        max_col = f"{target}_rmax_{window}"
        median_col = f"{target}_rmedian_{window}"

        new_cols[mean_col] = roll.mean().reset_index(level=0, drop=True)
        new_cols[std_col] = roll.std().reset_index(level=0, drop=True)
        new_cols[min_col] = roll.min().reset_index(level=0, drop=True)
        new_cols[max_col] = roll.max().reset_index(level=0, drop=True)
        new_cols[median_col] = roll.median().reset_index(level=0, drop=True)

    for window in (2, 3, 4, 6, 12):
        rmean_col = f"{target}_rmean_{window}"
        if rmean_col in new_cols:
            new_cols[f"{target}_mean_{window}"] = new_cols[rmean_col]
    for window in (3, 6):
        rmedian_col = f"{target}_rmedian_{window}"
        rstd_col = f"{target}_rstd_{window}"
        if rmedian_col in new_cols:
            new_cols[f"{target}_median_{window}"] = new_cols[rmedian_col]
        if rstd_col in new_cols:
            new_cols[f"{target}_std_{window}"] = new_cols[rstd_col]
    if f"{target}_rmin_6" in new_cols:
        new_cols[f"{target}_min_6"] = new_cols[f"{target}_rmin_6"]
    if f"{target}_rmax_6" in new_cols:
        new_cols[f"{target}_max_6"] = new_cols[f"{target}_rmax_6"]
    if f"{target}_mean_6" in new_cols and f"{target}_std_6" in new_cols:
        new_cols[f"{target}_cv_6"] = safe_divide(new_cols[f"{target}_std_6"], new_cols[f"{target}_mean_6"])
    return with_columns(df, new_cols)


def add_diff_features(df: pd.DataFrame, target: str = "rto", group: str = "store_id") -> pd.DataFrame:
    g = df.groupby(group)[target]
    lag1 = g.shift(1)
    lag2 = g.shift(2)
    lag3 = g.shift(3)
    lag6 = g.shift(6)
    lag12 = g.shift(12)

    return with_columns(
        df,
        {
            f"{target}_diff_1": lag1 - lag2,
            f"{target}_diff_2": lag2 - lag3,
            f"{target}_pct_1": safe_divide(lag1 - lag2, lag2),
            f"{target}_pct_2": safe_divide(lag2 - lag3, lag3),
            f"{target}_diff_6": lag1 - lag6,
            f"{target}_pct_6": safe_divide(lag1 - lag6, lag6),
            f"{target}_yoy_diff": lag1 - lag12,
            f"{target}_yoy_ratio": safe_divide(lag1, lag12),
            f"{target}_log_ratio_1_12": np.log1p(lag1) - np.log1p(lag12),
        },
    )


def add_growth_ratio_features(
    df: pd.DataFrame,
    target: str = "rto",
    group: str = "store_id",
) -> pd.DataFrame:
    g = df.groupby(group)[target]
    lag1 = g.shift(1)
    lag2 = g.shift(2)
    lag3 = g.shift(3)
    lag4 = g.shift(4)
    lag5 = g.shift(5)
    lag6 = g.shift(6)
    lag7 = g.shift(7)
    lag12 = g.shift(12)
    lag13 = g.shift(13)
    lag14 = g.shift(14)

    new_cols: dict[str, pd.Series | np.ndarray] = {
        "log_growth_lag_1": np.log(safe_divide(lag1, lag2)),
        "log_growth_lag_2": np.log(safe_divide(lag2, lag3)),
        "log_growth_lag_4": np.log(safe_divide(lag4, lag5)),
        "log_growth_lag_6": np.log(safe_divide(lag6, lag7)),
        f"{target}_lag_1_div_lag_2": safe_divide(lag1, lag2),
        f"{target}_lag_2_div_lag_4": safe_divide(lag2, lag4),
        f"{target}_lag_12_div_lag_13": safe_divide(lag12, lag13),
        f"{target}_lag_1_div_lag_12": safe_divide(lag1, lag12),
        f"{target}_lag_2_div_lag_14": safe_divide(lag2, lag14),
    }
    new_cols["abs_log_growth_lag_1"] = pd.Series(new_cols["log_growth_lag_1"], copy=False).abs()
    if f"{target}_mean_3" in df.columns:
        new_cols[f"{target}_lag_1_div_mean_3"] = safe_divide(lag1, df[f"{target}_mean_3"])
    if f"{target}_mean_3" in df.columns and f"{target}_mean_12" in df.columns:
        new_cols[f"{target}_mean_3_div_mean_12"] = safe_divide(
            df[f"{target}_mean_3"],
            df[f"{target}_mean_12"],
        )
    new_cols[f"{target}_seasonal_ratio_baseline"] = (
        lag1 * pd.Series(new_cols[f"{target}_lag_12_div_lag_13"], copy=False)
    )
    return with_columns(df, new_cols)


def add_seasonal_features(df: pd.DataFrame, target: str = "rto", group: str = "store_id") -> pd.DataFrame:
    g = df.groupby(group)[target]
    same_month_1y = g.shift(12)
    same_month_2y = g.shift(24)
    lag12 = g.shift(12)
    lag24 = g.shift(24)
    season_trend = safe_divide(lag12, lag24)
    return with_columns(
        df,
        {
            f"{target}_same_month_1y": same_month_1y,
            f"{target}_same_month_2y": same_month_2y,
            f"{target}_same_month_mean": pd.concat([same_month_1y, same_month_2y], axis=1).mean(axis=1),
            f"{target}_naive_seasonal": same_month_1y * season_trend,
        },
    )


def add_trend_features(df: pd.DataFrame, target: str = "rto", group: str = "store_id") -> pd.DataFrame:
    shifted = df.groupby(group)[target].shift(1)

    def _rolling_slope(arr: pd.Series) -> float:
        values = np.asarray(arr, dtype=np.float64)
        valid = np.isfinite(values)
        values = values[valid]
        if len(values) < 3:
            return np.nan
        x = np.arange(len(values), dtype=np.float64)
        x_centered = x - x.mean()
        y_centered = values - values.mean()
        denom = np.square(x_centered).sum()
        if denom == 0:
            return np.nan
        return float((x_centered * y_centered).sum() / denom)

    new_cols: dict[str, pd.Series | np.ndarray] = {}
    for window in (3, 6, 12):
        new_cols[f"{target}_slope_{window}"] = (
            shifted.groupby(df[group])
            .rolling(window=window, min_periods=3)
            .apply(_rolling_slope, raw=False)
            .reset_index(level=0, drop=True)
        )
    return with_columns(df, new_cols)


def add_expanding_stats(df: pd.DataFrame, target: str = "rto", group: str = "store_id") -> pd.DataFrame:
    shifted = df.groupby(group)[target].shift(1)
    expanding = shifted.groupby(df[group]).expanding()
    cummean = expanding.mean().reset_index(level=0, drop=True)
    return with_columns(
        df,
        {
            f"{target}_cummean": cummean,
            f"{target}_cumstd": expanding.std().reset_index(level=0, drop=True),
            f"{target}_cummax": expanding.max().reset_index(level=0, drop=True),
            f"{target}_cummin": expanding.min().reset_index(level=0, drop=True),
            f"{target}_lag1_to_cummean": safe_divide(df.groupby(group)[target].shift(1), cummean),
        },
    )


def add_dynamic_lags(df: pd.DataFrame, group: str = "store_id") -> pd.DataFrame:
    new_cols: dict[str, pd.Series | np.ndarray] = {}
    for col in DYNAMIC_COLS:
        if col not in df.columns:
            continue
        g = df.groupby(group)[col]
        shifted = g.shift(1)
        new_cols[f"{col}_lag_1"] = shifted
        new_cols[f"{col}_lag_2"] = g.shift(2)
        new_cols[f"{col}_lag_12"] = g.shift(12)
        new_cols[f"{col}_rmean_3"] = (
            shifted.groupby(df[group]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        new_cols[f"{col}_rmean_12"] = (
            shifted.groupby(df[group]).rolling(12, min_periods=3).mean().reset_index(level=0, drop=True)
        )
    return with_columns(df, new_cols)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    days_in_month = days_in_month_vec(df["year"].values, df["month"].values)
    non_working_vec = np.array([
        non_working_days.get((year, month), 0)
        for year, month in zip(df["year"].values, df["month"].values)
    ])

    return with_columns(
        df,
        {
            "month_sin": np.sin(2 * np.pi * df["month"] / 12).astype(np.float32),
            "month_cos": np.cos(2 * np.pi * df["month"] / 12).astype(np.float32),
            "is_jan": (df["month"] == 1).astype(np.int8),
            "is_feb": (df["month"] == 2).astype(np.int8),
            "is_dec": (df["month"] == 12).astype(np.int8),
            "quarter": ((df["month"] - 1) // 3 + 1).astype(np.int8),
            "non_working_days": non_working_vec.astype(np.int8),
            "days_in_month": days_in_month.astype(np.int8),
        },
    )


def add_days_features(df: pd.DataFrame, target: str = "rto", group: str = "store_id") -> pd.DataFrame:
    """Нормирует ключевые лаги на длину месяца."""
    if "days_in_month" not in df.columns:
        return df

    g_dim = df.groupby(group)["days_in_month"]
    days_in_month = df["days_in_month"].astype(np.float32)
    days_in_month_lag_1 = g_dim.shift(1).astype(np.float32)
    days_in_month_lag_12 = g_dim.shift(12).astype(np.float32)
    new_cols: dict[str, pd.Series | np.ndarray] = {
        "days_in_month_lag_1": days_in_month_lag_1,
        "days_in_month_lag_12": days_in_month_lag_12,
        "days_ratio_curr_to_lag1": safe_divide(days_in_month, days_in_month_lag_1).astype(np.float32),
        "days_ratio_curr_to_lag12": safe_divide(days_in_month, days_in_month_lag_12).astype(np.float32),
    }
    if f"{target}_lag_1" in df.columns:
        new_cols[f"{target}_lag_1_per_day"] = safe_divide(
            df[f"{target}_lag_1"],
            days_in_month_lag_1,
        ).astype(np.float32)
        new_cols[f"{target}_lag_1_scaled_by_days"] = (
            df[f"{target}_lag_1"] * pd.Series(new_cols["days_ratio_curr_to_lag1"], copy=False)
        ).astype(np.float32)
    if f"{target}_lag_12" in df.columns:
        new_cols[f"{target}_lag_12_per_day"] = safe_divide(
            df[f"{target}_lag_12"],
            days_in_month_lag_12,
        ).astype(np.float32)
    if f"{target}_same_month_1y" in df.columns:
        new_cols[f"{target}_same_month_1y_per_day"] = safe_divide(
            df[f"{target}_same_month_1y"],
            days_in_month_lag_12,
        ).astype(np.float32)
    if f"{target}_lag_1_per_day" in new_cols and f"{target}_lag_12_per_day" in new_cols:
        new_cols[f"{target}_per_day_yoy_ratio"] = safe_divide(
            new_cols[f"{target}_lag_1_per_day"],
            new_cols[f"{target}_lag_12_per_day"],
        ).astype(np.float32)
    return with_columns(df, new_cols)


def add_store_state_features(df: pd.DataFrame, group: str = "store_id") -> pd.DataFrame:
    """Добавляет признаки доступной глубины истории и недавних скачков без утечки."""
    months_with_history = df.groupby(group).cumcount() + 1
    log_growth = pd.Series(df.get("log_growth_lag_1"), index=df.index, copy=False)

    return with_columns(
        df,
        {
            "months_with_history": months_with_history.astype(np.int16),
            "is_new_store": (months_with_history <= 3).astype(np.int8),
            "is_short_history_store": (months_with_history <= 12).astype(np.int8),
            "is_recent_jump_up": (log_growth > np.log(1.12)).astype(np.int8),
            "is_recent_jump_down": (log_growth < np.log(0.88)).astype(np.int8),
        },
    )


def _add_group_month_aggregate(
    df: pd.DataFrame,
    value: pd.Series,
    group_cols: list[str],
    feature_name: str,
    stat: str = "mean",
) -> pd.DataFrame:
    if any(col not in df.columns for col in group_cols):
        return df

    temp = df[group_cols + ["t"]].copy()
    temp["_value"] = value.values
    agg = (
        temp.groupby(group_cols + ["t"], dropna=False)["_value"]
        .agg(stat)
        .reset_index(name=feature_name)
    )
    return df.merge(agg, on=group_cols + ["t"], how="left")


def add_group_aggregations(df: pd.DataFrame, target: str = "rto") -> pd.DataFrame:
    """Кросс-магазинные агрегаты только из уже известных лагов."""
    df = df.copy().reset_index(drop=True)
    df["_orig_pos"] = np.arange(len(df), dtype=np.int64)
    df["_rto_lag1"] = df.groupby("store_id")[target].shift(1)
    df["_rto_lag12"] = df.groupby("store_id")[target].shift(12)
    df["_rto_lag13"] = df.groupby("store_id")[target].shift(13)
    df = df.sort_values("t", kind="mergesort").reset_index(drop=True)

    yoy = safe_divide(df["_rto_lag1"], df["_rto_lag12"])
    for key in ["region", "locality", "area_cat", "open_date_cat"]:
        if key not in df.columns:
            continue
        df[f"grp_{key}_lag1_mean"] = expanding_mean_shifted(df["_rto_lag1"], df[key])
        df[f"grp_{key}_lag1_median"] = expanding_quantile_shifted(df["_rto_lag1"], df[key], 0.5)
        df[f"grp_{key}_yoy_mean"] = expanding_mean_shifted(yoy, df[key])
        df[f"grp_{key}_yoy_median"] = expanding_quantile_shifted(yoy, df[key], 0.5)

    df["grp_all_lag1_mean"] = df.groupby("t")["_rto_lag1"].transform("mean")
    df["grp_all_lag12_mean"] = df.groupby("t")["_rto_lag12"].transform("mean")
    yoy_macro = safe_divide(df["_rto_lag1"], df["_rto_lag13"])
    df["grp_all_yoy_macro_mean"] = yoy_macro.groupby(df["t"]).transform("mean")
    df["grp_all_yoy_macro_median"] = yoy_macro.groupby(df["t"]).transform("median")

    df = df.drop(columns=["_rto_lag1", "_rto_lag12", "_rto_lag13"])
    df = df.sort_values("_orig_pos", kind="mergesort").reset_index(drop=True)
    return df.drop(columns=["_orig_pos"])


def add_group_seasonality_features(df: pd.DataFrame, target: str = "rto") -> pd.DataFrame:
    df = add_prev_year_group_mean(df, target, ["region"], "region_month_rto_mean_lag12")
    df = compute_historical_march_feb_ratio(df, target, [], "global")
    for prefix, group_cols in (
        ("region", ["region"]),
        ("locality", ["locality"]),
    ):
        df = compute_historical_march_feb_ratio(df, target, group_cols, prefix)

    df["region_march_feb_ratio"] = df["region_march_feb_ratio"].fillna(df["global_march_feb_ratio"])

    use_locality = df["locality_march_feb_ratio_hist_pairs"] >= 50
    df["city_march_feb_ratio"] = np.where(
        use_locality,
        df["locality_march_feb_ratio"],
        df["region_march_feb_ratio"],
    )
    df["city_march_feb_ratio"] = pd.Series(df["city_march_feb_ratio"]).fillna(df["global_march_feb_ratio"])
    return df


def encode_categoricals_ordinal(
    df: pd.DataFrame,
    cat_cols: list[str] | tuple[str, ...] = CAT_COLS,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    mappings: dict[str, list[str]] = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype("category")
        mappings[col] = list(df[col].cat.categories)
        df[col] = df[col].cat.codes.astype(np.int32)
    return df, mappings


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include="int64").columns:
        df[col] = df[col].astype(np.int32 if col == "store_id" else pd.to_numeric(df[col], downcast="integer").dtype)
    return df


def get_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """Разбивает список фичей на логические группы для абляций."""
    groups = {
        "external_macro": [],
        "ets": [],
        "target_encoding": [],
        "static_calendar": [],
        "basic_lags": [],
        "rolling": [],
        "growth_ratio": [],
        "seasonality": [],
        "group_aggregates": [],
        "anomaly_residual": [],
        "dynamic_covariates": [],
        "other": [],
    }

    basic_lag_re = re.compile(r"^rto_lag_(1|2|3|4|6|12|13|14)$")
    rolling_names = {
        "rto_mean_2",
        "rto_mean_3",
        "rto_mean_4",
        "rto_mean_6",
        "rto_mean_12",
        "rto_median_3",
        "rto_median_6",
        "rto_std_3",
        "rto_std_6",
        "rto_min_6",
        "rto_max_6",
        "rto_cv_6",
    }
    growth_names = {
        "rto_diff_1",
        "rto_diff_2",
        "rto_diff_6",
        "rto_pct_1",
        "rto_pct_2",
        "rto_pct_6",
        "rto_yoy_diff",
        "rto_yoy_ratio",
        "rto_log_ratio_1_12",
        "log_growth_lag_1",
        "log_growth_lag_2",
        "log_growth_lag_4",
        "log_growth_lag_6",
        "rto_lag_1_div_lag_2",
        "rto_lag_2_div_lag_4",
        "rto_lag_1_div_mean_3",
        "rto_mean_3_div_mean_12",
    }
    seasonal_names = {
        "rto_same_month_1y",
        "rto_same_month_2y",
        "rto_same_month_mean",
        "rto_naive_seasonal",
        "rto_same_month_1y_per_day",
        "rto_ratio_lag1_sm1y",
        "rto_lag_12_div_lag_13",
        "rto_lag_1_div_lag_12",
        "rto_lag_2_div_lag_14",
        "rto_seasonal_ratio_baseline",
        "rto_lag_12_per_day",
        "rto_per_day_yoy_ratio",
        "global_march_feb_ratio",
        "region_march_feb_ratio",
        "city_march_feb_ratio",
    }

    for col in feature_cols:
        if col.startswith(EXTERNAL_MACRO_PREFIXES):
            groups["external_macro"].append(col)
        elif col.startswith("ets_"):
            groups["ets"].append(col)
        elif col.startswith("te_"):
            groups["target_encoding"].append(col)
        elif col in CAT_COLS or col in STATIC_NUMERIC_COLS or col in CALENDAR_COLS:
            groups["static_calendar"].append(col)
        elif basic_lag_re.match(col):
            groups["basic_lags"].append(col)
        elif col in rolling_names:
            groups["rolling"].append(col)
        elif col in growth_names:
            groups["growth_ratio"].append(col)
        elif col in seasonal_names or "march_feb_ratio_hist_pairs" in col:
            groups["seasonality"].append(col)
        elif col.startswith("grp_") or col.endswith("_month_rto_mean_lag12"):
            groups["group_aggregates"].append(col)
        elif col in ANOMALY_FEATURES:
            groups["anomaly_residual"].append(col)
        elif any(col.startswith(prefix) for prefix in DYNAMIC_COLS):
            groups["dynamic_covariates"].append(col)
        else:
            groups["other"].append(col)

    return groups


def select_feature_columns(feature_cols: list[str], feature_groups: list[str] | None) -> list[str]:
    if not feature_groups or feature_groups == ["all"]:
        return list(feature_cols)

    group_map = get_feature_groups(feature_cols)
    selected: set[str] = set()
    for group_name in feature_groups:
        if group_name == "all":
            selected.update(feature_cols)
            continue
        if group_name not in group_map:
            raise ValueError(f"Неизвестная группа фичей: {group_name}")
        selected.update(group_map[group_name])
    return [col for col in feature_cols if col in selected]


def audit_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, object]:
    feature_df = df[feature_cols]
    nan_counts = feature_df.isna().sum()
    all_nan_cols = nan_counts[nan_counts == len(feature_df)].index.tolist()

    numeric_df = feature_df.select_dtypes(include=[np.number])
    inf_counts = pd.Series(0, index=feature_df.columns, dtype=np.int64)
    if not numeric_df.empty:
        inf_values = np.isinf(numeric_df.to_numpy(dtype=np.float64, copy=True))
        inf_counts.loc[numeric_df.columns] = inf_values.sum(axis=0)
    inf_cols = inf_counts[inf_counts > 0].sort_values(ascending=False)

    return {
        "feature_count": len(feature_cols),
        "rows": int(len(df)),
        "nan_feature_count": int((nan_counts > 0).sum()),
        "total_nan": int(nan_counts.sum()),
        "all_nan_columns": all_nan_cols,
        "inf_columns": inf_cols.index.tolist(),
        "top_nan_columns": nan_counts.sort_values(ascending=False).head(20).to_dict(),
        "top_inf_columns": inf_cols.head(20).to_dict(),
    }


def build_features(df: pd.DataFrame, target: str = "rto") -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df.sort_values(["store_id", "t"]).copy().reset_index(drop=True)
    df = add_external_macro_features(df)
    df = add_lag_features(df, target=target)
    df = add_rolling_features(df, target=target)
    df = add_diff_features(df, target=target)
    df = add_growth_ratio_features(df, target=target)
    df = add_seasonal_features(df, target=target)
    df = add_trend_features(df, target=target)
    df = add_expanding_stats(df, target=target)
    df = df.copy()
    df = add_dynamic_lags(df)
    df = add_calendar_features(df)
    df = add_days_features(df, target=target)
    df = add_group_aggregations(df, target=target)
    df = add_group_seasonality_features(df, target=target)
    df = df.copy()
    df, _ = encode_categoricals_ordinal(df)
    df = downcast(df)

    drop_cols = {target, "store_id", "year", "t", "month"} | set(DYNAMIC_COLS)
    feature_cols = [col for col in df.columns if col not in drop_cols]
    cat_features = [col for col in CAT_COLS if col in feature_cols]
    return df, feature_cols, cat_features


def add_group_aggregations(df: pd.DataFrame, target: str = "rto") -> pd.DataFrame:
    """Кросс-магазинные агрегаты по группе и месяцу только из уже известных лагов."""
    df = df.copy().reset_index(drop=True)
    lag1 = df.groupby("store_id")[target].shift(1)
    lag12 = df.groupby("store_id")[target].shift(12)
    lag13 = df.groupby("store_id")[target].shift(13)
    yoy = safe_divide(lag1, lag12)
    yoy_macro = safe_divide(lag1, lag13)

    group_defs = [
        ("region", ["region"]),
        ("locality", ["locality"]),
        ("area_cat", ["area_cat"]),
        ("open_date_cat", ["open_date_cat"]),
        ("region_area", ["region", "area_cat"]),
    ]
    for prefix, group_cols in group_defs:
        df = _add_group_month_aggregate(df, lag1, group_cols, f"grp_{prefix}_lag1_mean", stat="mean")
        df = _add_group_month_aggregate(df, lag1, group_cols, f"grp_{prefix}_lag1_median", stat="median")
        df = _add_group_month_aggregate(df, yoy, group_cols, f"grp_{prefix}_yoy_mean", stat="mean")
        df = _add_group_month_aggregate(df, yoy, group_cols, f"grp_{prefix}_yoy_median", stat="median")

    df["grp_all_lag1_mean"] = lag1.groupby(df["t"]).transform("mean")
    df["grp_all_lag12_mean"] = lag12.groupby(df["t"]).transform("mean")
    df["grp_all_yoy_macro_mean"] = yoy_macro.groupby(df["t"]).transform("mean")
    df["grp_all_yoy_macro_median"] = yoy_macro.groupby(df["t"]).transform("median")

    alias_map = {
        "region_rto_mean_lag_1": "grp_region_lag1_mean",
        "city_rto_mean_lag_1": "grp_locality_lag1_mean",
        "area_rto_mean_lag_1": "grp_area_cat_lag1_mean",
        "region_area_rto_mean_lag_1": "grp_region_area_lag1_mean",
    }
    for alias, source in alias_map.items():
        if source in df.columns:
            df[alias] = df[source]
    return df


def add_group_seasonality_features(df: pd.DataFrame, target: str = "rto") -> pd.DataFrame:
    df = add_prev_year_group_mean(df, target, ["region"], "region_month_rto_mean_lag12")
    df = add_prev_year_group_mean(df, target, ["region", "area_cat"], "region_area_month_rto_mean_lag12")
    df = compute_historical_march_feb_ratio(df, target, [], "global")
    for prefix, group_cols in (
        ("region", ["region"]),
        ("locality", ["locality"]),
        ("area", ["area_cat"]),
    ):
        df = compute_historical_march_feb_ratio(df, target, group_cols, prefix)

    df["region_march_feb_ratio"] = pd.Series(df["region_march_feb_ratio"], copy=False).fillna(
        df["global_march_feb_ratio"]
    )
    df["area_march_feb_ratio"] = (
        pd.Series(df["area_march_feb_ratio"], copy=False)
        .fillna(df["region_march_feb_ratio"])
        .fillna(df["global_march_feb_ratio"])
    )

    use_locality = df["locality_march_feb_ratio_hist_pairs"] >= 50
    df["city_march_feb_ratio"] = np.where(
        use_locality,
        df["locality_march_feb_ratio"],
        df["region_march_feb_ratio"],
    )
    df["city_march_feb_ratio"] = pd.Series(df["city_march_feb_ratio"]).fillna(df["global_march_feb_ratio"])
    return df


def get_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
    """Разбивает список фичей на логические группы для абляций."""
    groups = {
        "x5_public": [],
        "external_macro": [],
        "ets": [],
        "target_encoding": [],
        "static_calendar": [],
        "basic_lags": [],
        "rolling": [],
        "growth_ratio": [],
        "seasonality": [],
        "group_aggregates": [],
        "anomaly_residual": [],
        "dynamic_covariates": [],
        "other": [],
    }

    basic_lag_re = re.compile(r"^rto_lag_(1|2|3|4|6|12|13|14)$")
    rolling_names = {
        "rto_mean_2",
        "rto_mean_3",
        "rto_mean_4",
        "rto_mean_6",
        "rto_mean_12",
        "rto_median_3",
        "rto_median_6",
        "rto_std_3",
        "rto_std_6",
        "rto_min_6",
        "rto_max_6",
        "rto_cv_6",
    }
    growth_names = {
        "rto_diff_1",
        "rto_diff_2",
        "rto_diff_6",
        "rto_pct_1",
        "rto_pct_2",
        "rto_pct_6",
        "rto_yoy_diff",
        "rto_yoy_ratio",
        "rto_log_ratio_1_12",
        "log_growth_lag_1",
        "log_growth_lag_2",
        "log_growth_lag_4",
        "log_growth_lag_6",
        "rto_lag_1_div_lag_2",
        "rto_lag_2_div_lag_4",
        "rto_lag_1_div_mean_3",
        "rto_mean_3_div_mean_12",
    }
    seasonal_names = {
        "rto_same_month_1y",
        "rto_same_month_2y",
        "rto_same_month_mean",
        "rto_naive_seasonal",
        "rto_same_month_1y_per_day",
        "rto_ratio_lag1_sm1y",
        "rto_lag_12_div_lag_13",
        "rto_lag_1_div_lag_12",
        "rto_lag_2_div_lag_14",
        "rto_seasonal_ratio_baseline",
        "rto_lag_12_per_day",
        "rto_per_day_yoy_ratio",
        "global_march_feb_ratio",
        "region_march_feb_ratio",
        "area_march_feb_ratio",
        "city_march_feb_ratio",
    }

    for col in feature_cols:
        if col.startswith("x5pub_"):
            groups["x5_public"].append(col)
        elif col.startswith(EXTERNAL_MACRO_PREFIXES):
            groups["external_macro"].append(col)
        elif col.startswith("ets_"):
            groups["ets"].append(col)
        elif col.startswith("te_"):
            groups["target_encoding"].append(col)
        elif col in CAT_COLS or col in STATIC_NUMERIC_COLS or col in CALENDAR_COLS:
            groups["static_calendar"].append(col)
        elif basic_lag_re.match(col):
            groups["basic_lags"].append(col)
        elif col in rolling_names:
            groups["rolling"].append(col)
        elif col in growth_names:
            groups["growth_ratio"].append(col)
        elif col in seasonal_names or "march_feb_ratio_hist_pairs" in col:
            groups["seasonality"].append(col)
        elif (
            col.startswith("grp_")
            or col.endswith("_month_rto_mean_lag12")
            or col.endswith("_rto_mean_lag_1")
        ):
            groups["group_aggregates"].append(col)
        elif col in ANOMALY_FEATURES:
            groups["anomaly_residual"].append(col)
        elif any(col.startswith(prefix) for prefix in DYNAMIC_COLS):
            groups["dynamic_covariates"].append(col)
        else:
            groups["other"].append(col)

    return groups


def build_features(df: pd.DataFrame, target: str = "rto") -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df.sort_values(["store_id", "t"]).copy().reset_index(drop=True)
    df = add_external_macro_features(df)
    df = add_lag_features(df, target=target)
    df = add_rolling_features(df, target=target)
    df = add_diff_features(df, target=target)
    df = add_growth_ratio_features(df, target=target)
    df = add_seasonal_features(df, target=target)
    df = add_trend_features(df, target=target)
    df = add_expanding_stats(df, target=target)
    df = df.copy()
    df = add_dynamic_lags(df)
    df = add_calendar_features(df)
    df = add_days_features(df, target=target)
    df = add_store_state_features(df)
    df = add_group_aggregations(df, target=target)
    df = add_group_seasonality_features(df, target=target)
    df = df.copy()
    df, _ = encode_categoricals_ordinal(df)
    df = downcast(df)

    drop_cols = {target, "store_id", "year", "t", "month"} | set(DYNAMIC_COLS)
    feature_cols = [col for col in df.columns if col not in drop_cols]
    cat_features = [col for col in CAT_COLS if col in feature_cols]
    return df, feature_cols, cat_features
