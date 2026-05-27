"""Безутечковая фичегенерация для помесячного прогноза РТО.
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
    region_panel["region_spendings_lag_12"] = region_g.shift(12)

    shifted_region_spend = region_g.shift(1)
    region_panel["region_spendings_rmean_3"] = (
        shifted_region_spend
        .groupby(region_panel["region"])
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
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
        "region_spendings_lag_12",
        "region_spendings_rmean_3",
        "region_spendings_yoy_ratio",
        "region_spendings_lag1_to_rmean3",
    ]
    return df.merge(region_panel[macro_cols], on=["year", "month", "region"], how="left")


def add_cci_features(df: pd.DataFrame) -> pd.DataFrame:
    """CCI: глобальный макро-сигнал; оставляем cci и cci_diff_1."""
    df = df.copy()
    cci_values = np.array([
        CCI.get((year, quarter), np.nan)
        for year, quarter in zip(df["year"].values, df["quarter"].values)
    ], dtype=np.float64)
    df["cci"] = cci_values
    cci_panel = df[["year", "quarter", "t", "cci"]].drop_duplicates().sort_values("t").reset_index(drop=True)
    cci_panel["cci_lag_1"] = cci_panel["cci"].shift(1)
    cci_panel["cci_diff_1"] = cci_panel["cci"] - cci_panel["cci_lag_1"]
    cci_cols = ["year", "quarter", "t", "cci", "cci_diff_1"]
    df = df.drop(columns=["cci"], errors="ignore")
    df = df.merge(cci_panel[cci_cols], on=["year", "quarter", "t"], how="left")
    return df


def add_lag_features(
    df: pd.DataFrame,
    target: str = "rto",
    # ВАЖНО: сокращённый набор лагов. Раньше было 1..18, что для месячных данных
    # розницы добавляло шум на лагах 5,7,8,9,10,11,15-18.
    lags: tuple[int, ...] = (1, 2, 3, 4, 6, 12, 13, 14),
    group: str = "store_id",
) -> pd.DataFrame:
    g = df.groupby(group)[target]
    return with_columns(df, {f"{target}_lag_{lag}": g.shift(lag) for lag in lags})


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "rto",
    windows: tuple[int, ...] = (3, 6, 12),
    group: str = "store_id",
) -> pd.DataFrame:
    """Удалены дубли (rto_mean_N был копией rto_rmean_N), сокращён список окон."""
    shifted = df.groupby(group)[target].shift(1)
    new_cols: dict[str, pd.Series | np.ndarray] = {}
    for window in windows:
        roll = shifted.groupby(df[group]).rolling(window=window, min_periods=max(1, window // 3))
        new_cols[f"{target}_rmean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        new_cols[f"{target}_rstd_{window}"] = roll.std().reset_index(level=0, drop=True)
        new_cols[f"{target}_rmedian_{window}"] = roll.median().reset_index(level=0, drop=True)

    # CV-фичи: std/mean. Стабильнее, чем raw std при сильно разных масштабах магазинов.
    if f"{target}_rmean_3" in new_cols and f"{target}_rstd_3" in new_cols:
        new_cols[f"{target}_cv_3"] = safe_divide(new_cols[f"{target}_rstd_3"], new_cols[f"{target}_rmean_3"])
    if f"{target}_rmean_6" in new_cols and f"{target}_rstd_6" in new_cols:
        new_cols[f"{target}_cv_6"] = safe_divide(new_cols[f"{target}_rstd_6"], new_cols[f"{target}_rmean_6"])

    # Алиасы для обратной совместимости с group/seasonal-фичами (через rmean_3, rmean_12)
    if f"{target}_rmean_3" in new_cols:
        new_cols[f"{target}_mean_3"] = new_cols[f"{target}_rmean_3"]
    if f"{target}_rmean_12" in new_cols:
        new_cols[f"{target}_mean_12"] = new_cols[f"{target}_rmean_12"]

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
    lag12 = g.shift(12)
    lag13 = g.shift(13)

    new_cols: dict[str, pd.Series | np.ndarray] = {
        "log_growth_lag_1": np.log(safe_divide(lag1, lag2)),
        "log_growth_lag_2": np.log(safe_divide(lag2, lag3)),
        f"{target}_lag_1_div_lag_2": safe_divide(lag1, lag2),
        f"{target}_lag_12_div_lag_13": safe_divide(lag12, lag13),
        f"{target}_lag_1_div_lag_12": safe_divide(lag1, lag12),
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
    lag12 = g.shift(12)
    lag24 = g.shift(24)
    season_trend = safe_divide(lag12, lag24)
    return with_columns(
        df,
        {
            f"{target}_same_month_1y": same_month_1y,
            f"{target}_naive_seasonal": same_month_1y * season_trend,
        },
    )


def add_trend_features(df: pd.DataFrame, target: str = "rto", group: str = "store_id") -> pd.DataFrame:
    """Оставлен только slope_12 — более длинное окно стабильнее коротких."""
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
    new_cols[f"{target}_slope_12"] = (
        shifted.groupby(df[group])
        .rolling(window=12, min_periods=3)
        .apply(_rolling_slope, raw=False)
        .reset_index(level=0, drop=True)
    )
    return with_columns(df, new_cols)


def add_expanding_stats(df: pd.DataFrame, target: str = "rto", group: str = "store_id") -> pd.DataFrame:
    """Оставлены только cummean и lag1_to_cummean.

    cummin/cummax/cumstd монотонно растут со временем для типичного магазина с трендом,
    что даёт модели неявную «date-feature» через store-history и потенциальный date-leakage.
    """
    shifted = df.groupby(group)[target].shift(1)
    expanding = shifted.groupby(df[group]).expanding()
    cummean = expanding.mean().reset_index(level=0, drop=True)
    return with_columns(
        df,
        {
            f"{target}_cummean": cummean,
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
    if "days_in_month" not in df.columns:
        return df

    g_dim = df.groupby(group)["days_in_month"]
    days_in_month = df["days_in_month"].astype(np.float32)
    days_in_month_lag_1 = g_dim.shift(1).astype(np.float32)
    days_in_month_lag_12 = g_dim.shift(12).astype(np.float32)
    new_cols: dict[str, pd.Series | np.ndarray] = {
        "days_in_month_lag_1": days_in_month_lag_1,
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


def add_group_aggregations(df: pd.DataFrame, target: str = "rto") -> pd.DataFrame:
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
    df.drop(columns=["global_march_feb_ratio", "global_march_feb_ratio_hist_pairs"], inplace=True)
    return df


def encode_categoricals_ordinal(
    df: pd.DataFrame,
    cat_cols: list[str] | tuple[str, ...] = CAT_COLS,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Ordinal: cat.codes + 1, чтобы NaN→0 (а не -1) и не было phantom-класса."""
    mappings: dict[str, list[str]] = {}
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype("category")
        mappings[col] = list(df[col].cat.categories)
        # КРИТИЧНО: +1 чтобы NaN-код (-1) превратился в 0, а реальные категории — в 1..N.
        df[col] = (df[col].cat.codes.astype(np.int32) + np.int32(1))
    return df, mappings


def encode_categoricals_native(
    df: pd.DataFrame,
    cat_cols: list[str] | tuple[str, ...] = CAT_COLS,
) -> pd.DataFrame:
    """Native: оставляем pandas category dtype. Для XGBoost (enable_categorical=True),
    LightGBM (auto-detect) и CatBoost (через Pool.cat_features)."""
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def downcast(df):
    for c in df.select_dtypes(include="float64").columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include="int64").columns:
        if c == "store_id":
            df[c] = df[c].astype(np.int32)
        else:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def get_feature_groups(feature_cols: list[str]) -> dict[str, list[str]]:
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
    rolling_names = {"rto_cv_3", "rto_cv_6"}
    growth_names = {
        "rto_diff_1", "rto_diff_2", "rto_pct_1", "rto_pct_6",
        "rto_yoy_diff", "rto_yoy_ratio", "rto_log_ratio_1_12",
        "log_growth_lag_1", "log_growth_lag_2",
        "rto_lag_1_div_lag_2", "rto_lag_1_div_mean_3",
        "rto_mean_3_div_mean_12",
    }
    seasonal_names = {
        "rto_same_month_1y", "rto_naive_seasonal",
        "rto_lag_12_div_lag_13", "rto_lag_1_div_lag_12",
        "rto_seasonal_ratio_baseline",
        "rto_lag_12_per_day", "rto_per_day_yoy_ratio",
        "global_march_feb_ratio", "region_march_feb_ratio", "city_march_feb_ratio",
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


def build_features(
    df: pd.DataFrame,
    target: str = "rto",
    cat_encoding: str = "native",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Главный билдер фичей.

    cat_encoding:
        "native" — оставить CAT_COLS как pandas category dtype. Подходит для
            LightGBM, XGBoost (enable_categorical=True), CatBoost.
        "ordinal" — закодировать целыми (cat.codes + 1, NaN→0). Подходит для
            Ridge/MLP, которые не умеют категории.
    """
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
    df = add_cci_features(df)
    df = add_group_aggregations(df, target=target)
    df = add_group_seasonality_features(df, target=target)
    df = df.copy()

    if cat_encoding == "ordinal":
        df, _ = encode_categoricals_ordinal(df)
    elif cat_encoding == "native":
        df = encode_categoricals_native(df)
    else:
        raise ValueError(f"Unknown cat_encoding: {cat_encoding}")

    df = downcast(df)

    drop_cols = {target, "store_id", "year", "t", "month"} | set(DYNAMIC_COLS)
    feature_cols = [col for col in df.columns if col not in drop_cols]
    cat_features = [col for col in CAT_COLS if col in feature_cols]
    return df, feature_cols, cat_features


def audit_and_clean_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Удаляет полностью пустые фичи и заменяет inf на NaN."""
    audit = audit_feature_frame(df, feature_cols)
    all_nan = set(audit["all_nan_columns"])
    if all_nan:
        feature_cols = [c for c in feature_cols if c not in all_nan]

    # Заменяем inf/-inf на NaN только в числовых колонках (категории не трогаем).
    numeric_feats = [c for c in feature_cols if df[c].dtype.kind in "fi"]
    if numeric_feats:
        # inplace через assign в фичах
        sub = df[numeric_feats].replace([np.inf, -np.inf], np.nan)
        df[numeric_feats] = sub
    return df, feature_cols
