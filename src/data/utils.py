import numpy as np
import pandas as pd

BASE_YEAR: int = 2023

RENAME_MAP = {
    "new_id": "store_id",
    "Год": "year",
    "Месяц": "month",
    "Среднее количество промо товаров в чеке": "promo_per_check",
    "Среднее количество товаров в чеке": "items_per_check",
    "Среднее количество отмен": "cancellations",
    "Рабочие часы в день": "work_hours",
    "Дата открытия, категориальный": "open_date_cat",
    "Торговая площадь, категориальный": "area_cat",
    "Населенный пункт": "locality",
    "Регион": "region",
    "Численность населения": "population",
    "Количество домохозяйств": "households",
    "Трафик пеший, в час": "foot_traffic",
    "Трафик авто, в час": "car_traffic",
    "Маркетплейсы, доставки, постаматы (100 м)": "marketplaces_100",
    "Медицинские уч. и аптеки (300 м)": "medical_300",
    "Школы (300 м)": "schools_300",
    "Остановки (300 м)": "stops_300",
    "Продуктовые магазины (500 м)": "grocery_500",
    "Пятерочки (500 м)": "p5_500",
    "Количество касс": "cashboxes",
    "Флаг алкогольной лицензии": "alco_flag",
    "РТО": "rto",
}

CAT_COLS = ["open_date_cat", "area_cat", "locality", "region"]
STORE_STATIC_COLS = [
    "open_date_cat", "area_cat", "locality", "region",
    "population", "households", "foot_traffic", "car_traffic",
    "marketplaces_100", "medical_300", "schools_300", "stops_300",
    "grocery_500", "p5_500", "cashboxes", "alco_flag",
]
DYNAMIC_COLS = ["promo_per_check", "items_per_check", "cancellations"]

STATIC_NUMERIC_COLS = [
    "work_hours",
    "population",
    "households",
    "p5_500",
    "cashboxes",
    "alco_flag",
    "region_spendings_inflated",
    "medical_300_clipped",
    "stops_300_clipped",
    "grocery_500_clipped",
    "schools_300_clipped",
    "marketplaces_100_clipped",
    "foot_traffic_clipped",
    "car_traffic_clipped",
]

CALENDAR_COLS = [
    "month_sin",
    "month_cos",
    "is_jan",
    "is_feb",
    "is_dec",
    "quarter",
    "days_in_month",
    "days_per_month_ratio",
    "days_in_month_lag_1",
    "days_ratio_curr_to_lag1",
]

ANOMALY_FEATURES = [
    "abs_log_growth_lag_1",
]

EXTERNAL_MACRO_PREFIXES = (
    "cci_",
    "region_spendings_",
    "country_spendings_",
    "region_to_country_spendings_",
)

DAYS_IN_MONTH_BASE = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    dtype=np.int8,
)

non_working_days = {
    # 2023 год
    (2023, 1): 19,  # 31.12.2022 - 08.01.2023 (9 дней) + субботы/воскресенья (10 дней) ⚠️
    (2023, 2): 9,   # 23-26 февраля (4 дня) + субботы/воскресенья (5 дней)
    (2023, 3): 8,   # Только субботы/воскресенья (8 дней)
    (2023, 4): 10,  # 29-30 апреля (2 дня) + субботы/воскресенья (8 дней)
    (2023, 5): 11,  # 1, 6-9 мая (5 дней) + субботы/воскресенья (6 дней)
    (2023, 6): 9,   # 12 июня (1 день) + субботы/воскресенья (8 дней)
    (2023, 7): 10,  # Только субботы/воскресенья (10 дней)
    (2023, 8): 8,   # Только субботы/воскресенья (8 дней)
    (2023, 9): 9,   # Только субботы/воскресенья (9 дней)
    (2023, 10): 9,  # Только субботы/воскресенья (9 дней)
    (2023, 11): 9,  # 4 ноября (1 день) + субботы/воскресенья (8 дней)
    (2023, 12): 10, # Только субботы/воскресенья (10 дней)

    # 2024 год
    (2024, 1): 14,  # 1-8 января (8 дней) + субботы/воскресенья (6 дней) ⚠️
    (2024, 2): 9,   # 23-25 февраля (3 дня) + субботы/воскресенья (6 дней)
    (2024, 3): 10,  # 8-10 марта (3 дня) + субботы/воскресенья (7 дней)
    (2024, 4): 9,   # 28 апреля - 1 мая (4 дня) + субботы/воскресенья (5 дней)
    (2024, 5): 12,  # 1, 9-12 мая (5 дней) + субботы/воскресенья (7 дней)
    (2024, 6): 10,  # 12 июня (1 день) + субботы/воскресенья (9 дней)
    (2024, 7): 8,   # Только субботы/воскресенья (8 дней)
    (2024, 8): 9,   # Только субботы/воскресенья (9 дней)
    (2024, 9): 9,   # Только субботы/воскресенья (9 дней)
    (2024, 10): 8,  # Только субботы/воскресенья (8 дней)
    (2024, 11): 9,  # 3-4 ноября (2 дня) + субботы/воскресенья (7 дней)
    (2024, 12): 11, # 29-31 декабря (3 дня) + субботы/воскресенья (8 дней)

    # 2025 год
    (2025, 1): 14,  # 1-8, 29-31 декабря 2024 (9 дней) + субботы/воскресенья (5 дней) ⚠️
    (2025, 2): 9,   # 22-23 февраля (2 дня) + субботы/воскресенья (7 дней)
    (2025, 3): 9,   # 8-9 марта (2 дня) + субботы/воскресенья (7 дней)
}


CCI = {
    (2023, 1): -19,
    (2023, 2): -15,
    (2023, 3): -13,
    (2023, 4): -13,
    (2024, 1): -7,
    (2024, 2): -6,
    (2024, 3): -7,
    (2024, 4): -9,
    (2025, 1): -11
}

avg_ticket = {
    (2023, 1): 465.4,
    (2023, 2): 442.7,
    (2023, 3): 440.1,
    (2023, 4): 497.8,
    (2024, 1): 517.6,
    (2024, 2): 494.2,
    (2024, 3): 488.0,
    (2024, 4): 576.8,
    (2025, 1): 561.7, 
}


def is_leap_year(year: int) -> bool:
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


def days_in_month_vec(years: np.ndarray, months: np.ndarray) -> np.ndarray:
    months = np.asarray(months, dtype=np.int64)
    years = np.asarray(years, dtype=np.int64)
    result = DAYS_IN_MONTH_BASE[months - 1].astype(np.int8).copy()
    leap_mask = np.array([is_leap_year(int(year)) for year in years], dtype=bool)
    result[(months == 2) & leap_mask] = np.int8(29)
    return result

def safe_divide(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> pd.Series:
    num_s = pd.Series(num, copy=False)
    den_s = pd.Series(den, copy=False).replace(0, np.nan)
    return num_s / den_s


def expanding_mean_shifted(series: pd.Series, group_key: pd.Series) -> pd.Series:
    return (
        series.groupby(group_key)
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )


def expanding_quantile_shifted(series: pd.Series, group_key: pd.Series, q: float) -> pd.Series:
    return (
        series.groupby(group_key)
        .expanding()
        .quantile(q)
        .shift(1)
        .reset_index(level=0, drop=True)
    )


def with_columns(
    df: pd.DataFrame,
    columns: dict[str, pd.Series | np.ndarray],
) -> pd.DataFrame:
    if not columns:
        return df
    extra = pd.DataFrame(columns, index=df.index)
    overlap = [col for col in extra.columns if col in df.columns]
    if overlap:
        df = df.drop(columns=overlap)
    return pd.concat([df, extra], axis=1)


def add_prev_year_group_mean(
    df: pd.DataFrame,
    target: str,
    group_cols: list[str],
    feature_name: str,
) -> pd.DataFrame:
    if any(col not in df.columns for col in group_cols):
        return df
    agg = (
        df.groupby(group_cols + ["year", "month"], dropna=False)[target]
        .mean()
        .reset_index(name=feature_name)
    )
    agg["year"] = agg["year"] + 1
    return df.merge(agg, on=group_cols + ["year", "month"], how="left")


def compute_historical_prev_month_ratio(
    df: pd.DataFrame,
    target: str,
    group_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Computes historical ratio of current month to previous month for each group.
    
    For each (group, month), computes the median ratio of month_t to month_t-1 across stores.
    Fills NaN ratios for forecast months with group's historical medians.
    Memory-optimized: only keeps necessary columns during computation.
    """
    if df.empty:
        df = df.copy()
        df[f"{prefix}_month_prev_ratio"] = np.nan
        return df

    # Work only with necessary columns to reduce memory footprint
    necessary_cols = ["store_id", "year", "month", target] + group_cols
    necessary_cols = [c for c in necessary_cols if c in df.columns]
    df_work = df[necessary_cols].sort_values(["store_id"] + group_cols + ["year", "month"])
    
    # Compute lagged value (previous month for each store)
    if group_cols:
        prev_month_value = df_work.groupby(["store_id"] + group_cols, dropna=False)[target].shift(1)
    else:
        prev_month_value = df_work.groupby("store_id", dropna=False)[target].shift(1)
    
    # Compute month-to-month ratio
    month_ratio = safe_divide(df_work[target], prev_month_value)
    
    # Find valid ratios without creating a full copy - use boolean indexing
    valid_mask = (
        (df_work[target].notna()) & 
        (prev_month_value.notna()) & 
        (month_ratio.notna())
    )
    
    # Extract only valid rows with minimal copying
    if valid_mask.any():
        idx_valid = df_work.index[valid_mask]
        valid_month = df_work.loc[idx_valid, "month"].values
        valid_ratio = month_ratio.loc[idx_valid].values
        
        # Build data for aggregation
        agg_data = {"month": valid_month, "month_ratio": valid_ratio}
        if group_cols:
            for col in group_cols:
                agg_data[col] = df_work.loc[idx_valid, col].values
        
        valid_df = pd.DataFrame(agg_data)
        
        # Compute median ratios
        if group_cols:
            median_ratios = valid_df.groupby(group_cols + ["month"], dropna=False)["month_ratio"].median().reset_index()
        else:
            median_ratios = valid_df.groupby("month", dropna=False)["month_ratio"].median().reset_index()
        
        median_ratios = median_ratios.rename(columns={"month_ratio": f"{prefix}_month_prev_ratio"})
    else:
        # No valid ratios found
        if group_cols:
            median_ratios = pd.DataFrame(columns=group_cols + ["month", f"{prefix}_month_prev_ratio"])
        else:
            median_ratios = pd.DataFrame(columns=["month", f"{prefix}_month_prev_ratio"])
    
    # Merge back to main dataframe
    merge_cols = (group_cols + ["month"]) if group_cols else ["month"]
    result = df.merge(median_ratios, on=merge_cols, how="left")
    
    # Fill any remaining NaNs with the group's overall median (fallback)
    ratio_col = f"{prefix}_month_prev_ratio"
    if group_cols:
        # Create group keys efficiently without string conversion
        group_tuples = df[group_cols].drop_duplicates().itertuples(index=False, name=None)
        for group_values in group_tuples:
            mask = pd.Series(True, index=result.index)
            for col, val in zip(group_cols, group_values if isinstance(group_values, tuple) else (group_values,)):
                mask = mask & (result[col] == val)
            
            ratio_values = result.loc[mask, ratio_col]
            if ratio_values.notna().any():
                median_ratio = ratio_values.median()
                result.loc[mask & result[ratio_col].isna(), ratio_col] = median_ratio
    else:
        # For global: use overall median
        ratio_values = result[ratio_col]
        if ratio_values.notna().any():
            median_ratio = ratio_values.median()
            result.loc[result[ratio_col].isna(), ratio_col] = median_ratio
    
    return result



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