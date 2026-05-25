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
    "foot_traffic",
    "car_traffic",
    "marketplaces_100",
    "medical_300",
    "schools_300",
    "stops_300",
    "grocery_500",
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
    "cancellations_clipped",
]

CALENDAR_COLS = [
    "month_sin",
    "month_cos",
    "is_jan",
    "is_feb",
    "is_mar",
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
    "inflation_",
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