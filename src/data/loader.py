from pathlib import Path
import pandas as pd
import numpy as np

# mapping to canonical column names
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
    "РТО": "rto", # Inflation-adjusted РТО
    # Target-encoded categorical features (from preprocessing)
    "Торговая площадь сред РТО": "area_mean_rto",
    "Дата открытия сред РТО": "open_date_mean_rto",
    "Регион сред РТО": "region_mean_rto",
    "Населенный пункт сред РТО": "locality_mean_rto",
}

CAT_COLS = ["open_date_cat", "area_cat", "locality", "region"]
STORE_STATIC_COLS = [
    "open_date_cat", "area_cat", "locality", "region",
    "population", "households", "foot_traffic", "car_traffic",
    "marketplaces_100", "medical_300", "schools_300", "stops_300",
    "grocery_500", "p5_500", "cashboxes", "alco_flag",
    # Target-encoded features (static per store)
    "area_mean_rto", "open_date_mean_rto", "region_mean_rto", "locality_mean_rto",
]
DYNAMIC_COLS = ["promo_per_check", "items_per_check", "cancellations", "work_hours"]


def load_raw(train_path: str | Path = "data/raw/train_2.csv") -> pd.DataFrame:
    df = pd.read_csv(train_path)
    df = df.rename(columns=RENAME_MAP)
    # Унифицируем типы
    df["store_id"] = df["store_id"].astype(np.int32)
    df["year"] = df["year"].astype(np.int16)
    df["month"] = df["month"].astype(np.int8)
    # period index: 2023-01 -> 0, 2023-02 -> 1, ..., 2025-03 -> 26
    df["t"] = ((df["year"] - df["year"].min()) * 12 + (df["month"] - 1)).astype(np.int16)
    df = df.sort_values(["store_id", "t"]).reset_index(drop=True)
    return df


def add_target_row_for_march_2025(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт пустые строки для March 2025 (target month) для каждого магазина.
    Если они уже есть — ничего не делает."""
    has_march = ((df["year"] == 2025) & (df["month"] == 3)).any()
    if has_march:
        return df
    base_t = (2025 - df["year"].min()) * 12 + (3 - 1)
    static_cols = [c for c in STORE_STATIC_COLS if c in df.columns]
    last_per_store = (df.sort_values("t").groupby("store_id", as_index=False).tail(1)
                      [["store_id"] + static_cols].copy())
    new_rows = last_per_store.copy()
    new_rows["year"] = np.int16(2025)
    new_rows["month"] = np.int8(3)
    new_rows["t"] = np.int16(base_t)
    for col in DYNAMIC_COLS:
        if col in df.columns:
            new_rows[col] = np.nan
    new_rows["rto"] = np.nan
    out = pd.concat([df, new_rows], ignore_index=True)
    out = out.sort_values(["store_id", "t"]).reset_index(drop=True)
    return out
