"""Препроцессинг сырых данных без внешних источников.
Запуск:
    python -m src.data.preprocess --in data/raw/train_2.csv --out data/processed/v2.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .loader import RENAME_MAP, DYNAMIC_COLS, STORE_STATIC_COLS


# Индекс цен на продовольственные товары в ценах марта 2025
INFLATION_COEFFICIENTS = {
    (2023, 1): 1.2267,
    (2023, 2): 1.2171,
    (2023, 3): 1.2155,
    (2023, 4): 1.2120,
    (2023, 5): 1.2158,
    (2023, 6): 1.2159,
    (2023, 7): 1.2100,
    (2023, 8): 1.2107,
    (2023, 9): 1.2004,
    (2023, 10): 1.1844,
    (2023, 11): 1.1663,
    (2023, 12): 1.1492,
    (2024, 1): 1.1349,
    (2024, 2): 1.1262,
    (2024, 3): 1.1243,
    (2024, 4): 1.1188,
    (2024, 5): 1.1143,
    (2024, 6): 1.1073,
    (2024, 7): 1.1033,
    (2024, 8): 1.1034,
    (2024, 9): 1.0997,
    (2024, 10): 1.0863,
    (2024, 11): 1.0616,
    (2024, 12): 1.0347,
    (2025, 1): 1.0211,
    (2025, 2): 1.0083,
    (2025, 3): 1.0000,
}


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Аккуратные клиппинги без обрезки массива (теряем мало сигнала)."""
    df = df.copy()
    # Рабочие часы > 24 — артефакт ввода. Клипуем сверху.
    if "work_hours" in df.columns:
        df["work_hours"] = df["work_hours"].clip(lower=5, upper=25).astype(np.float32)
    # Жёсткие хвосты: 99.5-perc clip с сохранением исходного значения в отдельной колонке
    for col in ["medical_300", "stops_300", "grocery_500", "schools_300",
                "marketplaces_100", "foot_traffic", "car_traffic",
                "cancellations"]:
        if col not in df.columns:
            continue
        q = df[col].quantile(0.995)
        df[f"{col}_clipped"] = df[col].clip(upper=q).astype(np.float32)
    return df


def make_static_consistent(df: pd.DataFrame) -> pd.DataFrame:
    """Static-колонки магазина не должны меняться между месяцами. Берём последнее значение."""
    df = df.sort_values(["store_id", "t"]).copy()
    for c in STORE_STATIC_COLS:
        if c not in df.columns:
            continue
        df[c] = df.groupby("store_id")[c].transform("last")
    return df


def adjust_rto_for_inflation(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust РТО to March 2025 prices using inflation coefficients."""
    df = df.copy()
    if "rto" not in df.columns or "year" not in df.columns or "month" not in df.columns:
        return df
    
    # Apply inflation coefficients
    df["rto"] = df.apply(
        lambda row: row["rto"] * INFLATION_COEFFICIENTS.get((row["year"], row["month"]), 1.0),
        axis=1
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/raw/train_2.csv")
    ap.add_argument("--out_path", default="data/processed/v2.parquet")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    df = df.rename(columns=RENAME_MAP)
    df["store_id"] = df["store_id"].astype(np.int32)
    df["year"] = df["year"].astype(np.int16)
    df["month"] = df["month"].astype(np.int8)
    base_year = int(df["year"].min())
    df["t"] = ((df["year"] - base_year) * 12 + (df["month"] - 1)).astype(np.int16)
    df = df.sort_values(["store_id", "t"]).reset_index(drop=True)

    df = make_static_consistent(df)
    df = adjust_rto_for_inflation(df)
    df = clean_outliers(df)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"Saved: {args.out_path}  shape={df.shape}")


if __name__ == "__main__":
    main()
