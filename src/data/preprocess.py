"""Препроцессинг сырых данных.

Изменения:
- make_static_consistent теперь применяется ТОЛЬКО к truly-static категориальным
  колонкам (region/locality/open_date_cat/area_cat) и использует "first" вместо "last".
  Это устраняет утечку: раньше для CV-фолда Sep-2024 все строки получали
  актуальные на Feb-2025 значения. Численные time-varying фичи (population,
  households, traffic, infrastructure counts) больше не перезаписываются — модель
  видит честные исторические значения, что также является дополнительным сигналом.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .loader import RENAME_MAP, DYNAMIC_COLS, STORE_STATIC_COLS  # noqa: F401


# Truly static — эти признаки магазина в реальности не меняются (или меняются крайне
# редко: например, переход магазина в другую категорию по площади/возрасту).
# Используем "first" — никакого взгляда в будущее.
_TRULY_STATIC_COLS = ["open_date_cat", "area_cat", "locality", "region"]


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "work_hours" in df.columns:
        df["work_hours"] = df["work_hours"].clip(lower=5, upper=25).astype(np.float32)
    for col in ["medical_300", "stops_300", "grocery_500", "schools_300",
                "marketplaces_100", "foot_traffic", "car_traffic",
                "cancellations"]: # возможно cancellations стоит оставить
        if col not in df.columns:
            continue
        q = df[col].quantile(0.995)
        df[f"{col}_clipped"] = df[col].clip(upper=q).astype(np.float32)
    return df

def correct_population(df: pd.DataFrame) -> pd.DataFrame:
    """Меняет население в паре (город, region) на медиану населения по его значениям 
    (отбрасывая предварительно нулевые значения)"""
    df = df.copy()
    population_medians = df.groupby(['locality', 'region']).apply(
        lambda x: x[x['population'] != 0]['population'].median()
    ).reset_index()
    population_medians.columns = ['locality', 'region', 'Median_Population']
    population_medians.fillna(100, inplace=True)

    # Create a mapping dictionary from (city, region) to median population
    population_mapping = dict(zip(
        zip(population_medians['locality'], population_medians['region']),
        population_medians['Median_Population']
    ))

    # Replace population values
    df['population'] = df.apply(
        lambda row: population_mapping.get((row['locality'], row['region']), row['population']),
        axis=1
    )
    return df

def make_static_consistent(df: pd.DataFrame) -> pd.DataFrame:
    """Применяется только к truly-static категориальным колонкам.
    "first" вместо "last" — нет утечки из будущего.
    Time-varying численные признаки (population/traffic/infrastructure) НЕ трогаем."""
    df = df.sort_values(["store_id", "t"]).copy()
    for c in _TRULY_STATIC_COLS:
        if c not in df.columns:
            continue
        df[c] = df.groupby("store_id")[c].transform("first")
    return df


def adjust_rto_for_inflation(df: pd.DataFrame,
                             inflation_file="data/processed/inflation_coefficients.csv") -> pd.DataFrame:
    """
    Adjust РТО to March 2025 prices using inflation coefficients.
    """
    df = df.copy()
    required_cols = ['year', 'month', 'rto']
    if not all(col in df.columns for col in required_cols):
        return df

    # Load inflation coefficients
    infl_df = pd.read_csv(inflation_file)

    # Merge coefficients
    df = df.merge(infl_df[['year', 'month', 'inflation_coefficient']],
                  on=['year', 'month'], how='left')

    # If no coefficient found (e.g., future months), keep original
    df['inflation_coefficient'] = df['inflation_coefficient'].fillna(1.0)

    df['rto'] = df['rto'] * df['inflation_coefficient']
    df.drop(columns=['inflation_coefficient'], inplace=True)

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
    df = correct_population(df)
    df = clean_outliers(df)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"Saved: {args.out_path}  shape={df.shape}")


if __name__ == "__main__":
    main()
