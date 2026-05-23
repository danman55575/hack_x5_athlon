"""Препроцессинг сырых данных.

"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .loader import RENAME_MAP, DYNAMIC_COLS, STORE_STATIC_COLS, BASE_YEAR  # noqa: F401


_TRULY_STATIC_COLS = ["open_date_cat", "area_cat", "locality", "region"]


INFLATION_COEFFICIENTS = {
    (2023, 1): 1.2267, (2023, 2): 1.2171, (2023, 3): 1.2155, (2023, 4): 1.2120,
    (2023, 5): 1.2158, (2023, 6): 1.2159, (2023, 7): 1.2100, (2023, 8): 1.2107,
    (2023, 9): 1.2004, (2023, 10): 1.1844, (2023, 11): 1.1663, (2023, 12): 1.1492,
    (2024, 1): 1.1349, (2024, 2): 1.1262, (2024, 3): 1.1243, (2024, 4): 1.1188,
    (2024, 5): 1.1143, (2024, 6): 1.1073, (2024, 7): 1.1033, (2024, 8): 1.1034,
    (2024, 9): 1.0997, (2024, 10): 1.0863, (2024, 11): 1.0616, (2024, 12): 1.0347,
    (2025, 1): 1.0211, (2025, 2): 1.0083, (2025, 3): 1.0000,
}


def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "work_hours" in df.columns:
        df["work_hours"] = df["work_hours"].clip(lower=5, upper=25).astype(np.float32)
    for col in ["medical_300", "stops_300", "grocery_500", "schools_300",
                "marketplaces_100", "foot_traffic", "car_traffic",
                "cancellations"]:
        if col not in df.columns:
            continue
        q = df[col].quantile(0.995)
        df[f"{col}_clipped"] = df[col].clip(upper=q).astype(np.float32)
    return df


def correct_population(df: pd.DataFrame) -> pd.DataFrame:
    """Заменяет population в паре (locality, region) на медиану по ненулевым значениям.
    """
    df = df.copy()
    mask_nonzero = df['population'] != 0
    medians = (
        df.loc[mask_nonzero]
          .groupby(['locality', 'region'], dropna=False)['population']
          .median()
          .rename('_median_pop')
          .reset_index()
    )
    df = df.merge(medians, on=['locality', 'region'], how='left')
    df['_median_pop'] = df['_median_pop'].fillna(100)
    df['population'] = df['_median_pop'].astype(df['population'].dtype)
    df = df.drop(columns=['_median_pop'])
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
    # Используем фиксированный BASE_YEAR из loader
    df["t"] = ((df["year"] - BASE_YEAR) * 12 + (df["month"] - 1)).astype(np.int16)
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
