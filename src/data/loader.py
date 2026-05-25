from pathlib import Path
import pandas as pd
import numpy as np
from .utils import *


def load_raw(train_path: str | Path = "data/processed/v2.parquet") -> pd.DataFrame:
    p = Path(train_path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
        if "new_id" in df.columns:
            df = df.rename(columns=RENAME_MAP)

    if "РТО" in df.columns:
        import warnings
        warnings.warn(
            "load_raw() читает сырой train_2.csv (русские названия колонок остались). "
            "Для боевого pipeline сначала запусти `python -m src.data.preprocess`.",
            UserWarning,
        )
        df = df.rename(columns=RENAME_MAP)

    df["store_id"] = df["store_id"].astype(np.int32)
    df["year"] = df["year"].astype(np.int16)
    df["month"] = df["month"].astype(np.int8)
    df["t"] = ((df["year"] - BASE_YEAR) * 12 + (df["month"] - 1)).astype(np.int16)
    df = df.sort_values(["store_id", "t"]).reset_index(drop=True)
    return df


def add_target_row_for_march_2025(df: pd.DataFrame) -> pd.DataFrame:
    has_march = ((df["year"] == 2025) & (df["month"] == 3)).any()
    if has_march:
        return df

    base_t = (2025 - BASE_YEAR) * 12 + (3 - 1)
    exclude = set(DYNAMIC_COLS) | {"rto", "year", "month", "t"}
    cols_to_copy = [c for c in df.columns if c not in exclude]

    last_per_store = (df.sort_values("t")
                      .groupby("store_id", as_index=False)
                      .tail(1)[cols_to_copy].copy())
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
