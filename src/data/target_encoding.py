"""KFold target encoding по time-aware схеме.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def expanding_target_encode(df: pd.DataFrame, cat_col: str, target: str = "rto",
                            min_samples: int = 30, smoothing: float = 50.0,
                            time_col: str = "t") -> np.ndarray:
    """Возвращает row-level TE строго в исходном порядке df.

    Smoothing prior — это глобальное среднее target по строкам с t' < t (без утечки).
    Категорный mean — expanding по t (без утечки).
    Финальный fallback при NaN — тоже time-aware prior, а не global mean.
    """
    df = df.copy()
    df["_orig_idx"] = np.arange(len(df), dtype=np.int64)

    # ---------- TIME-AWARE GLOBAL PRIOR (без утечки) ----------
    t_agg = (df.groupby(time_col)[target]
             .agg(["sum", "count"])
             .reset_index()
             .sort_values(time_col))
    t_agg["sum"] = t_agg["sum"].fillna(0.0)
    t_agg["count"] = t_agg["count"].fillna(0).astype(np.int64)
    t_agg["cum_sum"] = t_agg["sum"].cumsum().shift(1, fill_value=0.0)
    t_agg["cum_cnt"] = t_agg["count"].cumsum().shift(1, fill_value=0).astype(np.int64)
    first_nonempty_mean = float(df.loc[df[target].notna(), target].iloc[:max(1, int(df[target].notna().sum() // 26))].mean()) \
        if df[target].notna().any() else 0.0
    t_agg["prior"] = np.where(
        t_agg["cum_cnt"] > 0,
        t_agg["cum_sum"] / np.maximum(t_agg["cum_cnt"], 1),
        first_nonempty_mean,
    )
    prior_by_t = dict(zip(t_agg[time_col].values, t_agg["prior"].values.astype(np.float64)))

    # ---------- PER-CATEGORY EXPANDING MEAN ----------
    grouped = (df.groupby([cat_col, time_col], dropna=False)[target]
                 .agg(["sum", "count"])
                 .reset_index()
                 .sort_values([cat_col, time_col]))
    grouped["sum"] = grouped["sum"].fillna(0.0)
    grouped["count"] = grouped["count"].fillna(0).astype(np.int64)
    grouped["cum_sum"] = grouped.groupby(cat_col)["sum"].cumsum() - grouped["sum"]
    grouped["cum_cnt"] = grouped.groupby(cat_col)["count"].cumsum() - grouped["count"]
    grouped["prior"] = grouped[time_col].map(prior_by_t).astype(np.float64)
    grouped["prior"] = grouped["prior"].fillna(first_nonempty_mean)
    grouped["te"] = (grouped["cum_sum"] + smoothing * grouped["prior"]) / (grouped["cum_cnt"] + smoothing)
    low_mask = grouped["cum_cnt"] < min_samples
    grouped.loc[low_mask, "te"] = grouped.loc[low_mask, "prior"]

    out = df.merge(grouped[[cat_col, time_col, "te"]], on=[cat_col, time_col], how="left")
    out = out.sort_values("_orig_idx", kind="stable")

    # Time-aware fallback вместо global overall_mean.
    te = out["te"].astype(np.float64).values
    nan_mask = np.isnan(te)
    if nan_mask.any():
        prior_series = out["t"].map(prior_by_t).astype(np.float64).values
        te = np.where(nan_mask, prior_series, te)
        # Если и prior_by_t NaN (для самого раннего t) — используем first_nonempty_mean
        te = np.where(np.isnan(te), first_nonempty_mean, te)
    return te.astype(np.float32)


def add_target_encodings(df: pd.DataFrame, target: str = "rto",
                         cat_cols=("region", "locality", "open_date_cat", "area_cat")) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[f"te_{col}"] = expanding_target_encode(df, col, target=target)
    if {"region", "month"}.issubset(df.columns):
        df["_region_month"] = df["region"].astype(str) + "_" + df["month"].astype(str)
        df["te_region_month"] = expanding_target_encode(df, "_region_month", target=target)
        df = df.drop(columns=["_region_month"])
    if {"area_cat", "open_date_cat"}.issubset(df.columns):
        df["_area_open"] = df["area_cat"].astype(str) + "_" + df["open_date_cat"].astype(str)
        df["te_area_open"] = expanding_target_encode(df, "_area_open", target=target)
        df = df.drop(columns=["_area_open"])
    if {"region", "area_cat"}.issubset(df.columns):
        df["_region_area"] = df["region"].astype(str) + "_" + df["area_cat"].astype(str)
        df["te_region_area"] = expanding_target_encode(df, "_region_area", target=target)
        df = df.drop(columns=["_region_area"])
    return df
