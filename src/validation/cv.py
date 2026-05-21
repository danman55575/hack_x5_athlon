from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TimeFold:
    name: str
    train_max_t: int
    val_t: int

    def split(self, df: pd.DataFrame, time_col: str = "t"):
        train_idx = df.index[df[time_col] <= self.train_max_t].to_numpy()
        val_idx = df.index[df[time_col] == self.val_t].to_numpy()
        return train_idx, val_idx


def default_folds(df: pd.DataFrame, time_col: str = "t",
                  val_months: list[tuple[int, int]] | None = None) -> list[TimeFold]:
    if val_months is None:
        val_months = [(2025, 2), (2024, 12), (2024, 9), (2024, 3)]
    base_year = int(df["year"].min())
    folds = []
    for y, m in val_months:
        val_t = (y - base_year) * 12 + (m - 1)
        if not ((df["year"] == y) & (df["month"] == m)).any():
            continue
        folds.append(TimeFold(name=f"{y}-{m:02d}", train_max_t=val_t - 1, val_t=val_t))
    return folds


def predict_split(df: pd.DataFrame, time_col: str = "t",
                  predict_year: int = 2025, predict_month: int = 3):
    base_year = int(df["year"].min())
    predict_t = (predict_year - base_year) * 12 + (predict_month - 1)
    train_idx = df.index[df[time_col] < predict_t].to_numpy()
    predict_idx = df.index[df[time_col] == predict_t].to_numpy()
    return train_idx, predict_idx
