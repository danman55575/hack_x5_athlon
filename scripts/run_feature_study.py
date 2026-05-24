"""Автоматизированный EDA и быстрые абляции leakage-safe фичей.

Скрипт использует существующие модули проекта:
- загрузку и препроцессинг из `src.data`;
- модельные обёртки и CV-логику из `src.pipeline` / `src.validation`;
- конфиг LightGBM из `configs/lgbm.yaml`.

Цель скрипта — не строить новый пайплайн с нуля, а воспроизводимо:
1. собрать EDA-артефакты;
2. прогнать быстрые абляции групп фичей на одной быстрой модели;
3. сохранить краткий отчёт и таблицу результатов.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.features import audit_feature_frame, get_feature_groups, select_feature_columns
from src.data.loader import add_target_row_for_march_2025, load_raw
from src.pipeline import (
    _compute_segment_mapes,
    _maybe_weights,
    _prepare,
    _sanity_cap,
    _train_seed_bag,
)
from src.utils.io import load_yaml
from src.validation.cv import default_folds
from src.validation.metrics import mape


EDA_DIR = Path("artifacts/eda")
REPORTS_DIR = Path("experiments/reports")


class _NullLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None


NULL_LOGGER = _NullLogger()


@dataclass
class StudyResult:
    name: str
    model: str
    target: str
    feature_groups: list[str]
    feature_count: int
    total_nan: int
    all_nan_count: int
    folds: list[str]
    fold_mapes: dict[str, float]
    mean_mape: float
    march_fold_mape: float | None
    segment_small: float | None
    segment_medium: float | None
    segment_large: float | None
    top_features: list[str]
    comment: str
    oof_pred: np.ndarray | None = None


def _ensure_dirs() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _month_label(frame: pd.DataFrame) -> pd.Series:
    return frame["year"].astype(str) + "-" + frame["month"].astype(str).str.zfill(2)


def _panel_acf(series: np.ndarray, max_lag: int = 14) -> dict[int, float]:
    out: dict[int, float] = {}
    values = np.asarray(series, dtype=float)
    for lag in range(1, max_lag + 1):
        left = values[:-lag]
        right = values[lag:]
        mask = np.isfinite(left) & np.isfinite(right)
        if mask.sum() < 3:
            out[lag] = np.nan
            continue
        out[lag] = float(np.corrcoef(left[mask], right[mask])[0, 1])
    return out


def _save_plot(fig: plt.Figure, filename: str) -> Path:
    path = EDA_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run_eda(raw_df: pd.DataFrame, model_df: pd.DataFrame) -> tuple[dict[str, object], list[Path]]:
    artifacts: list[Path] = []
    summary: dict[str, object] = {}

    panel_history = raw_df.groupby("store_id")["t"].nunique()
    summary["n_stores"] = int(raw_df["store_id"].nunique())
    summary["n_months"] = int(raw_df["t"].nunique())
    summary["dup_store_month"] = int(raw_df.duplicated(["store_id", "t"]).sum())
    summary["panel_full_share"] = float((panel_history == raw_df["t"].nunique()).mean())
    summary["history_stats"] = panel_history.describe().to_dict()
    summary["stores_ge_12"] = int((panel_history >= 12).sum())
    summary["stores_ge_13"] = int((panel_history >= 13).sum())
    summary["stores_ge_14"] = int((panel_history >= 14).sum())
    summary["stores_ge_24"] = int((panel_history >= 24).sum())

    summary["rto_na"] = int(model_df["rto"].isna().sum())
    summary["rto_zero"] = int((model_df["rto"] == 0).sum())
    summary["rto_neg"] = int((model_df["rto"] < 0).sum())
    summary["rto_quantiles"] = model_df["rto"].quantile(
        [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    ).to_dict()
    summary["log_rto_quantiles"] = np.log1p(model_df["rto"]).quantile(
        [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    ).to_dict()

    static_cols = [
        "region",
        "locality",
        "area_cat",
        "cashboxes",
        "alco_flag",
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
    ]
    summary["changing_static_counts"] = {
        col: int((raw_df.groupby("store_id")[col].nunique(dropna=False) > 1).sum())
        for col in static_cols
        if col in raw_df.columns
    }

    stores_by_month = raw_df.groupby(["year", "month"])["store_id"].nunique().reset_index(name="n_stores")
    stores_by_month["label"] = _month_label(stores_by_month)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(stores_by_month["label"], stores_by_month["n_stores"], marker="o")
    ax.set_title("Число магазинов по месяцам")
    ax.set_ylabel("Число магазинов")
    ax.tick_params(axis="x", rotation=70)
    artifacts.append(_save_plot(fig, "stores_by_month.png"))

    month_agg = model_df.groupby(["year", "month"]).agg(
        total_rto=("rto", "sum"),
        mean_rto=("rto", "mean"),
        median_rto=("rto", "median"),
    ).reset_index()
    month_agg["label"] = _month_label(month_agg)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(month_agg["label"], month_agg["total_rto"], marker="o", color="#1f77b4")
    ax.set_title("Суммарный РТО сети по месяцам")
    ax.set_ylabel("Суммарный РТО")
    ax.tick_params(axis="x", rotation=70)
    artifacts.append(_save_plot(fig, "total_rto_by_month.png"))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(month_agg["label"], month_agg["mean_rto"], marker="o", label="Средний РТО")
    ax.plot(month_agg["label"], month_agg["median_rto"], marker="o", label="Медианный РТО")
    ax.set_title("Средний и медианный РТО магазина по месяцам")
    ax.legend()
    ax.tick_params(axis="x", rotation=70)
    artifacts.append(_save_plot(fig, "mean_median_rto_by_month.png"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(model_df["rto"], bins=60, color="#4c78a8", alpha=0.85)
    axes[0, 0].set_title("Распределение РТО")
    axes[0, 1].hist(np.log1p(model_df["rto"]), bins=60, color="#f58518", alpha=0.85)
    axes[0, 1].set_title("Распределение log1p(РТО)")
    month_samples = model_df[["year", "month", "rto"]].copy()
    month_samples["label"] = _month_label(month_samples)
    labels = month_samples["label"].drop_duplicates().tolist()
    grouped_boxes = [month_samples.loc[month_samples["label"] == label, "rto"].values for label in labels]
    axes[1, 0].boxplot(grouped_boxes, tick_labels=labels, showfliers=False)
    axes[1, 0].set_title("Распределение РТО по месяцам")
    axes[1, 0].tick_params(axis="x", rotation=70)
    abs_pct_error = (model_df["rto"].median() / model_df["rto"]).replace([np.inf, -np.inf], np.nan)
    axes[1, 1].hist(abs_pct_error.dropna(), bins=60, color="#54a24b", alpha=0.85)
    axes[1, 1].set_title("Насколько MAPE чувствителен к малым РТО")
    axes[1, 1].set_xlabel("median_rto / rto")
    artifacts.append(_save_plot(fig, "rto_distributions.png"))

    month_total = model_df.groupby("t")["rto"].sum().sort_index()
    network_log_acf = _panel_acf(np.log1p(month_total.values), max_lag=14)
    summary["network_log_acf"] = network_log_acf

    demeaned = np.log1p(model_df["rto"]) - np.log1p(model_df.groupby("store_id")["rto"].transform("mean"))
    demeaned_acf = {}
    for lag in range(1, 15):
        lagged = demeaned.groupby(model_df["store_id"]).shift(lag)
        mask = np.isfinite(demeaned.values) & np.isfinite(lagged.values)
        demeaned_acf[lag] = float(np.corrcoef(demeaned.values[mask], lagged.values[mask])[0, 1])
    summary["demeaned_log_acf"] = demeaned_acf

    lag1 = model_df.groupby("store_id")["rto"].shift(1)
    lag2 = model_df.groupby("store_id")["rto"].shift(2)
    growth = np.log(lag1 / lag2)
    growth_acf = {}
    for lag in (1, 2, 4, 6):
        lagged = growth.groupby(model_df["store_id"]).shift(lag)
        mask = np.isfinite(growth.values) & np.isfinite(lagged.values)
        growth_acf[lag] = float(np.corrcoef(growth.values[mask], lagged.values[mask])[0, 1])
    summary["growth_acf"] = growth_acf

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(list(network_log_acf.keys()), list(network_log_acf.values()), color="#4c78a8")
    axes[0].set_title("ACF log1p суммарного РТО")
    axes[1].bar(list(demeaned_acf.keys()), list(demeaned_acf.values()), color="#f58518")
    axes[1].set_title("ACF demeaned log1p(РТО)")
    axes[2].bar(list(growth_acf.keys()), list(growth_acf.values()), color="#54a24b")
    axes[2].set_title("ACF лог-роста")
    for ax in axes:
        ax.set_xlabel("Лаг")
    artifacts.append(_save_plot(fig, "acf_overview.png"))

    march_feb_parts = []
    for year in [2023, 2024]:
        feb = model_df[(model_df["year"] == year) & (model_df["month"] == 2)][["store_id", "rto"]].rename(
            columns={"rto": "feb_rto"}
        )
        mar = model_df[(model_df["year"] == year) & (model_df["month"] == 3)][["store_id", "rto"]].rename(
            columns={"rto": "mar_rto"}
        )
        part = feb.merge(mar, on="store_id")
        part["year"] = year
        part["ratio"] = part["mar_rto"] / part["feb_rto"]
        march_feb_parts.append(part)
    march_feb = pd.concat(march_feb_parts, ignore_index=True)
    summary["global_march_feb_ratio_median"] = float(march_feb["ratio"].median())
    summary["global_march_feb_ratio_mean"] = float(march_feb["ratio"].mean())
    meta = model_df[["store_id", "region", "area_cat"]].drop_duplicates("store_id")
    march_feb = march_feb.merge(meta, on="store_id", how="left")

    region_ratio = (
        march_feb.groupby("region")
        .agg(n=("ratio", "size"), median_ratio=("ratio", "median"))
        .sort_values("n", ascending=False)
        .head(15)
    )
    area_ratio = (
        march_feb.groupby("area_cat")
        .agg(n=("ratio", "size"), median_ratio=("ratio", "median"))
        .sort_values("n", ascending=False)
    )
    summary["region_ratio_top"] = region_ratio.round(4).reset_index().to_dict(orient="records")
    summary["area_ratio"] = area_ratio.round(4).reset_index().to_dict(orient="records")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(region_ratio.index[::-1], region_ratio["median_ratio"][::-1], color="#4c78a8")
    axes[0].set_title("Медианный коэффициент март/февраль по регионам")
    axes[1].barh(area_ratio.index[::-1], area_ratio["median_ratio"][::-1], color="#f58518")
    axes[1].set_title("Медианный коэффициент март/февраль по площади")
    artifacts.append(_save_plot(fig, "march_feb_ratios.png"))

    return summary, artifacts


def _baseline_predictions(df_feat: pd.DataFrame) -> dict[str, pd.Series]:
    pred_last = df_feat["rto_lag_1"]
    pred_yoy = df_feat["rto_lag_12"]
    pred_seasonal = df_feat["rto_seasonal_ratio_baseline"]
    pred_blend = pd.DataFrame(
        {
            "last": pred_last,
            "yoy": pred_yoy,
            "seasonal": pred_seasonal,
        }
    ).mean(axis=1)
    return {
        "baseline_last_month": pred_last,
        "baseline_same_month_last_year": pred_yoy,
        "baseline_seasonal_ratio": pred_seasonal,
        "baseline_blend": pred_blend,
    }


def run_baselines(df_feat: pd.DataFrame, folds: list) -> list[StudyResult]:
    results: list[StudyResult] = []
    baselines = _baseline_predictions(df_feat)
    for name, pred in baselines.items():
        fold_mapes: dict[str, float] = {}
        segment_rows: list[dict[str, float]] = []
        oof = np.full(len(df_feat), np.nan, dtype=np.float64)
        for fold in folds:
            _, va_idx = fold.split(df_feat)
            valid = df_feat.loc[va_idx, ["store_id", "rto"]].copy()
            valid["pred"] = pred.loc[va_idx].values
            valid = valid.dropna()
            fold_mapes[fold.name] = float(mape(valid["rto"].values, valid["pred"].values))
            oof[valid.index] = valid["pred"].values
            segment_rows.append(
                _compute_segment_mapes(
                    df_feat.loc[df_feat["t"] < df_feat.loc[va_idx, "t"].iloc[0], ["store_id", "rto"]],
                    valid[["store_id", "rto"]],
                    valid["pred"].values,
                )
            )
        mean_mape = float(np.mean(list(fold_mapes.values())))
        march_fold = next((value for fold_name, value in fold_mapes.items() if fold_name.endswith("-03")), None)
        seg_small = float(np.mean([row.get("small", np.nan) for row in segment_rows]))
        seg_medium = float(np.mean([row.get("medium", np.nan) for row in segment_rows]))
        seg_large = float(np.mean([row.get("large", np.nan) for row in segment_rows]))
        results.append(
            StudyResult(
                name=name,
                model="baseline",
                target="rto",
                feature_groups=[],
                feature_count=0,
                total_nan=int(pred.isna().sum()),
                all_nan_count=0,
                folds=list(fold_mapes.keys()),
                fold_mapes=fold_mapes,
                mean_mape=mean_mape,
                march_fold_mape=march_fold,
                segment_small=seg_small,
                segment_medium=seg_medium,
                segment_large=seg_large,
                top_features=[],
                comment="Ручной baseline без обучения",
                oof_pred=oof,
            )
        )
    return results


def _aggregate_segment_metric(segment_mapes: list[dict[str, float]], key: str) -> float | None:
    values = [row[key] for row in segment_mapes if key in row]
    if not values:
        return None
    return float(np.mean(values))


def run_log_experiment(
    df_feat: pd.DataFrame,
    all_feature_cols: list[str],
    cat_features: list[str],
    y_train_all: pd.Series,
    config: dict,
    feature_groups: list[str],
    folds: list,
    comment: str,
) -> StudyResult:
    feat_cols = select_feature_columns(all_feature_cols, feature_groups)
    feat_audit = audit_feature_frame(df_feat, feat_cols)
    cat_in = [c for c in cat_features if c in feat_cols] if config.get("use_cat", True) else None
    seeds = list(config.get("seed_bag", [config.get("seed", 2026)]))
    params = dict(config.get("params", {}))

    fold_mapes: dict[str, float] = {}
    segment_rows: list[dict[str, float]] = []
    oof = np.full(len(df_feat), np.nan, dtype=np.float64)
    fi_parts: list[pd.Series] = []

    for fold in folds:
        tr_idx, va_idx = fold.split(df_feat)
        tr_mask = ~y_train_all.iloc[tr_idx].isna().values
        va_mask = ~y_train_all.iloc[va_idx].isna().values
        tr_idx = tr_idx[tr_mask]
        va_idx = va_idx[va_mask]

        X_tr = df_feat.loc[tr_idx, feat_cols]
        X_va = df_feat.loc[va_idx, feat_cols]
        y_tr = y_train_all.iloc[tr_idx].values
        y_va_orig = df_feat.loc[va_idx, "rto"].values
        sw = _maybe_weights(bool(config.get("mape_weights", False)), df_feat.loc[tr_idx, "rto"].values)

        pred_log, _, fold_model = _train_seed_bag(
            config["model"],
            params,
            X_tr,
            y_tr,
            X_va,
            y_train_all.iloc[va_idx].values,
            cat_in,
            sw,
            None,
            seeds,
        )
        pred_orig = np.clip(np.expm1(pred_log), 1.0, None)
        pred_orig = _sanity_cap(
            df_feat.loc[va_idx],
            pred_orig,
            logger=NULL_LOGGER,
            up_mul=float(config.get("sanity_cap_up", 2.0)),
            down_mul=float(config.get("sanity_cap_down", 0.5)),
            dump_path=None,
            tag=f"study_{fold.name}",
        )
        fold_mapes[fold.name] = float(mape(y_va_orig, pred_orig))
        segment_rows.append(
            _compute_segment_mapes(
                df_feat.loc[tr_idx, ["store_id", "rto"]],
                df_feat.loc[va_idx, ["store_id", "rto"]],
                pred_orig,
            )
        )
        oof[va_idx] = pred_orig
        if hasattr(fold_model, "feature_importance"):
            fold_fi = fold_model.feature_importance()
            if fold_fi is not None:
                fi_parts.append(fold_fi.rename(fold.name))

    top_features: list[str] = []
    if fi_parts:
        mean_fi = pd.concat(fi_parts, axis=1).fillna(0.0).mean(axis=1).sort_values(ascending=False)
        top_features = mean_fi.head(15).index.tolist()

    march_fold = next((value for fold_name, value in fold_mapes.items() if fold_name.endswith("-03")), None)
    return StudyResult(
        name=config["name"],
        model=config["model"],
        target="log1p_rto",
        feature_groups=feature_groups,
        feature_count=len(feat_cols),
        total_nan=int(feat_audit["total_nan"]),
        all_nan_count=len(feat_audit["all_nan_columns"]),
        folds=list(fold_mapes.keys()),
        fold_mapes=fold_mapes,
        mean_mape=float(np.mean(list(fold_mapes.values()))),
        march_fold_mape=march_fold,
        segment_small=_aggregate_segment_metric(segment_rows, "small"),
        segment_medium=_aggregate_segment_metric(segment_rows, "medium"),
        segment_large=_aggregate_segment_metric(segment_rows, "large"),
        top_features=top_features,
        comment=comment,
        oof_pred=oof,
    )


def run_ratio_experiment(
    df_feat: pd.DataFrame,
    all_feature_cols: list[str],
    cat_features: list[str],
    config: dict,
    feature_groups: list[str],
    folds: list,
    mode: str,
) -> StudyResult:
    feat_cols = select_feature_columns(all_feature_cols, feature_groups)
    feat_audit = audit_feature_frame(df_feat, feat_cols)
    cat_in = [c for c in cat_features if c in feat_cols] if config.get("use_cat", True) else None
    seeds = list(config.get("seed_bag", [config.get("seed", 2026)]))
    params = dict(config.get("params", {}))
    params["objective"] = "regression_l1"
    params["metric"] = "mae"

    fold_mapes: dict[str, float] = {}
    segment_rows: list[dict[str, float]] = []

    if mode == "mom_ratio":
        denom_col = "rto_lag_1"
    elif mode == "yoy_ratio":
        denom_col = "rto_lag_12"
    else:
        raise ValueError(f"Неизвестный режим ratio-target: {mode}")

    for fold in folds:
        tr_idx, va_idx = fold.split(df_feat)
        train = df_feat.loc[tr_idx].copy()
        valid = df_feat.loc[va_idx].copy()
        train["ratio_target"] = train["rto"] / train[denom_col]
        valid["ratio_target"] = valid["rto"] / valid[denom_col]

        train = train[np.isfinite(train["ratio_target"]) & (train[denom_col] > 0)]
        valid = valid[np.isfinite(valid["ratio_target"]) & (valid[denom_col] > 0)]

        clip_low, clip_high = train["ratio_target"].quantile([0.01, 0.99]).tolist()
        y_tr = train["ratio_target"].clip(clip_low, clip_high).values.astype(np.float64)
        pred_ratio, _, _ = _train_seed_bag(
            config["model"],
            params,
            train[feat_cols],
            y_tr,
            valid[feat_cols],
            valid["ratio_target"].values.astype(np.float64),
            cat_in,
            None,
            None,
            seeds,
        )
        pred_ratio = np.clip(pred_ratio, clip_low, clip_high)
        pred_orig = np.clip(valid[denom_col].values * pred_ratio, 1.0, None)

        fold_mapes[fold.name] = float(mape(valid["rto"].values, pred_orig))
        segment_rows.append(
            _compute_segment_mapes(
                train[["store_id", "rto"]],
                valid[["store_id", "rto"]],
                pred_orig,
            )
        )

    march_fold = next((value for fold_name, value in fold_mapes.items() if fold_name.endswith("-03")), None)
    return StudyResult(
        name=f"{config['name']}_{mode}",
        model=config["model"],
        target=mode,
        feature_groups=feature_groups,
        feature_count=len(feat_cols),
        total_nan=int(feat_audit["total_nan"]),
        all_nan_count=len(feat_audit["all_nan_columns"]),
        folds=list(fold_mapes.keys()),
        fold_mapes=fold_mapes,
        mean_mape=float(np.mean(list(fold_mapes.values()))),
        march_fold_mape=march_fold,
        segment_small=_aggregate_segment_metric(segment_rows, "small"),
        segment_medium=_aggregate_segment_metric(segment_rows, "medium"),
        segment_large=_aggregate_segment_metric(segment_rows, "large"),
        top_features=[],
        comment="Быстрый ratio-target без финального прогона",
        oof_pred=None,
    )


def _result_rows(results: list[StudyResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "name": result.name,
                "model": result.model,
                "target": result.target,
                "feature_groups": ",".join(result.feature_groups),
                "feature_count": result.feature_count,
                "total_nan": result.total_nan,
                "all_nan_count": result.all_nan_count,
                "folds": ",".join(result.folds),
                "fold_mapes": "; ".join(f"{k}:{v:.4f}" for k, v in result.fold_mapes.items()),
                "mean_mape": result.mean_mape,
                "march_fold_mape": result.march_fold_mape,
                "segment_small": result.segment_small,
                "segment_medium": result.segment_medium,
                "segment_large": result.segment_large,
                "top_features": ", ".join(result.top_features[:10]),
                "comment": result.comment,
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_mape", "march_fold_mape"], na_position="last")


def _save_best_experiment_plots(df_feat: pd.DataFrame, result: StudyResult) -> list[Path]:
    if result.oof_pred is None:
        return []
    artifacts: list[Path] = []
    valid = df_feat.loc[np.isfinite(result.oof_pred), ["store_id", "rto"]].copy()
    valid["pred"] = result.oof_pred[np.isfinite(result.oof_pred)]
    valid["error_pct"] = (valid["pred"] - valid["rto"]) / valid["rto"] * 100.0

    fig, ax = plt.subplots(figsize=(6, 6))
    sample = valid.sample(min(5000, len(valid)), random_state=2026)
    ax.scatter(sample["rto"], sample["pred"], s=8, alpha=0.25)
    max_val = float(max(sample["rto"].max(), sample["pred"].max()))
    ax.plot([0, max_val], [0, max_val], color="black", linewidth=1)
    ax.set_title("Fact vs prediction для лучшего быстрого эксперимента")
    ax.set_xlabel("Факт")
    ax.set_ylabel("Прогноз")
    artifacts.append(_save_plot(fig, "best_experiment_fact_vs_pred.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(valid["error_pct"], bins=80, color="#4c78a8", alpha=0.85)
    ax.set_title("Распределение процентной ошибки")
    ax.set_xlabel("Ошибка, %")
    artifacts.append(_save_plot(fig, "best_experiment_error_distribution.png"))

    return artifacts


def _save_experiment_plots(results_df: pd.DataFrame) -> list[Path]:
    artifacts: list[Path] = []
    model_rows = results_df[results_df["model"] != "baseline"].copy()
    if model_rows.empty:
        return artifacts

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(model_rows["name"], model_rows["mean_mape"], color="#4c78a8")
    ax.set_title("Средний MAPE по быстрым экспериментам")
    ax.set_ylabel("MAPE")
    ax.tick_params(axis="x", rotation=70)
    artifacts.append(_save_plot(fig, "quick_experiments_mean_mape.png"))

    best_row = model_rows.iloc[0]
    segment_values = [
        best_row["segment_small"],
        best_row["segment_medium"],
        best_row["segment_large"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["small", "medium", "large"], segment_values, color="#f58518")
    ax.set_title("MAPE по сегментам у лучшего эксперимента")
    ax.set_ylabel("MAPE")
    artifacts.append(_save_plot(fig, "best_experiment_segment_mapes.png"))
    return artifacts


def _build_experiments(base_cfg: dict) -> list[tuple[str, list[str], str]]:
    return [
        (
            "lgbm_fs1_static_te",
            ["static_calendar", "target_encoding"],
            "Статика, календарь и preexisting target encoding",
        ),
        (
            "lgbm_fs2_basic_lags",
            ["static_calendar", "target_encoding", "basic_lags"],
            "Добавлены ключевые лаги 1/2/3/4/6/12/13/14",
        ),
        (
            "lgbm_fs3_rolling",
            ["static_calendar", "target_encoding", "basic_lags", "rolling"],
            "Добавлены rolling mean/median/std/min/max",
        ),
        (
            "lgbm_fs4_growth",
            ["static_calendar", "target_encoding", "basic_lags", "rolling", "growth_ratio"],
            "Добавлены growth/ratio-фичи",
        ),
        (
            "lgbm_fs5_seasonality",
            ["static_calendar", "target_encoding", "basic_lags", "rolling", "growth_ratio", "seasonality"],
            "Добавлены сезонные фичи и март/февраль коэффициенты",
        ),
        (
            "lgbm_fs6_group",
            [
                "static_calendar",
                "target_encoding",
                "basic_lags",
                "rolling",
                "growth_ratio",
                "seasonality",
                "group_aggregates",
            ],
            "Добавлены групповые leakage-safe агрегаты",
        ),
        (
            "lgbm_fs7_anomaly",
            [
                "static_calendar",
                "target_encoding",
                "basic_lags",
                "rolling",
                "growth_ratio",
                "seasonality",
                "group_aggregates",
                "anomaly_residual",
            ],
            "Добавлены anomaly/residual-like индикаторы",
        ),
        (
            "lgbm_fs8_plus_dynamic_ets",
            [
                "static_calendar",
                "target_encoding",
                "basic_lags",
                "rolling",
                "growth_ratio",
                "seasonality",
                "group_aggregates",
                "anomaly_residual",
                "dynamic_covariates",
                "ets",
            ],
            "Поверх лучшего табличного набора добавлены динамические ковариаты и ETS",
        ),
        (
            "lgbm_fs9_external_macro",
            [
                "static_calendar",
                "target_encoding",
                "basic_lags",
                "rolling",
                "external_macro",
            ],
            "Добавлены внешние макро-фичи: инфляция и лаговые региональные расходы",
        ),
    ]


def _write_text_report(
    path: Path,
    eda_summary: dict[str, object],
    results_df: pd.DataFrame,
    artifacts: list[Path],
) -> None:
    lines = []
    lines.append("Отчёт по EDA и быстрым экспериментам")
    lines.append("")
    if eda_summary:
        lines.append("EDA")
        lines.append(f"- Магазинов: {eda_summary['n_stores']}")
        lines.append(f"- Месяцев истории: {eda_summary['n_months']}")
        lines.append(f"- Дублей store_id x month: {eda_summary['dup_store_month']}")
        lines.append(f"- Полная панель: {eda_summary['panel_full_share']:.2%}")
        lines.append(
            f"- Глобальный коэффициент март/февраль: median={eda_summary['global_march_feb_ratio_median']:.4f}, "
            f"mean={eda_summary['global_march_feb_ratio_mean']:.4f}"
        )
        lines.append(
            f"- ACF log1p суммарного РТО: lag_1={eda_summary['network_log_acf'][1]:.4f}, "
            f"lag_12={eda_summary['network_log_acf'][12]:.4f}"
        )
        lines.append(
            f"- ACF demeaned log1p(РТО): lag_1={eda_summary['demeaned_log_acf'][1]:.4f}, "
            f"lag_2={eda_summary['demeaned_log_acf'][2]:.4f}, lag_12={eda_summary['demeaned_log_acf'][12]:.4f}"
        )
        lines.append(
            f"- ACF лог-роста: lag_1={eda_summary['growth_acf'][1]:.4f}, lag_4={eda_summary['growth_acf'][4]:.4f}, "
            f"lag_6={eda_summary['growth_acf'][6]:.4f}"
        )
        lines.append("")
    lines.append("")
    lines.append("Быстрые эксперименты")
    for _, row in results_df.iterrows():
        lines.append(
            f"- {row['name']}: mean_mape={row['mean_mape']:.4f}, "
            f"march_fold={row['march_fold_mape'] if pd.notna(row['march_fold_mape']) else 'n/a'}, "
            f"groups={row['feature_groups']}"
        )
    lines.append("")
    lines.append("Артефакты")
    for artifact in artifacts:
        lines.append(f"- {artifact.as_posix()}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/v2.parquet")
    parser.add_argument("--config", default="configs/lgbm.yaml")
    parser.add_argument("--mode", choices=["all", "eda", "experiments"], default="all")
    args = parser.parse_args()

    _ensure_dirs()
    raw_df = load_raw("data/raw/train_2.csv")
    model_df = load_raw(args.train)

    artifacts: list[Path] = []
    eda_summary: dict[str, object] = {}
    if args.mode in {"all", "eda"}:
        eda_summary, eda_artifacts = run_eda(raw_df, model_df)
        artifacts.extend(eda_artifacts)

    if args.mode == "eda":
        report_path = REPORTS_DIR / "feature_study_eda.txt"
        _write_text_report(report_path, eda_summary, pd.DataFrame(), artifacts)
        print(f"EDA saved to {EDA_DIR}")
        print(f"Report saved to {report_path}")
        return

    base_cfg = load_yaml(args.config)
    base_cfg["seed_bag"] = [base_cfg.get("seed", 2026)]
    base_cfg["params"]["num_boost_round"] = 400
    base_cfg["params"]["early_stopping_rounds"] = 60
    base_cfg["cv_val_months"] = [[2025, 2], [2024, 12], [2024, 3]]

    df_feat, all_feature_cols, cat_features, y_train_all, _, full_audit = _prepare(
        args.train,
        target_transform="log1p",
        winsorize_quantile=float(base_cfg.get("winsorize_quantile", 0.999)),
        feature_groups=None,
    )
    print(
        "Подготовлен датасет для быстрых экспериментов:",
        f"rows={len(df_feat)}",
        f"features={len(all_feature_cols)}",
        f"all_nan={len(full_audit['all_nan_columns'])}",
    )
    print("Группы фичей:", {k: len(v) for k, v in get_feature_groups(all_feature_cols).items()})

    folds = default_folds(df_feat, val_months=[tuple(x) for x in base_cfg["cv_val_months"]])
    results: list[StudyResult] = []
    results.extend(run_baselines(df_feat, folds))

    for exp_name, feature_groups, comment in _build_experiments(base_cfg):
        cfg = {
            **base_cfg,
            "name": exp_name,
            "feature_groups": feature_groups,
            "save_oof": False,
            "skip_final_train": True,
        }
        result = run_log_experiment(
            df_feat=df_feat,
            all_feature_cols=all_feature_cols,
            cat_features=cat_features,
            y_train_all=y_train_all,
            config=cfg,
            feature_groups=feature_groups,
            folds=folds,
            comment=comment,
        )
        results.append(result)
        print(
            f"{result.name}: mean_mape={result.mean_mape:.4f}, "
            f"march_fold={result.march_fold_mape}, features={result.feature_count}"
        )

    model_results = [row for row in results if row.model != "baseline"]
    best_log = min(model_results, key=lambda row: (row.mean_mape, math.inf if row.march_fold_mape is None else row.march_fold_mape))

    ratio_cfg = {
        **base_cfg,
        "name": "lgbm_best_ratio",
        "save_oof": False,
        "skip_final_train": True,
    }
    results.append(
        run_ratio_experiment(
            df_feat=df_feat,
            all_feature_cols=all_feature_cols,
            cat_features=cat_features,
            config=ratio_cfg,
            feature_groups=best_log.feature_groups,
            folds=folds,
            mode="mom_ratio",
        )
    )
    results.append(
        run_ratio_experiment(
            df_feat=df_feat,
            all_feature_cols=all_feature_cols,
            cat_features=cat_features,
            config=ratio_cfg,
            feature_groups=best_log.feature_groups,
            folds=folds,
            mode="yoy_ratio",
        )
    )

    # Повторяем лучший log1p-эксперимент с OOF для графиков ошибок.
    best_oof_cfg = {
        **base_cfg,
        "name": f"{best_log.name}_oof",
        "feature_groups": best_log.feature_groups,
        "save_oof": False,
        "skip_final_train": True,
    }
    best_log_with_oof = run_log_experiment(
        df_feat=df_feat,
        all_feature_cols=all_feature_cols,
        cat_features=cat_features,
        y_train_all=y_train_all,
        config=best_oof_cfg,
        feature_groups=best_log.feature_groups,
        folds=folds,
        comment=f"{best_log.comment}; повтор для OOF-графиков",
    )

    results_df = _result_rows(results)
    results_path = REPORTS_DIR / "feature_study_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8")

    artifacts.extend(_save_experiment_plots(results_df))
    artifacts.extend(_save_best_experiment_plots(df_feat, best_log_with_oof))

    report_path = REPORTS_DIR / "feature_study_report.txt"
    _write_text_report(report_path, eda_summary, results_df, artifacts)

    print("\nЛучший быстрый log1p-эксперимент:")
    print(
        f"  {best_log.name}: mean_mape={best_log.mean_mape:.4f}, "
        f"march_fold={best_log.march_fold_mape}, groups={best_log.feature_groups}"
    )
    print(f"Таблица экспериментов: {results_path}")
    print(f"Текстовый отчёт: {report_path}")
    print(f"EDA-артефакты: {EDA_DIR}")


if __name__ == "__main__":
    main()
