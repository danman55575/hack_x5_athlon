"""Главный пайплайн: seed bagging, OOF, time-aware TE, ensemble финальной модели.
"""
from __future__ import annotations
import time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from .data.loader import load_raw, add_target_row_for_march_2025
from .data.target_encoding import add_target_encodings
from .data.features import audit_feature_frame, build_features, select_feature_columns
from .validation.cv import default_folds, predict_split
from .validation.metrics import mape, mape_to_score
from .models.gbm import MODEL_REGISTRY
from .models.linear import LinearModel
from .models.mlp import MLPModel
from .utils.seed import set_seed
from .utils.logging import get_logger
from .utils.io import save_json, save_pickle


ALL_MODELS = {**MODEL_REGISTRY, "linear": LinearModel, "mlp": MLPModel}

# Hackathon submission requirements
EXPECTED_SUBMISSION_ROWS = 18657
MAX_SUBMISSION_FILE_SIZE = 1_000_000  # 1 MB


def get_model(name, params=None):
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    return ALL_MODELS[name](params or {})


def _prepare(train_path, target_transform, winsorize_quantile, feature_groups=None):
    df = load_raw(train_path)
    df = add_target_row_for_march_2025(df)
    df = add_target_encodings(df, target="rto")
    df_feat, feat_cols, cat_features = build_features(df, target="rto")
    df_feat = df_feat.reset_index(drop=True)

    ets_path = Path("data/processed/ets_features.parquet")
    if ets_path.exists():
        ets = pd.read_parquet(ets_path)
        df_feat = df_feat.merge(ets, on=["store_id", "t"], how="left")
        df_feat = df_feat.reset_index(drop=True)
        for col in ets.columns:
            if col.startswith("ets_") and col in df_feat.columns and col not in feat_cols:
                feat_cols.append(col)

    if feature_groups:
        feat_cols = select_feature_columns(feat_cols, feature_groups)
        cat_features = [c for c in cat_features if c in feat_cols]

    df_feat = df_feat.copy()
    feature_audit = audit_feature_frame(df_feat, feat_cols)

    # Display March 2025 features for the first store
    march_2025_df = df_feat[(df_feat["year"] == 2025) & (df_feat["month"] == 3)].copy()
    if len(march_2025_df) > 0:
        first_store_id = df_feat["store_id"].min()
        first_store_march = march_2025_df[march_2025_df["store_id"] == first_store_id]
        if len(first_store_march) > 0:
            row = first_store_march.iloc[0]
            print("\n" + "="*80)
            print(f"March 2025 Features for Store ID: {first_store_id}")
            print("="*80)
            for feat in feat_cols:
                if feat in row.index:
                    value = row[feat]
                    print(f"{feat}: {value}")
            print("="*80)
            

    cap = df_feat["rto"].quantile(winsorize_quantile)
    df_feat["_rto_train"] = df_feat["rto"].clip(upper=cap)

    if target_transform == "log1p":
        y_train_all = np.log1p(df_feat["_rto_train"].astype(np.float64))
        inv = lambda yp: np.clip(np.expm1(yp), 1.0, None)
    else:
        y_train_all = df_feat["_rto_train"].astype(np.float64)
        inv = lambda yp: np.clip(yp, 1.0, None)

    return df_feat, feat_cols, cat_features, y_train_all, inv, feature_audit


def _maybe_weights(use_mape, y_orig):
    if not use_mape:
        return None
    eps = 1.0
    w = 1.0 / np.maximum(np.abs(y_orig), eps)
    return w.astype(np.float64)


def _apply_num_boost_override(model_name, m_params, num_boost_override):
    if num_boost_override is None:
        return m_params
    nb = int(num_boost_override)
    if model_name in ("lightgbm", "xgboost"):
        m_params["num_boost_round"] = nb
        m_params.pop("early_stopping_rounds", None)
    elif model_name == "catboost":
        m_params["iterations"] = nb
        m_params.pop("od_wait", None)
        m_params.pop("od_type", None)
    return m_params


def _train_seed_bag(model_name, params, X_tr, y_tr, X_va, y_va,
                    cat_features, sample_weight, sample_weight_val,
                    seeds, num_boost_override=None):
    preds_val, best_iters = [], []
    last_model = None
    for s in seeds:
        m_params = dict(params)
        m_params = _apply_num_boost_override(model_name, m_params, num_boost_override)
        m = get_model(model_name, m_params)
        m.fit(X_tr, y_tr, X_va, y_va, cat_features=cat_features,
              sample_weight=sample_weight, sample_weight_val=sample_weight_val, seed=s)
        if X_va is not None:
            preds_val.append(m.predict(X_va))
        if m.best_iteration_:
            best_iters.append(m.best_iteration_)
        last_model = m
    avg = np.mean(preds_val, axis=0) if preds_val else None
    return avg, best_iters, last_model


def _weighted_geom_mean_iters(per_fold_data, multiplier, logger=None):
    medians, weights, names = [], [], []
    for d in per_fold_data:
        if d["iters"]:
            medians.append(float(np.median(d["iters"])))
            weights.append(float(d["weight"]))
            names.append(d["name"])
    if not medians:
        return None
    medians = np.asarray(medians, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    log_iters = np.log(np.maximum(medians, 1.0))
    if weights.sum() <= 0:
        weighted_log = float(np.mean(log_iters))
    else:
        weighted_log = float(np.average(log_iters, weights=weights))
    nb = int(np.exp(weighted_log) * multiplier)
    if logger is not None:
        logger.info(
            f"Final num_iters = {nb} = weighted_geom_mean("
            f"iters={dict(zip(names, medians.astype(int).tolist()))}, "
            f"weights={dict(zip(names, weights.tolist()))}) * {multiplier}"
        )
    return nb


def _sanity_cap(df_pred_rows: pd.DataFrame, pred: np.ndarray, logger,
                up_mul: float = 2.0, down_mul: float = 0.5,
                dump_path: Path | None = None,
                tag: str = "") -> np.ndarray:
    """Мягкая защита от единичных экстремальных предсказаний.

    Если pred/lag1 выходит за [down_mul, up_mul] — заменяем на медиану от безопасных
    fallback-предсказаний. Данные УЖЕ нормированы под цены марта 2025.

    Логируем заменённые new_id вместе с original/replacement в CSV (если dump_path задан).
    """
    pred = np.asarray(pred, dtype=np.float64).copy()
    lag1 = df_pred_rows.get("rto_lag_1", pd.Series(np.nan)).values.astype(np.float64)
    macro = df_pred_rows.get("grp_all_yoy_macro_median",
                              pd.Series(np.nan)).values.astype(np.float64)
    naive_seasonal = df_pred_rows.get("rto_naive_seasonal",
                                       pd.Series(np.nan)).values.astype(np.float64)
    same_m_1y = df_pred_rows.get("rto_same_month_1y",
                                  pd.Series(np.nan)).values.astype(np.float64)
    lag1_scaled = df_pred_rows.get("rto_lag_1_scaled_by_days",
                                    pd.Series(np.nan)).values.astype(np.float64)

    store_ids = df_pred_rows.get("store_id", pd.Series(np.arange(len(pred)))).values

    replaced_rows = []
    for i in range(len(pred)):
        if not np.isfinite(lag1[i]) or lag1[i] <= 0:
            continue
        ratio = pred[i] / lag1[i]
        if ratio > up_mul or ratio < down_mul:
            cand = []
            if np.isfinite(macro[i]) and macro[i] > 0:
                cand.append(lag1[i] * macro[i])
            if np.isfinite(naive_seasonal[i]) and naive_seasonal[i] > 0:
                cand.append(naive_seasonal[i])
            if np.isfinite(same_m_1y[i]) and same_m_1y[i] > 0:
                cand.append(same_m_1y[i])
            if np.isfinite(lag1_scaled[i]) and lag1_scaled[i] > 0:
                cand.append(lag1_scaled[i])
            if cand:
                old = pred[i]
                pred[i] = float(np.median(cand))
                replaced_rows.append({
                    "new_id": int(store_ids[i]) if np.isfinite(store_ids[i]) else -1,
                    "lag1": float(lag1[i]),
                    "ratio_before": float(ratio),
                    "pred_before": float(old),
                    "pred_after": float(pred[i]),
                    "ratio_after": float(pred[i] / lag1[i]),
                    "n_candidates": len(cand),
                })
    n_replaced = len(replaced_rows)
    if n_replaced > 0:
        logger.info(f"Sanity cap [{tag}]: replaced {n_replaced} extreme predictions "
                    f"(thresholds: up={up_mul}, down={down_mul})")
        if dump_path is not None and n_replaced <= 5000:
            try:
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(replaced_rows).to_csv(dump_path, index=False)
                logger.info(f"Sanity-cap dump written: {dump_path}")
            except Exception as e:
                logger.warning(f"Sanity-cap dump failed: {e}")
    return pred


def _validate_submission(submission: pd.DataFrame, logger,
                          expected_rows: int = EXPECTED_SUBMISSION_ROWS) -> None:
    """Жёсткая валидация формата посылки ПЕРЕД сохранением.

    Проверки соответствуют требованиям хакатона:
    - ровно 18657 строк
    - колонки {new_id, rto}
    - уникальные new_id, без NaN
    - все rto конечны и > 0
    Любое нарушение → AssertionError (намеренный cras).
    """
    assert set(submission.columns) == {"new_id", "rto"}, (
        f"Submission must have exactly columns {{new_id, rto}}, "
        f"got {list(submission.columns)}"
    )
    assert len(submission) == expected_rows, (
        f"Submission has {len(submission)} rows, expected {expected_rows}"
    )
    assert submission["new_id"].notna().all(), "NaN in new_id column"
    assert submission["new_id"].is_unique, (
        f"Duplicate new_id in submission: "
        f"{submission['new_id'].duplicated().sum()} duplicates"
    )
    n_nan = int(submission["rto"].isna().sum())
    assert n_nan == 0, f"NaN in predictions: {n_nan} NaN values"
    n_inf = int((~np.isfinite(submission["rto"])).sum())
    assert n_inf == 0, f"Non-finite (inf) predictions: {n_inf}"
    n_nonpos = int((submission["rto"] <= 0).sum())
    assert n_nonpos == 0, (
        f"Non-positive predictions: {n_nonpos} found, "
        f"min={submission['rto'].min()}"
    )
    rto = submission["rto"].astype(np.float64)
    logger.info(
        f"Submission validation OK: rows={len(submission)}, "
        f"rto min={rto.min():,.0f}, p50={rto.median():,.0f}, "
        f"mean={rto.mean():,.0f}, max={rto.max():,.0f}"
    )


def _validate_submission_file(path: Path, logger,
                               max_bytes: int = MAX_SUBMISSION_FILE_SIZE) -> None:
    size = Path(path).stat().st_size
    assert size < max_bytes, (
        f"Submission file too large: {size} bytes (max {max_bytes}). "
        f"Reduce numeric precision."
    )
    logger.info(f"Submission file size: {size:,} bytes (limit {max_bytes:,})")


def _compute_segment_mapes(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    pred_orig: np.ndarray,
) -> dict[str, float]:
    """Считает MAPE по сегментам магазинов на основе масштаба train-fold."""
    store_scale = df_train.groupby("store_id")["rto"].mean()
    if len(store_scale) < 3:
        return {}

    q1, q2 = store_scale.quantile([1 / 3, 2 / 3]).tolist()
    segment_map = pd.Series("medium", index=store_scale.index, dtype="object")
    segment_map.loc[store_scale <= q1] = "small"
    segment_map.loc[store_scale >= q2] = "large"

    valid = df_valid[["store_id", "rto"]].copy()
    valid["pred"] = pred_orig
    valid["segment"] = valid["store_id"].map(segment_map).fillna("medium")

    result: dict[str, float] = {}
    for segment_name, part in valid.groupby("segment"):
        if len(part) == 0:
            continue
        result[str(segment_name)] = float(mape(part["rto"].values, part["pred"].values))
    return result


def run_experiment(config: dict, train_path: str = "data/processed/v2.parquet") -> dict:
    set_seed(config.get("seed", 2026))
    exp_name = config["name"]
    logger = get_logger(exp_name, log_dir="experiments/logs")
    logger.info(f"=== {exp_name} ===")
    logger.info(json.dumps(config, ensure_ascii=False))

    t0 = time.time()
    target_transform = config.get("target_transform", "log1p")
    winsorize_q = config.get("winsorize_quantile", 1.0)
    feature_groups = config.get("feature_groups")
    use_mape_weights = bool(config.get("mape_weights", False))
    skip_final_train = bool(config.get("skip_final_train", False))
    seeds = config.get("seed_bag", [config.get("seed", 2026)])
    if isinstance(seeds, (int, str)):
        seeds = [int(seeds)]
    seeds = list(seeds)
    logger.info(f"Seed bag in use: {seeds} ({len(seeds)} seeds)")

    final_iter_multiplier = float(config.get("final_iter_multiplier", 1.10))

    # Конфигурируемые пороги sanity cap
    sanity_up = float(config.get("sanity_cap_up", 2.0))
    sanity_down = float(config.get("sanity_cap_down", 0.5))
    apply_cap_in_cv = bool(config.get("apply_sanity_cap_in_cv", True))

    df_feat, feat_cols, cat_features, y_train_all, inv, feature_audit = _prepare(
        train_path,
        target_transform,
        winsorize_q,
        feature_groups=feature_groups,
    )
    logger.info(f"Features: {len(feat_cols)}; rows: {len(df_feat)}")
    logger.info(
        "Аудит фичей: total_nan=%s, nan_feature_count=%s, all_nan=%s, inf_cols=%s",
        feature_audit["total_nan"],
        feature_audit["nan_feature_count"],
        len(feature_audit["all_nan_columns"]),
        len(feature_audit["inf_columns"]),
    )
    if feature_audit["all_nan_columns"]:
        logger.warning(
            "Полностью пустые фичи: %s",
            feature_audit["all_nan_columns"][:10],
        )
    if feature_audit["inf_columns"]:
        logger.warning(
            "Фичи с inf: %s",
            feature_audit["inf_columns"][:10],
        )

    cv_val_months = config.get("cv_val_months")
    if cv_val_months is not None:
        cv_val_months = [tuple(x) for x in cv_val_months]
    folds = default_folds(df_feat, val_months=cv_val_months)
    logger.info(f"Folds: {[f.name for f in folds]}")

    # ----- Веса фолдов и фильтр для финального num_iters -----
    cv_fold_weights = config.get("cv_fold_weights")
    if cv_fold_weights is None:
        cv_fold_weights = [1.0] * len(folds)
    cv_fold_weights = list(cv_fold_weights)[:len(folds)]
    while len(cv_fold_weights) < len(folds):
        cv_fold_weights.append(1.0)
    iter_thr = float(config.get("fold_iter_use_weights_gt", 0.5))
    logger.info(f"Fold weights: {dict(zip([f.name for f in folds], cv_fold_weights))}")
    logger.info(f"Iter selection threshold: only folds with weight > {iter_thr} are USED for choosing final num_iters")
    logger.info(f"Final iter multiplier: {final_iter_multiplier}")
    logger.info(f"Sanity cap thresholds: up={sanity_up}, down={sanity_down}, "
                f"apply_in_cv={apply_cap_in_cv}")

    use_cat = config.get("use_cat", True)
    cat_features_in = cat_features if use_cat else None

    fold_metrics = []
    trusted_fold_data: list[dict] = []
    oof_pred = np.full(len(df_feat), np.nan, dtype=np.float64)
    feature_importances: list[pd.Series] = []

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for fold, w in zip(folds, cv_fold_weights):
        tr_idx, va_idx = fold.split(df_feat)
        tr_mask = ~y_train_all.iloc[tr_idx].isna().values
        va_mask = ~y_train_all.iloc[va_idx].isna().values
        tr_idx = tr_idx[tr_mask]; va_idx = va_idx[va_mask]

        X_tr = df_feat.loc[tr_idx, feat_cols]
        y_tr = y_train_all.iloc[tr_idx].values
        X_va = df_feat.loc[va_idx, feat_cols]
        y_va_log = y_train_all.iloc[va_idx].values
        y_va_orig = df_feat.loc[va_idx, "rto"].values
        
        # Normalize validation target by days_in_month (for MAPE metric consistency)
        if "days_in_month" in df_feat.columns:
            y_va_orig = y_va_orig / df_feat.loc[va_idx, "days_in_month"].astype(np.float32).values

        # Compute sample weights (also normalized)
        tr_rto = df_feat.loc[tr_idx, "rto"].values
        if "days_in_month" in df_feat.columns:
            tr_rto = tr_rto / df_feat.loc[tr_idx, "days_in_month"].astype(np.float32).values
        sw = _maybe_weights(use_mape_weights, tr_rto)
        sw_v = None

        pred_log_avg, best_iters, fold_model = _train_seed_bag(
            config["model"], config.get("params", {}),
            X_tr, y_tr, X_va, y_va_log, cat_features_in, sw, sw_v, seeds)
        pred_orig = inv(pred_log_avg)

        if apply_cap_in_cv:
            cap_dump = Path("experiments/reports") / f"{exp_name}_{ts}_sanity_cap_{fold.name}.csv"
            pred_orig = _sanity_cap(
                df_feat.loc[va_idx], pred_orig, logger,
                up_mul=sanity_up, down_mul=sanity_down,
                dump_path=cap_dump, tag=f"CV_{fold.name}",
            )

        m_val = mape(y_va_orig, pred_orig)
        score = mape_to_score(m_val)
        oof_pred[va_idx] = pred_orig
        segment_mapes = _compute_segment_mapes(
            df_feat.loc[tr_idx, ["store_id", "rto"]],
            df_feat.loc[va_idx, ["store_id", "rto"]],
            pred_orig,
        )
        fold_metrics.append({"fold": fold.name, "mape": m_val, "score": score,
                             "n_train": len(tr_idx), "n_val": len(va_idx),
                             "best_iters": best_iters, "weight": float(w),
                             "segment_mapes": segment_mapes})
        used_for_iters = w > iter_thr
        if used_for_iters:
            trusted_fold_data.append({"name": fold.name, "weight": float(w),
                                      "iters": list(best_iters)})
        if hasattr(fold_model, "feature_importance"):
            fold_fi = fold_model.feature_importance()
            if fold_fi is not None:
                feature_importances.append(fold_fi.rename(fold.name))
        logger.info(f"Fold {fold.name} (w={w}): MAPE={m_val:.4f} score={score:.3f} "
                    f"best_iters={best_iters} "
                    f"{'[USED FOR ITERS]' if used_for_iters else '[DIAG ONLY]'}")
        if segment_mapes:
            logger.info(f"Fold {fold.name} сегменты: {segment_mapes}")

    total_w = sum(f["weight"] for f in fold_metrics)
    if total_w > 0:
        mean_mape = float(sum(f["mape"] * f["weight"] for f in fold_metrics) / total_w)
        mean_score = float(sum(f["score"] * f["weight"] for f in fold_metrics) / total_w)
    else:
        mean_mape = float(np.mean([f["mape"] for f in fold_metrics]))
        mean_score = float(np.mean([f["score"] for f in fold_metrics]))
    simple_mean_mape = float(np.mean([f["mape"] for f in fold_metrics]))
    logger.info(f"CV weighted MAPE = {mean_mape:.4f} | weighted score = {mean_score:.3f}")
    logger.info(f"CV simple   MAPE = {simple_mean_mape:.4f}")

    mean_feature_importance = None
    top_feature_importance = None
    if feature_importances:
        mean_feature_importance = (
            pd.concat(feature_importances, axis=1)
            .fillna(0.0)
            .mean(axis=1)
            .sort_values(ascending=False)
        )
        top_feature_importance = mean_feature_importance.head(20).to_dict()

    submission_path = None
    nb_final = None
    last_model = None
    if not skip_final_train:
        tr_idx, pred_idx = predict_split(df_feat, predict_year=2025, predict_month=3)
        tr_mask = ~y_train_all.iloc[tr_idx].isna().values
        tr_idx = tr_idx[tr_mask]
        X_tr = df_feat.loc[tr_idx, feat_cols]
        y_tr = y_train_all.iloc[tr_idx].values
        X_pr = df_feat.loc[pred_idx, feat_cols]
        sw = _maybe_weights(use_mape_weights, df_feat.loc[tr_idx, "rto"].values)

        nb_final = _weighted_geom_mean_iters(trusted_fold_data, final_iter_multiplier, logger=logger)
        if nb_final is None:
            all_data = [{"name": f["fold"], "weight": max(f["weight"], 1.0),
                         "iters": f["best_iters"]} for f in fold_metrics]
            nb_final = _weighted_geom_mean_iters(all_data, final_iter_multiplier, logger=logger)
            logger.warning(f"No trusted folds for iter selection; fallback nb_final={nb_final}")

        pred_log_avg_pred = []
        for s in seeds:
            m_params = dict(config.get("params", {}))
            m_params = _apply_num_boost_override(config["model"], m_params, nb_final)
            m = get_model(config["model"], m_params)
            m.fit(X_tr, y_tr, None, None, cat_features=cat_features_in,
                  sample_weight=sw, sample_weight_val=None, seed=s)
            pred_log_avg_pred.append(m.predict(X_pr))
            last_model = m
        pred_log_avg = np.mean(pred_log_avg_pred, axis=0)
        pred_orig = inv(pred_log_avg)

        cap_dump = Path("experiments/reports") / f"{exp_name}_{ts}_sanity_cap_FINAL.csv"
        pred_orig = _sanity_cap(
            df_feat.loc[pred_idx], pred_orig, logger,
            up_mul=sanity_up, down_mul=sanity_down,
            dump_path=cap_dump, tag="FINAL",
        )

        # Denormalize predictions: multiply by days_in_month to convert from per-day back to per-month
        days_in_month = df_feat.loc[pred_idx, "days_in_month"].values.astype(np.float32)
        pred_orig = pred_orig * days_in_month

        submission = pd.DataFrame({
            "new_id": df_feat.loc[pred_idx, "store_id"].values.astype(int),
            "rto": pred_orig,
        }).sort_values("new_id").reset_index(drop=True)

        _validate_submission(submission, logger)

        sub_path = Path("experiments/predictions") / f"{exp_name}_{ts}.csv"
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(sub_path, index=False)
        _validate_submission_file(sub_path, logger)

        final_sub_path = Path("data/submissions") / f"{exp_name}_{ts}_test.csv"
        final_sub_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(final_sub_path, index=False)
        _validate_submission_file(final_sub_path, logger)
        submission_path = str(sub_path)

    if config.get("save_oof", True):
        oof_path = Path("experiments/oof") / f"{exp_name}_{ts}_oof.parquet"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_df = df_feat[["store_id", "year", "month", "t", "rto"]].copy()
        oof_df["oof_pred"] = oof_pred
        oof_df.to_parquet(oof_path)

    if mean_feature_importance is not None:
        fi_path = Path("experiments/reports") / f"{exp_name}_{ts}_fi.csv"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        mean_feature_importance.to_csv(fi_path)

    march_fold = next((f for f in fold_metrics if f["fold"].endswith("-03")), None)
    result = {
        "name": exp_name, "timestamp": ts, "model": config["model"],
        "params": config.get("params", {}),
        "feature_groups": feature_groups,
        "feature_count": len(feat_cols),
        "feature_audit": feature_audit,
        "cv_folds": fold_metrics,
        "cv_mean_mape": mean_mape,
        "cv_mean_score": mean_score,
        "cv_simple_mean_mape": simple_mean_mape,
        "march_fold_mape": None if march_fold is None else march_fold["mape"],
        "top_feature_importance": top_feature_importance,
        "submission_path": submission_path,
        "elapsed_seconds": time.time() - t0,
        "n_seeds": len(seeds),
        "final_num_iters": nb_final,
        "use_cat": use_cat,
        "mape_weights": use_mape_weights,
        "target_transform": target_transform,
        "final_iter_multiplier": final_iter_multiplier,
        "skip_final_train": skip_final_train,
        "sanity_cap_up": sanity_up,
        "sanity_cap_down": sanity_down,
        "apply_sanity_cap_in_cv": apply_cap_in_cv,
    }
    save_json(result, Path("experiments/reports") / f"{exp_name}_{ts}.json")
    logger.info(f"Done in {result['elapsed_seconds']:.1f}s")
    if last_model is not None:
        try:
            save_pickle(last_model, Path("experiments/models") / f"{exp_name}_{ts}.pkl")
        except Exception as e:
            logger.warning(f"pickle failed: {e}")
    return result
