"""Главный пайплайн с seed bagging, OOF, time-aware TE, ensemble финальной модели."""
from __future__ import annotations
import time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from .data.loader import load_raw, add_target_row_for_march_2025
from .data.target_encoding import add_target_encodings
from .data.features import build_features
from .validation.cv import default_folds, predict_split
from .validation.metrics import mape, mape_to_score
from .models.gbm import MODEL_REGISTRY
from .models.linear import LinearModel
from .models.mlp import MLPModel
from .utils.seed import set_seed
from .utils.logging import get_logger
from .utils.io import save_json, save_pickle


ALL_MODELS = {**MODEL_REGISTRY, "linear": LinearModel, "mlp": MLPModel}


def get_model(name, params=None):
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    return ALL_MODELS[name](params or {})


def _prepare(train_path, target_transform, winsorize_quantile):
    df = load_raw(train_path)
    df = add_target_row_for_march_2025(df)
    # TE до ordinal-кодирования (нужны исходные строки категорий)
    df = add_target_encodings(df, target="rto")
    df_feat, feat_cols, cat_features = build_features(df, target="rto")
    df_feat = df_feat.reset_index(drop=True)

    # ETS pred (если сохранён в кэше)
    ets_path = Path("data/processed/ets_features.parquet")
    if ets_path.exists():
        ets = pd.read_parquet(ets_path)
        df_feat = df_feat.merge(ets, on=["store_id", "t"], how="left")
        if "ets_pred" in df_feat.columns and "ets_pred" not in feat_cols:
            feat_cols.append("ets_pred")

    # Winsorize таргета только для train (val/test не трогаем)
    cap = df_feat["rto"].quantile(winsorize_quantile)
    df_feat["_rto_train"] = df_feat["rto"].clip(upper=cap)

    if target_transform == "log1p":
        y_train_all = np.log1p(df_feat["_rto_train"].astype(np.float64))
        inv = lambda yp: np.clip(np.expm1(yp), 1.0, None)
    else:
        y_train_all = df_feat["_rto_train"].astype(np.float64)
        inv = lambda yp: np.clip(yp, 1.0, None)

    return df_feat, feat_cols, cat_features, y_train_all, inv


def _maybe_weights(use_mape, y_orig):
    if not use_mape:
        return None
    eps = 1.0
    w = 1.0 / np.maximum(np.abs(y_orig), eps)
    return w.astype(np.float64)


def _train_seed_bag(model_name, params, X_tr, y_tr, X_va, y_va,
                    cat_features, sample_weight, sample_weight_val,
                    seeds, num_boost_override=None):
    """Тренируем модель для каждого seed, усредняем prediction."""
    preds_val = []
    best_iters = []
    last_model = None
    for s in seeds:
        m_params = dict(params)
        if num_boost_override is not None:
            m_params["num_boost_round"] = int(num_boost_override)
            m_params.pop("early_stopping_rounds", None)
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


def run_experiment(config: dict, train_path: str = "data/processed/v2.parquet") -> dict:
    set_seed(config.get("seed", 2026))
    exp_name = config["name"]
    logger = get_logger(exp_name, log_dir="experiments/logs")
    logger.info(f"=== {exp_name} ===")
    logger.info(json.dumps(config, ensure_ascii=False))

    t0 = time.time()
    target_transform = config.get("target_transform", "log1p")
    winsorize_q = config.get("winsorize_quantile", 1.0)
    use_mape_weights = bool(config.get("mape_weights", False))
    seeds = config.get("seed_bag", [config.get("seed", 2026)])

    df_feat, feat_cols, cat_features, y_train_all, inv = _prepare(
        train_path, target_transform, winsorize_q)
    logger.info(f"Features: {len(feat_cols)}; rows: {len(df_feat)}")

    cv_val_months = config.get("cv_val_months")
    if cv_val_months is not None:
        cv_val_months = [tuple(x) for x in cv_val_months]
    folds = default_folds(df_feat, val_months=cv_val_months)
    logger.info(f"Folds: {[f.name for f in folds]}")

    use_cat = config.get("use_cat", True)
    cat_features_in = cat_features if use_cat else None

    fold_metrics = []
    fold_best_iters = []
    oof_pred = np.full(len(df_feat), np.nan, dtype=np.float64)

    for fold in folds:
        tr_idx, va_idx = fold.split(df_feat)
        tr_mask = ~y_train_all.iloc[tr_idx].isna().values
        va_mask = ~y_train_all.iloc[va_idx].isna().values
        tr_idx = tr_idx[tr_mask]; va_idx = va_idx[va_mask]

        X_tr = df_feat.loc[tr_idx, feat_cols]
        y_tr = y_train_all.iloc[tr_idx].values
        X_va = df_feat.loc[va_idx, feat_cols]
        y_va_log = y_train_all.iloc[va_idx].values
        y_va_orig = df_feat.loc[va_idx, "rto"].values

        sw = _maybe_weights(use_mape_weights, df_feat.loc[tr_idx, "rto"].values)
        sw_v = _maybe_weights(use_mape_weights, y_va_orig)

        pred_log_avg, best_iters, _ = _train_seed_bag(
            config["model"], config.get("params", {}),
            X_tr, y_tr, X_va, y_va_log, cat_features_in, sw, sw_v, seeds)
        pred_orig = inv(pred_log_avg)
        m = mape(y_va_orig, pred_orig)
        score = mape_to_score(m)
        oof_pred[va_idx] = pred_orig
        fold_metrics.append({"fold": fold.name, "mape": m, "score": score,
                             "n_train": len(tr_idx), "n_val": len(va_idx),
                             "best_iters": best_iters})
        fold_best_iters.extend(best_iters)
        logger.info(f"Fold {fold.name}: MAPE={m:.4f} score={score:.3f} best_iters={best_iters}")

    mean_mape = float(np.mean([f["mape"] for f in fold_metrics]))
    mean_score = float(np.mean([f["score"] for f in fold_metrics]))
    logger.info(f"CV mean MAPE = {mean_mape:.4f} | score = {mean_score:.3f}")

    # === FINAL TRAIN на всём ≤ feb-2025, predict march-2025 ===
    tr_idx, pred_idx = predict_split(df_feat, predict_year=2025, predict_month=3)
    tr_mask = ~y_train_all.iloc[tr_idx].isna().values
    tr_idx = tr_idx[tr_mask]
    X_tr = df_feat.loc[tr_idx, feat_cols]
    y_tr = y_train_all.iloc[tr_idx].values
    X_pr = df_feat.loc[pred_idx, feat_cols]
    sw = _maybe_weights(use_mape_weights, df_feat.loc[tr_idx, "rto"].values)

    # Подбираем num_boost из best_iters (медиана * 1.1)
    if fold_best_iters:
        nb_final = int(np.median(fold_best_iters) * 1.10)
        logger.info(f"Final num_boost_round = {nb_final}")
    else:
        nb_final = None

    pred_log_avg, _, last_model = _train_seed_bag(
        config["model"], config.get("params", {}),
        X_tr, y_tr, None, None, cat_features_in, sw, None, seeds,
        num_boost_override=nb_final)
    pred_orig = inv(pred_log_avg)

    submission = pd.DataFrame({
        "new_id": df_feat.loc[pred_idx, "store_id"].values.astype(int),
        "rto": pred_orig,
    }).sort_values("new_id").reset_index(drop=True)
    assert submission["new_id"].is_unique
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = Path("experiments/predictions") / f"{exp_name}_{ts}.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False)
    final_sub_path = Path("data/submissions") / f"{exp_name}_{ts}_test.csv"
    final_sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(final_sub_path, index=False)

    if config.get("save_oof", True):
        oof_path = Path("experiments/oof") / f"{exp_name}_{ts}_oof.parquet"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_df = df_feat[["store_id", "year", "month", "t", "rto"]].copy()
        oof_df["oof_pred"] = oof_pred
        oof_df.to_parquet(oof_path)

    if hasattr(last_model, "feature_importance"):
        fi = last_model.feature_importance()
        if fi is not None:
            fi_path = Path("experiments/reports") / f"{exp_name}_{ts}_fi.csv"
            fi_path.parent.mkdir(parents=True, exist_ok=True)
            fi.to_csv(fi_path)

    result = {
        "name": exp_name, "timestamp": ts, "model": config["model"],
        "params": config.get("params", {}),
        "feature_count": len(feat_cols),
        "cv_folds": fold_metrics,
        "cv_mean_mape": mean_mape, "cv_mean_score": mean_score,
        "submission_path": str(sub_path),
        "elapsed_seconds": time.time() - t0,
        "n_seeds": len(seeds),
    }
    save_json(result, Path("experiments/reports") / f"{exp_name}_{ts}.json")
    logger.info(f"Done in {result['elapsed_seconds']:.1f}s")
    try:
        save_pickle(last_model, Path("experiments/models") / f"{exp_name}_{ts}.pkl")
    except Exception as e:
        logger.warning(f"pickle failed: {e}")
    return result
