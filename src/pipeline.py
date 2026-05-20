"""Главный пайплайн: загрузка → фичи → CV → fit-full → predict March 2025 → submission."""
from __future__ import annotations
import time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from .data.loader import load_raw, add_target_row_for_march_2025
from .data.features import build_features
from .validation.cv import default_folds, predict_split
from .validation.metrics import mape, mape_to_score
from .models.gbm import MODEL_REGISTRY
from .models.linear import LinearModel
from .models.mlp import MLPModel
from .utils.seed import set_seed
from .utils.logging import get_logger
from .utils.io import save_json, save_pickle


ALL_MODELS = {
    **MODEL_REGISTRY,
    "linear": LinearModel,
    "mlp": MLPModel,
}


def get_model(name: str, params: dict | None = None):
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(ALL_MODELS)}")
    return ALL_MODELS[name](params)


def run_experiment(config: dict, train_path: str = "data/raw/train_2.csv") -> dict:
    """
    config:
        name: str — название эксперимента
        model: str — lightgbm/xgboost/catboost/linear/mlp
        params: dict
        target_transform: 'log1p' | 'none'
        feature_set: 'default'
        cv_val_months: list[[year, month]] | null
        save_oof: bool
        seed: int
    """
    set_seed(config.get("seed", 2026))
    exp_name = config["name"]
    logger = get_logger(exp_name, log_dir="experiments/logs")
    logger.info(f"=== Experiment: {exp_name} ===")
    logger.info(f"Config: {json.dumps(config, ensure_ascii=False)}")

    t0 = time.time()
    df = load_raw(train_path)
    df = add_target_row_for_march_2025(df)
    logger.info(f"Loaded raw: {df.shape}, time={(time.time()-t0):.1f}s")

    df_feat, feat_cols, cat_features = build_features(df, target="rto")
    logger.info(f"Built features: {len(feat_cols)} features, time={(time.time()-t0):.1f}s")

    # Лог-трансформ
    target_transform = config.get("target_transform", "log1p")
    if target_transform == "log1p":
        y_all = np.log1p(df_feat["rto"].astype(np.float64))
        inv = lambda yp: np.clip(np.expm1(yp), 1.0, None)
    else:
        y_all = df_feat["rto"].astype(np.float64)
        inv = lambda yp: np.clip(yp, 1.0, None)

    df_feat = df_feat.reset_index(drop=True)

    cv_val_months = config.get("cv_val_months")
    if cv_val_months is not None:
        cv_val_months = [tuple(x) for x in cv_val_months]
    folds = default_folds(df_feat, val_months=cv_val_months)
    logger.info(f"CV folds: {[f.name for f in folds]}")

    fold_metrics = []
    oof_pred = np.full(len(df_feat), np.nan, dtype=np.float64)

    for fold in folds:
        tr_idx, va_idx = fold.split(df_feat)
        # выкидываем строки без таргета
        tr_mask = ~y_all.iloc[tr_idx].isna().values
        va_mask = ~y_all.iloc[va_idx].isna().values
        tr_idx = tr_idx[tr_mask]; va_idx = va_idx[va_mask]
        X_tr = df_feat.loc[tr_idx, feat_cols]
        y_tr = y_all.iloc[tr_idx].values
        X_va = df_feat.loc[va_idx, feat_cols]
        y_va_log = y_all.iloc[va_idx].values
        y_va_orig = df_feat.loc[va_idx, "rto"].values

        model = get_model(config["model"], config.get("params", {}))
        model.fit(X_tr, y_tr, X_va, y_va_log,
                  cat_features=cat_features if config.get("use_cat", True) else None)
        pred_log = model.predict(X_va)
        pred_orig = inv(pred_log)
        m = mape(y_va_orig, pred_orig)
        score = mape_to_score(m)
        oof_pred[va_idx] = pred_orig
        fold_metrics.append({"fold": fold.name, "mape": m, "score": score,
                             "n_train": len(tr_idx), "n_val": len(va_idx)})
        logger.info(f"Fold {fold.name}: MAPE={m:.4f} | score={score:.3f} | train={len(tr_idx)} val={len(va_idx)}")

    mean_mape = float(np.mean([f["mape"] for f in fold_metrics]))
    mean_score = float(np.mean([f["score"] for f in fold_metrics]))
    logger.info(f"CV mean MAPE = {mean_mape:.4f}, mean score = {mean_score:.3f}")

    # === FINAL TRAIN: всё до марта 2025, предсказываем март 2025 ===
    tr_idx, pred_idx = predict_split(df_feat, predict_year=2025, predict_month=3)
    tr_mask = ~y_all.iloc[tr_idx].isna().values
    tr_idx = tr_idx[tr_mask]
    logger.info(f"Final train: {len(tr_idx)} rows, predict: {len(pred_idx)} stores")

    X_tr = df_feat.loc[tr_idx, feat_cols]
    y_tr = y_all.iloc[tr_idx].values
    X_pr = df_feat.loc[pred_idx, feat_cols]

    final_model = get_model(config["model"], config.get("params", {}))
    final_model.fit(X_tr, y_tr, cat_features=cat_features if config.get("use_cat", True) else None)
    pred_log = final_model.predict(X_pr)
    pred_orig = inv(pred_log)

    # Сабмишн
    submission = pd.DataFrame({
        "new_id": df_feat.loc[pred_idx, "store_id"].values.astype(int),
        "rto": pred_orig,
    })
    # Уникальность new_id
    assert submission["new_id"].is_unique, "new_id is not unique in submission!"
    # Сортируем как в sample (если есть) — иначе по new_id
    submission = submission.sort_values("new_id").reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = Path("experiments/predictions") / f"{exp_name}_{ts}.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False)
    # И финальный test.csv в data/submissions
    final_sub_path = Path("data/submissions") / f"{exp_name}_{ts}_test.csv"
    final_sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(final_sub_path, index=False)
    logger.info(f"Saved submission: {sub_path} and {final_sub_path}")

    # OOF
    if config.get("save_oof", True):
        oof_path = Path("experiments/oof") / f"{exp_name}_{ts}_oof.parquet"
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_df = df_feat[["store_id", "year", "month", "t", "rto"]].copy()
        oof_df["oof_pred"] = oof_pred
        oof_df.to_parquet(oof_path)
        logger.info(f"Saved OOF: {oof_path}")

    # Feature importance
    fi = final_model.feature_importance() if hasattr(final_model, "feature_importance") else None
    if fi is not None:
        fi_path = Path("experiments/reports") / f"{exp_name}_{ts}_fi.csv"
        fi_path.parent.mkdir(parents=True, exist_ok=True)
        fi.to_csv(fi_path)
        logger.info(f"Top-15 features:\n{fi.head(15)}")

    # Финальный summary
    result = {
        "name": exp_name,
        "timestamp": ts,
        "model": config["model"],
        "params": config.get("params", {}),
        "feature_count": len(feat_cols),
        "cv_folds": fold_metrics,
        "cv_mean_mape": mean_mape,
        "cv_mean_score": mean_score,
        "submission_path": str(sub_path),
        "elapsed_seconds": time.time() - t0,
    }
    save_json(result, Path("experiments/reports") / f"{exp_name}_{ts}.json")
    logger.info(f"Experiment finished in {result['elapsed_seconds']:.1f}s")

    # Сохраняем также модель
    try:
        save_pickle(final_model, Path("experiments/models") / f"{exp_name}_{ts}.pkl")
    except Exception as e:
        logger.warning(f"Could not pickle model: {e}")

    return result
