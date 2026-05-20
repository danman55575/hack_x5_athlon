"""Optuna hyperopt для LightGBM (по умолчанию). Использует CV из pipeline.
Пример: python -m scripts.tune_optuna --trials 60 --timeout 14400 --model lightgbm
"""
import argparse, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_raw, add_target_row_for_march_2025
from src.data.features import build_features
from src.validation.cv import default_folds
from src.validation.metrics import mape
from src.pipeline import get_model
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.utils.io import save_json


def build_dataset(train_path):
    df = load_raw(train_path)
    df = add_target_row_for_march_2025(df)
    df_feat, feat_cols, cat_features = build_features(df)
    df_feat = df_feat.reset_index(drop=True)
    return df_feat, feat_cols, cat_features


def evaluate_params(df_feat, feat_cols, cat_features, model_name, params, folds, log_target=True):
    if log_target:
        y_all = np.log1p(df_feat["rto"].astype(np.float64))
        inv = lambda yp: np.clip(np.expm1(yp), 1.0, None)
    else:
        y_all = df_feat["rto"].astype(np.float64)
        inv = lambda yp: np.clip(yp, 1.0, None)

    fold_mapes = []
    for fold in folds:
        tr_idx, va_idx = fold.split(df_feat)
        tr_mask = ~y_all.iloc[tr_idx].isna().values
        va_mask = ~y_all.iloc[va_idx].isna().values
        tr_idx = tr_idx[tr_mask]; va_idx = va_idx[va_mask]
        X_tr = df_feat.loc[tr_idx, feat_cols]
        y_tr = y_all.iloc[tr_idx].values
        X_va = df_feat.loc[va_idx, feat_cols]
        y_va_orig = df_feat.loc[va_idx, "rto"].values
        m = get_model(model_name, params)
        m.fit(X_tr, y_tr, X_va, y_all.iloc[va_idx].values, cat_features=cat_features)
        pred = inv(m.predict(X_va))
        fold_mapes.append(mape(y_va_orig, pred))
    return float(np.mean(fold_mapes)), fold_mapes


def suggest_lgbm(trial):
    return {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "num_boost_round": 6000,
        "early_stopping_rounds": 200,
        "num_threads": 4,
        "verbose": -1,
    }


def suggest_xgb(trial):
    return {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "tree_method": "hist",
        "nthread": 4,
        "num_boost_round": 6000,
        "early_stopping_rounds": 200,
    }


def suggest_cat(trial):
    return {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "depth": trial.suggest_int("depth", 5, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
        "iterations": 5000,
        "od_type": "Iter",
        "od_wait": 250,
        "thread_count": 4,
        "verbose": False,
    }


SUGGEST = {"lightgbm": suggest_lgbm, "xgboost": suggest_xgb, "catboost": suggest_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SUGGEST), default="lightgbm")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=None, help="seconds")
    ap.add_argument("--train", default="data/raw/train_2.csv")
    ap.add_argument("--study", default=None)
    args = ap.parse_args()

    set_seed(2026)
    logger = get_logger(f"optuna_{args.model}")
    df_feat, feat_cols, cat_features = build_dataset(args.train)
    folds = default_folds(df_feat)
    logger.info(f"Features: {len(feat_cols)}, folds: {[f.name for f in folds]}")

    def objective(trial: optuna.Trial):
        params = SUGGEST[args.model](trial)
        t0 = time.time()
        m, fmapes = evaluate_params(df_feat, feat_cols, cat_features, args.model, params, folds)
        trial.set_user_attr("fold_mapes", fmapes)
        trial.set_user_attr("time_s", time.time() - t0)
        logger.info(f"trial {trial.number}: MAPE={m:.4f}  folds={fmapes}  t={time.time()-t0:.1f}s")
        return m

    study_name = args.study or f"{args.model}_study"
    storage = f"sqlite:///experiments/{study_name}.db"
    study = optuna.create_study(direction="minimize", study_name=study_name,
                                storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=2026, multivariate=True))
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    logger.info(f"Best MAPE: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    save_json({"best_value": study.best_value, "best_params": study.best_params,
               "n_trials": len(study.trials)},
              f"experiments/reports/optuna_{args.model}_best.json")


if __name__ == "__main__":
    main()
