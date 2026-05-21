"""Optuna hyperopt. Использует обновлённый pipeline и кросс-валидацию.
Пример: python -m scripts.tune_optuna --trials 60 --timeout 28800 --model lightgbm
        python -m scripts.tune_optuna --model lightgbm --target_transform none --mape_weights"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import optuna
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_raw, add_target_row_for_march_2025
from src.data.target_encoding import add_target_encodings
from src.data.features import build_features
from src.validation.cv import default_folds
from src.validation.metrics import mape
from src.pipeline import get_model
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.utils.io import save_json


def build_dataset(train_path, winsorize_quantile=0.999):
    """Точная копия логики из pipeline._prepare, чтобы Optuna и боевой пайплайн
    видели идентичный набор фич, включая ETS и winsorize."""
    df = load_raw(train_path)
    df = add_target_row_for_march_2025(df)
    df = add_target_encodings(df, target="rto")
    df_feat, feat_cols, cat_features = build_features(df)
    df_feat = df_feat.reset_index(drop=True)

    ets_path = Path("data/processed/ets_features.parquet")
    if ets_path.exists():
        ets = __import__("pandas").read_parquet(ets_path)
        df_feat = df_feat.merge(ets, on=["store_id", "t"], how="left")
        df_feat = df_feat.reset_index(drop=True)
        if "ets_pred" in df_feat.columns and "ets_pred" not in feat_cols:
            feat_cols.append("ets_pred")
    else:
        print("WARN: ets_features.parquet not found — Optuna будет тюнить БЕЗ ETS!")

    # winsorize так же, как в боевом пайплайне
    cap = df_feat["rto"].quantile(winsorize_quantile)
    df_feat["_rto_train"] = df_feat["rto"].clip(upper=cap)
    return df_feat, feat_cols, cat_features


def _maybe_weights(use_mape, y_orig):
    if not use_mape:
        return None
    return (1.0 / np.maximum(np.abs(y_orig), 1.0)).astype(np.float64)


def evaluate(df_feat, feat_cols, cat_features, model_name, params, folds,
             target_transform="log1p", mape_weights=False):
    if target_transform == "log1p":
        y_all = np.log1p(df_feat["_rto_train"].astype(np.float64))
        inv = lambda yp: np.clip(np.expm1(yp), 1.0, None)
    else:
        y_all = df_feat["_rto_train"].astype(np.float64)
        inv = lambda yp: np.clip(yp, 1.0, None)
    fold_mapes = []
    for fold in folds:
        tr_idx, va_idx = fold.split(df_feat)
        tr_idx = tr_idx[~y_all.iloc[tr_idx].isna().values]
        va_idx = va_idx[~y_all.iloc[va_idx].isna().values]
        sw = _maybe_weights(mape_weights, df_feat.loc[tr_idx, "rto"].values)
        sw_v = _maybe_weights(mape_weights, df_feat.loc[va_idx, "rto"].values)
        m = get_model(model_name, params)
        m.fit(df_feat.loc[tr_idx, feat_cols], y_all.iloc[tr_idx].values,
              df_feat.loc[va_idx, feat_cols], y_all.iloc[va_idx].values,
              cat_features=cat_features, sample_weight=sw,
              sample_weight_val=sw_v, seed=2026)
        pred = inv(m.predict(df_feat.loc[va_idx, feat_cols]))
        fold_mapes.append(mape(df_feat.loc[va_idx, "rto"].values, pred))
    return float(np.mean(fold_mapes)), fold_mapes


def suggest_lgbm(trial, mape_weights=False):
    base = {
        "objective": "regression" if mape_weights else "regression_l1",
        "metric": "mape" if mape_weights else "mae",
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "num_boost_round": 8000, "early_stopping_rounds": 300,
        "num_threads": 4, "verbose": -1,
    }
    return base


def suggest_xgb(trial, **_):
    return {
        "objective": "reg:absoluteerror", "eval_metric": "mae",
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "tree_method": "hist", "nthread": 4,
        "num_boost_round": 8000, "early_stopping_rounds": 300,
    }


def suggest_cat(trial, **_):
    return {
        "loss_function": "MAE", "eval_metric": "MAE",
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "depth": trial.suggest_int("depth", 5, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
        "iterations": 6000, "od_type": "Iter", "od_wait": 300,
        "thread_count": 4, "verbose": False,
    }


SUGGEST = {"lightgbm": suggest_lgbm, "xgboost": suggest_xgb, "catboost": suggest_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SUGGEST), default="lightgbm")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--train", default="data/processed/v2.parquet")
    ap.add_argument("--study", default=None)
    ap.add_argument("--target_transform", choices=["log1p", "none"], default="log1p",
                    help="Тюним под ту же постановку, что в боевом конфиге.")
    ap.add_argument("--mape_weights", action="store_true",
                    help="Сэмпл-веса 1/y, как в lgbm_mape.yaml.")
    ap.add_argument("--winsorize_quantile", type=float, default=0.999)
    args = ap.parse_args()

    set_seed(2026)
    logger = get_logger(f"optuna_{args.model}")
    df_feat, feat_cols, cat_features = build_dataset(args.train,
                                                      winsorize_quantile=args.winsorize_quantile)
    folds = default_folds(df_feat)
    logger.info(f"Features: {len(feat_cols)}, folds: {[f.name for f in folds]}, "
                f"target_transform={args.target_transform}, mape_weights={args.mape_weights}")

    def objective(trial):
        params = SUGGEST[args.model](trial, mape_weights=args.mape_weights) \
            if args.model == "lightgbm" else SUGGEST[args.model](trial)
        t0 = time.time()
        m_val, fm = evaluate(df_feat, feat_cols, cat_features, args.model, params, folds,
                             target_transform=args.target_transform,
                             mape_weights=args.mape_weights)
        trial.set_user_attr("fold_mapes", fm)
        trial.set_user_attr("time_s", time.time() - t0)
        logger.info(f"trial {trial.number}: MAPE={m_val:.4f}  t={time.time()-t0:.1f}s")
        return m_val

    tag = f"{args.target_transform}{'_mw' if args.mape_weights else ''}"
    study_name = args.study or f"{args.model}_study_{tag}_v3"
    storage = f"sqlite:///experiments/{study_name}.db"
    study = optuna.create_study(direction="minimize", study_name=study_name,
                                storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=2026, multivariate=True))
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    logger.info(f"Best MAPE: {study.best_value:.4f}")
    save_json({"best_value": study.best_value, "best_params": study.best_params,
               "n_trials": len(study.trials),
               "target_transform": args.target_transform,
               "mape_weights": args.mape_weights},
              f"experiments/reports/optuna_{args.model}_{tag}_best.json")


if __name__ == "__main__":
    main()