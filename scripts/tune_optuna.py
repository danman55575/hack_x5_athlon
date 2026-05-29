"""Optuna-подбор гиперпараметров на текущем боевом пайплайне.

Скрипт не строит отдельный контур обучения, а использует те же:
- `_prepare` из `src.pipeline` для сборки фичей;
- временные фолды из `src.validation.cv`;
- MAPE на исходной шкале РТО;
- конфиги моделей из `configs/`.

Что умеет:
1. сравнивать LightGBM, XGBoost и CatBoost в одном запуске;
2. быстро отрезать слабые trial через pruning по промежуточным fold-метрикам;
3. сохранять лучшие параметры, merged YAML-конфиг и таблицу trial;
4. работать на уже выбранном allowlist фичей для ускорения и честного сравнения моделей.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import (
    _compute_target_clip_bounds,
    _is_ratio_target,
    _maybe_weights,
    _prepare,
    _resolve_feature_allowlist,
    get_model,
)
from src.utils.io import load_yaml, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.validation.cv import default_folds
from src.validation.metrics import mape


REPORTS_DIR = Path("experiments/reports")
OPTUNA_DIR = Path("experiments/optuna")
DEFAULT_CONFIG_MAP = {
    "lightgbm": Path("configs/lgbm.yaml"),
    "xgboost": Path("configs/xgb.yaml"),
    "catboost": Path("configs/catboost.yaml"),
}


@dataclass
class ModelTuneContext:
    model_name: str
    config_path: Path
    config: dict[str, Any]
    feature_allowlist_path: str | None
    df_feat: pd.DataFrame
    feat_cols: list[str]
    cat_features: list[str]
    y_train_all: pd.Series
    inv: Any
    folds: list
    fold_weights: list[float]


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_fold_weights(weights: list[float] | None, n_folds: int) -> list[float]:
    out = list(weights or [])
    out = out[:n_folds]
    while len(out) < n_folds:
        out.append(1.0)
    return out


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights[: len(values)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if w.sum() <= 0:
        return float(arr.mean())
    return float(np.average(arr, weights=w))


def _load_tuning_context(
    model_name: str,
    config_path: Path,
    train_path: str,
    feature_allowlist_path: str | None,
) -> ModelTuneContext:
    cfg = load_yaml(config_path)
    allowlist = _resolve_feature_allowlist(feature_allowlist_path=feature_allowlist_path)
    df_feat, feat_cols, cat_features, y_train_all, inv, audit = _prepare(
        train_path,
        target_transform=cfg.get("target_transform", "log1p"),
        winsorize_quantile=float(cfg.get("winsorize_quantile", 0.999)),
        feature_groups=cfg.get("feature_groups"),
        feature_allowlist=allowlist,
    )
    val_months = [tuple(item) for item in cfg.get("cv_val_months", [])] or None
    folds = default_folds(df_feat, val_months=val_months)
    fold_weights = _normalize_fold_weights(cfg.get("cv_fold_weights"), len(folds))
    logger = get_logger(f"optuna_{model_name}")
    logger.info(
        "Подготовлен датасет для тюнинга: rows=%s, features=%s, total_nan=%s, all_nan=%s",
        len(df_feat),
        len(feat_cols),
        audit["total_nan"],
        len(audit["all_nan_columns"]),
    )
    return ModelTuneContext(
        model_name=model_name,
        config_path=config_path,
        config=cfg,
        feature_allowlist_path=feature_allowlist_path,
        df_feat=df_feat,
        feat_cols=feat_cols,
        cat_features=cat_features,
        y_train_all=y_train_all,
        inv=inv,
        folds=folds,
        fold_weights=fold_weights,
    )


def _base_param_template(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    params = dict(config.get("params", {}))
    if model_name == "lightgbm":
        params.setdefault("objective", "regression_l1")
        params.setdefault("metric", "mae")
    elif model_name == "xgboost":
        params.setdefault("objective", "reg:absoluteerror")
        params.setdefault("eval_metric", "mae")
    elif model_name == "catboost":
        params.setdefault("loss_function", "MAE")
        params.setdefault("eval_metric", "MAE")
    return params


def _suggest_lightgbm(trial: optuna.Trial, base: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    params = dict(base)
    params.update(
        {
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 220),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 20.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 20.0, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 5, 6, 7, 8, 10, 12]),
            "bagging_freq": 1,
            "num_boost_round": int(args.tune_num_boost_round),
            "early_stopping_rounds": int(args.tune_early_stopping_rounds),
            "num_threads": int(args.threads),
            "verbose": -1,
        }
    )
    return params


def _suggest_xgboost(trial: optuna.Trial, base: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    params = dict(base)
    params.update(
        {
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 64.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            "tree_method": "hist",
            "nthread": int(args.threads),
            "num_boost_round": int(args.tune_num_boost_round),
            "early_stopping_rounds": int(args.tune_early_stopping_rounds),
        }
    )
    return params


def _suggest_catboost(trial: optuna.Trial, base: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    params = dict(base)
    params.update(
        {
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "border_count": trial.suggest_int("border_count", 64, 255),
            "iterations": int(args.tune_num_boost_round),
            "od_type": "Iter",
            "od_wait": int(args.tune_early_stopping_rounds),
            "thread_count": int(args.threads),
            "verbose": False,
        }
    )
    return params


SUGGESTERS = {
    "lightgbm": _suggest_lightgbm,
    "xgboost": _suggest_xgboost,
    "catboost": _suggest_catboost,
}

SEARCH_PARAM_KEYS = {
    "lightgbm": [
        "learning_rate",
        "num_leaves",
        "min_data_in_leaf",
        "feature_fraction",
        "bagging_fraction",
        "lambda_l1",
        "lambda_l2",
        "max_depth",
    ],
    "xgboost": [
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "gamma",
    ],
    "catboost": [
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "random_strength",
        "bagging_temperature",
        "border_count",
    ],
}

SEARCH_DEFAULTS = {
    "lightgbm": {
        "max_depth": -1,
    },
    "xgboost": {
        "gamma": 1e-4,
    },
    "catboost": {
        "bagging_temperature": 0.0,
        "border_count": 254,
    },
}


def _build_trial_params(model_name: str, config: dict[str, Any], trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    base = _base_param_template(model_name, config)
    return SUGGESTERS[model_name](trial, base, args)


def _current_config_trial_params(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    params = dict(config.get("params", {}))
    allowed = SEARCH_PARAM_KEYS[model_name]
    baseline = {}
    for key in allowed:
        if key in params:
            value = params[key]
            if model_name == "lightgbm" and key == "max_depth" and int(value) == 0:
                value = -1
            baseline[key] = value
        elif key in SEARCH_DEFAULTS.get(model_name, {}):
            baseline[key] = SEARCH_DEFAULTS[model_name][key]
    return baseline


def _evaluate_trial(
    trial: optuna.Trial,
    ctx: ModelTuneContext,
    params: dict[str, Any],
    args: argparse.Namespace,
    logger,
) -> float:
    fold_mapes: list[float] = []
    fold_iters: list[int] = []
    use_cat = bool(ctx.config.get("use_cat", False))
    cat_features = ctx.cat_features if use_cat else None
    target_transform = ctx.config.get("target_transform", "log1p")
    seeds = [int(ctx.config.get("seed", 2026))]

    for fold_idx, fold in enumerate(ctx.folds):
        tr_idx, va_idx = fold.split(ctx.df_feat)
        tr_idx = tr_idx[~ctx.y_train_all.iloc[tr_idx].isna().values]
        va_idx = va_idx[~ctx.y_train_all.iloc[va_idx].isna().values]

        X_tr = ctx.df_feat.loc[tr_idx, ctx.feat_cols]
        X_va = ctx.df_feat.loc[va_idx, ctx.feat_cols]
        y_tr = ctx.y_train_all.iloc[tr_idx].values.astype(np.float64)
        y_va = ctx.y_train_all.iloc[va_idx].values.astype(np.float64)
        sw = _maybe_weights(bool(ctx.config.get("mape_weights", False)), ctx.df_feat.loc[tr_idx, "rto"].values)
        clip_bounds = None
        if _is_ratio_target(target_transform):
            clip_bounds = _compute_target_clip_bounds(y_tr)
            if clip_bounds is not None:
                y_tr = np.clip(y_tr, clip_bounds[0], clip_bounds[1])

        model = get_model(ctx.model_name, dict(params))
        model.fit(
            X_tr,
            y_tr,
            X_va,
            y_va,
            cat_features=cat_features,
            sample_weight=sw,
            sample_weight_val=None,
            seed=seeds[0],
        )
        pred = model.predict(X_va)
        if target_transform == "log1p":
            pred = np.clip(np.expm1(pred), 1.0, None)
        elif _is_ratio_target(target_transform):
            pred = ctx.inv(pred, rows=ctx.df_feat.loc[va_idx], clip_bounds=clip_bounds)
        else:
            pred = np.clip(pred, 1.0, None)

        fold_mape = float(mape(ctx.df_feat.loc[va_idx, "rto"].values, pred))
        fold_mapes.append(fold_mape)
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is not None:
            fold_iters.append(int(best_iter))

        interim = _weighted_mean(fold_mapes, ctx.fold_weights)
        trial.report(interim, step=fold_idx)
        trial.set_user_attr("last_fold", fold.name)
        trial.set_user_attr("fold_mapes", fold_mapes.copy())
        trial.set_user_attr("fold_iters", fold_iters.copy())

        completed = trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        if completed and fold_idx + 1 >= int(args.prune_after_folds):
            best_value = min(item.value for item in completed if item.value is not None)
            if np.isfinite(best_value) and interim > best_value * float(args.prune_margin):
                raise TrialPruned(
                    f"Промежуточный weighted MAPE {interim:.4f} хуже лучшего {best_value:.4f} "
                    f"с запасом {args.prune_margin:.2f} после fold {fold.name}."
                )

        if trial.should_prune():
            raise TrialPruned(f"Optuna pruner остановил trial после fold {fold.name}.")

    weighted_mape = _weighted_mean(fold_mapes, ctx.fold_weights)
    trial.set_user_attr("weighted_mape", weighted_mape)
    trial.set_user_attr("mean_mape", float(np.mean(fold_mapes)))
    return weighted_mape


def _study_name(model_name: str, config_path: Path, feature_allowlist_path: str | None, suffix: str | None) -> str:
    feature_tag = Path(feature_allowlist_path).stem if feature_allowlist_path else "all_features"
    parts = [model_name, config_path.stem, feature_tag]
    if suffix:
        parts.append(suffix)
    return "_".join(parts)


def _build_pruner(args: argparse.Namespace) -> optuna.pruners.BasePruner:
    if args.pruner == "none":
        return optuna.pruners.NopPruner()
    if args.pruner == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(args.pruner_startup_trials),
            n_warmup_steps=max(0, int(args.prune_after_folds) - 1),
            interval_steps=1,
        )
    if args.pruner == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=float(args.pruner_percentile),
            n_startup_trials=int(args.pruner_startup_trials),
            n_warmup_steps=max(0, int(args.prune_after_folds) - 1),
            interval_steps=1,
        )
    raise ValueError(f"Неизвестный pruner: {args.pruner}")


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _merge_best_config(
    ctx: ModelTuneContext,
    best_params: dict[str, Any],
    keep_full_train_rounds: bool,
) -> dict[str, Any]:
    merged = dict(ctx.config)
    merged_params = dict(merged.get("params", {}))
    merged_params.update(best_params)

    if keep_full_train_rounds:
        base_params = ctx.config.get("params", {})
        if ctx.model_name in {"lightgbm", "xgboost"}:
            for key in ("num_boost_round", "early_stopping_rounds"):
                if key in base_params:
                    merged_params[key] = base_params[key]
        elif ctx.model_name == "catboost":
            for key in ("iterations", "od_type", "od_wait"):
                if key in base_params:
                    merged_params[key] = base_params[key]

    merged["params"] = merged_params
    if ctx.feature_allowlist_path:
        merged["feature_allowlist_path"] = ctx.feature_allowlist_path
        merged["name"] = f"{merged['name']}_{Path(ctx.feature_allowlist_path).stem}"
    return merged


def _study_trials_frame(study: optuna.Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": str(trial.state).split(".")[-1],
            "value": trial.value,
            "duration_s": trial.duration.total_seconds() if trial.duration is not None else None,
            "fold_mapes": trial.user_attrs.get("fold_mapes"),
            "fold_iters": trial.user_attrs.get("fold_iters"),
            "last_fold": trial.user_attrs.get("last_fold"),
        }
        row.update(trial.params)
        rows.append(row)
    return pd.DataFrame(rows)


def _run_study_for_model(
    model_name: str,
    config_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ctx = _load_tuning_context(model_name, config_path, args.train, args.feature_list_path)
    study_name = (
        f"{args.study_name}_{model_name}"
        if args.study_name
        else _study_name(model_name, config_path, args.feature_list_path, args.tag)
    )
    storage = f"sqlite:///{(OPTUNA_DIR / f'{study_name}.db').as_posix()}"
    logger = get_logger(f"optuna_{model_name}")
    logger.info(
        "Старт тюнинга: model=%s, config=%s, trials=%s, timeout=%s, pruner=%s, feature_allowlist=%s",
        model_name,
        config_path.as_posix(),
        args.trials,
        args.timeout,
        args.pruner,
        args.feature_list_path,
    )

    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True)
    pruner = _build_pruner(args)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    if args.enqueue_current_config and not study.trials:
        current_params = _current_config_trial_params(model_name, ctx.config)
        if current_params:
            study.enqueue_trial(current_params, user_attrs={"source": "current_config"})
            logger.info("В очередь добавлен baseline trial из текущего YAML: %s", current_params)

    def objective(trial: optuna.Trial) -> float:
        params = _build_trial_params(model_name, ctx.config, trial, args)
        started_at = time.time()
        value = _evaluate_trial(trial, ctx, params, args, logger)
        trial.set_user_attr("time_s", time.time() - started_at)
        logger.info(
            "trial=%s model=%s weighted_mape=%.4f folds=%s",
            trial.number,
            model_name,
            value,
            [f"{item:.4f}" for item in trial.user_attrs.get("fold_mapes", [])],
        )
        return value

    study.optimize(objective, n_trials=int(args.trials), timeout=args.timeout, gc_after_trial=True)

    best_params = dict(study.best_params)
    best_trial = study.best_trial
    best_json_path = REPORTS_DIR / f"optuna_{study_name}_best.json"
    best_yaml_path = REPORTS_DIR / f"optuna_{study_name}_best_config.yaml"
    trials_csv_path = REPORTS_DIR / f"optuna_{study_name}_trials.csv"

    merged_cfg = _merge_best_config(ctx, best_params, keep_full_train_rounds=not args.export_tuning_rounds)
    merged_cfg["optuna_meta"] = {
        "study_name": study_name,
        "best_value": float(study.best_value),
        "best_trial": int(best_trial.number),
        "feature_list_path": args.feature_list_path,
        "config_path": config_path.as_posix(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    trials_df = _study_trials_frame(study)
    trials_df.to_csv(trials_csv_path, index=False, encoding="utf-8")
    _dump_yaml(best_yaml_path, merged_cfg)
    save_json(
        {
            "model": model_name,
            "study_name": study_name,
            "best_value": float(study.best_value),
            "best_trial": int(best_trial.number),
            "best_params": best_params,
            "feature_list_path": args.feature_list_path,
            "config_path": config_path.as_posix(),
            "fold_mapes": best_trial.user_attrs.get("fold_mapes"),
            "fold_iters": best_trial.user_attrs.get("fold_iters"),
            "n_trials_total": len(study.trials),
            "n_trials_complete": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
            "n_trials_pruned": len([t for t in study.trials if t.state == TrialState.PRUNED]),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        best_json_path,
    )

    logger.info(
        "Лучшая модель %s: weighted_mape=%.4f, trial=%s, pruned=%s/%s",
        model_name,
        study.best_value,
        best_trial.number,
        len([t for t in study.trials if t.state == TrialState.PRUNED]),
        len(study.trials),
    )

    return {
        "model": model_name,
        "study_name": study_name,
        "best_value": float(study.best_value),
        "best_trial": int(best_trial.number),
        "config_path": config_path.as_posix(),
        "feature_list_path": args.feature_list_path,
        "best_json_path": best_json_path.as_posix(),
        "best_yaml_path": best_yaml_path.as_posix(),
        "trials_csv_path": trials_csv_path.as_posix(),
        "n_trials_total": len(study.trials),
        "n_trials_complete": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        "n_trials_pruned": len([t for t in study.trials if t.state == TrialState.PRUNED]),
        "fold_mapes": best_trial.user_attrs.get("fold_mapes"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(DEFAULT_CONFIG_MAP),
        default=["lightgbm", "xgboost", "catboost"],
        help="Какие модели тюнить и сравнивать.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Кастомный YAML-конфиг для тюнинга одной модели. Если задан, model берётся из самого YAML.",
    )
    parser.add_argument("--train", default="data/processed/v2.parquet")
    parser.add_argument(
        "--feature-list-path",
        default="experiments/reports/feature_study_selected_features.csv",
        help="Путь к allowlist фичей. Если файла нет, будут использованы все доступные фичи.",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tune-num-boost-round", type=int, default=4000)
    parser.add_argument("--tune-early-stopping-rounds", type=int, default=200)
    parser.add_argument("--pruner", choices=["none", "median", "percentile"], default="median")
    parser.add_argument("--pruner-startup-trials", type=int, default=5)
    parser.add_argument("--pruner-percentile", type=float, default=50.0)
    parser.add_argument("--prune-after-folds", type=int, default=2)
    parser.add_argument("--prune-margin", type=float, default=1.15)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--enqueue-current-config",
        action="store_true",
        help="Сначала прогнать текущие параметры из YAML как baseline trial.",
    )
    parser.add_argument(
        "--export-tuning-rounds",
        action="store_true",
        help="Сохранять в exported YAML укороченные tuning rounds вместо боевых rounds из исходного конфига.",
    )
    return parser.parse_args()


def _resolve_run_targets(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.config_path:
        config_path = Path(args.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Не найден config-path: {config_path}")
        cfg = load_yaml(config_path)
        model_name = str(cfg.get("model", "")).strip().lower()
        if model_name not in DEFAULT_CONFIG_MAP:
            raise ValueError(
                f"В config-path должен быть model из {list(DEFAULT_CONFIG_MAP)}, "
                f"получено: {cfg.get('model')}"
            )
        if args.models != ["lightgbm", "xgboost", "catboost"]:
            print("Использую model из --config-path, аргумент --models будет проигнорирован.")
        return [(model_name, config_path)]

    return [(model_name, DEFAULT_CONFIG_MAP[model_name]) for model_name in args.models]


def main() -> None:
    args = _parse_args()
    _ensure_dirs()
    if args.feature_list_path and not Path(args.feature_list_path).exists():
        print(f"Файл allowlist не найден, продолжаю без него: {args.feature_list_path}")
        args.feature_list_path = None

    set_seed(int(args.seed))
    results: list[dict[str, Any]] = []
    for model_name, config_path in _resolve_run_targets(args):
        results.append(_run_study_for_model(model_name, config_path, args))

    summary = pd.DataFrame(results).sort_values("best_value", ascending=True).reset_index(drop=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = args.tag or timestamp
    summary_path = REPORTS_DIR / f"optuna_model_compare_{suffix}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    best_row = summary.iloc[0].to_dict()
    save_json(
        {
            "best_model": best_row["model"],
            "best_value": best_row["best_value"],
            "best_yaml_path": best_row["best_yaml_path"],
            "summary_path": summary_path.as_posix(),
            "models": results,
        },
        REPORTS_DIR / f"optuna_model_compare_{suffix}.json",
    )

    print("Тюнинг завершён.")
    print(f"Лучшая модель: {best_row['model']}  weighted_mape={best_row['best_value']:.4f}")
    print(f"Сводная таблица: {summary_path.as_posix()}")
    print(f"Рекомендуемый YAML: {best_row['best_yaml_path']}")
    print("Команда полного прогона лучшего конфига:")
    print(f"uv run python scripts/run_experiment.py --config {best_row['best_yaml_path']}")


if __name__ == "__main__":
    main()
