from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from .base import BaseModel


def _prepare_cat_columns_for_catboost(X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    """CatBoost требует, чтобы категориальные колонки были строками или неотрицательными int.
    Преобразуем category dtype → string, NaN → '__NA__'.
    """
    if not cat_features:
        return X
    X = X.copy()
    for col in cat_features:
        if col not in X.columns:
            continue
        s = X[col]
        if pd.api.types.is_categorical_dtype(s):
            s = s.astype(str)
        elif s.dtype.kind in "iuf":
            # Если числовой, шифтуем коды от 0 (на случай -1 от cat.codes для NaN).
            s = s.astype(str)
        else:
            s = s.astype(str)
        s = s.replace({"nan": "__NA__", "NaN": "__NA__", "<NA>": "__NA__", "-1": "__NA__"})
        s = s.fillna("__NA__")
        X[col] = s
    return X


class LightGBMModel(BaseModel):
    name = "lightgbm"
    default_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "num_threads": 4,
    }

    def fit(self, X, y, X_val=None, y_val=None, cat_features=None,
            sample_weight=None, sample_weight_val=None, seed=None):
        params = {**self.default_params, **self.params}
        if seed is not None:
            params["seed"] = int(seed)
            params["bagging_seed"] = int(seed)
            params["feature_fraction_seed"] = int(seed)
        n_rounds = params.pop("num_boost_round", 5000)
        early_stop = params.pop("early_stopping_rounds", 200)

        if cat_features:
            dtrain = lgb.Dataset(X, label=y, categorical_feature=cat_features,
                                 weight=sample_weight)
        else:
            dtrain = lgb.Dataset(X, label=y, weight=sample_weight)

        valid_sets, valid_names = [dtrain], ["train"]
        callbacks = [lgb.log_evaluation(period=0)]
        if X_val is not None:
            if cat_features:
                dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features,
                                   reference=dtrain, weight=sample_weight_val)
            else:
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain,
                                   weight=sample_weight_val)
            valid_sets.append(dval); valid_names.append("valid")
            callbacks.insert(0, lgb.early_stopping(early_stop, verbose=False))
        self.model_ = lgb.train(params, dtrain, num_boost_round=n_rounds,
                                valid_sets=valid_sets, valid_names=valid_names,
                                callbacks=callbacks)
        self.best_iteration_ = getattr(self.model_, "best_iteration", n_rounds) or n_rounds
        return self

    def predict(self, X):
        return self.model_.predict(X, num_iteration=getattr(self.model_, "best_iteration", None))

    def feature_importance(self):
        if self.model_ is None: return None
        return pd.Series(self.model_.feature_importance(importance_type="gain"),
                         index=self.model_.feature_name()).sort_values(ascending=False)


class XGBoostModel(BaseModel):
    name = "xgboost"
    default_params = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "nthread": 4,
    }

    def fit(self, X, y, X_val=None, y_val=None, cat_features=None,
            sample_weight=None, sample_weight_val=None, seed=None):
        params = {**self.default_params, **self.params}
        if seed is not None:
            params["seed"] = int(seed)
        n_rounds = params.pop("num_boost_round", 5000)
        early_stop = params.pop("early_stopping_rounds", 200)

        has_pd_cats = False
        if hasattr(X, "dtypes"):
            try:
                has_pd_cats = any(str(t) == "category" for t in X.dtypes)
            except Exception:
                has_pd_cats = False
        enable_cat = bool(has_pd_cats)
        self._enable_categorical = enable_cat
        params["enable_categorical"] = enable_cat

        dtrain = xgb.DMatrix(X, label=y, enable_categorical=enable_cat, weight=sample_weight)
        evals = [(dtrain, "train")]
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=enable_cat,
                               weight=sample_weight_val)
            evals.append((dval, "valid"))
            self.model_ = xgb.train(params, dtrain, num_boost_round=n_rounds,
                                    evals=evals, early_stopping_rounds=early_stop,
                                    verbose_eval=False)
        else:
            self.model_ = xgb.train(params, dtrain, num_boost_round=n_rounds,
                                    evals=evals, verbose_eval=False)
        self.best_iteration_ = getattr(self.model_, "best_iteration", n_rounds) or n_rounds
        return self

    def predict(self, X):
        enable_cat = getattr(self, "_enable_categorical", False)
        d = xgb.DMatrix(X, enable_categorical=enable_cat)
        it = getattr(self.model_, "best_iteration", None)
        if it is not None:
            return self.model_.predict(d, iteration_range=(0, it + 1))
        return self.model_.predict(d)


class CatBoostModel(BaseModel):
    name = "catboost"
    default_params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "iterations": 5000,
        "od_type": "Iter",
        "od_wait": 200,
        "thread_count": 4,
        "verbose": False,
    }

    def fit(self, X, y, X_val=None, y_val=None, cat_features=None,
            sample_weight=None, sample_weight_val=None, seed=None):
        params = {**self.default_params, **self.params}
        if seed is not None:
            params["random_seed"] = int(seed)
        if X_val is None:
            for k in ("od_type", "od_wait", "od_pval"):
                params.pop(k, None)
        cat_features = cat_features or []

        X_cb = _prepare_cat_columns_for_catboost(X, cat_features)
        X_val_cb = _prepare_cat_columns_for_catboost(X_val, cat_features) if X_val is not None else None

        train_pool = Pool(X_cb, y, cat_features=cat_features, weight=sample_weight)
        eval_pool = (Pool(X_val_cb, y_val, cat_features=cat_features, weight=sample_weight_val)
                     if X_val_cb is not None else None)
        self.model_ = CatBoostRegressor(**params)
        self.model_.fit(train_pool, eval_set=eval_pool,
                        use_best_model=eval_pool is not None)
        self.best_iteration_ = self.model_.get_best_iteration() or params.get("iterations")
        self._cat_features = cat_features
        return self

    def predict(self, X):
        cat_features = getattr(self, "_cat_features", []) or []
        X_cb = _prepare_cat_columns_for_catboost(X, cat_features) if cat_features else X
        return self.model_.predict(X_cb)

    def feature_importance(self):
        if self.model_ is None: return None
        return pd.Series(self.model_.get_feature_importance(),
                         index=self.model_.feature_names_).sort_values(ascending=False)


MODEL_REGISTRY = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
}
