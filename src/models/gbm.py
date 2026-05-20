from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from .base import BaseModel


class LightGBMModel(BaseModel):
    name = "lightgbm"
    default_params = {
        "objective": "regression_l1",   # L1 на лог-таргете ≈ MAPE
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

    def fit(self, X, y, X_val=None, y_val=None, cat_features=None):
        params = {**self.default_params, **self.params}
        n_rounds = params.pop("num_boost_round", 5000)
        early_stop = params.pop("early_stopping_rounds", 200)
        cat_features = cat_features or "auto"
        dtrain = lgb.Dataset(X, label=y, categorical_feature=cat_features)
        valid_sets = [dtrain]
        valid_names = ["train"]
        if X_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=dtrain)
            valid_sets.append(dval); valid_names.append("valid")
            callbacks = [lgb.early_stopping(early_stop, verbose=False),
                         lgb.log_evaluation(period=0)]
        else:
            callbacks = [lgb.log_evaluation(period=0)]
        self.model_ = lgb.train(params, dtrain, num_boost_round=n_rounds,
                                valid_sets=valid_sets, valid_names=valid_names,
                                callbacks=callbacks)
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

    def fit(self, X, y, X_val=None, y_val=None, cat_features=None):
        params = {**self.default_params, **self.params}
        n_rounds = params.pop("num_boost_round", 5000)
        early_stop = params.pop("early_stopping_rounds", 200)
        enable_cat = bool(cat_features)
        params["enable_categorical"] = enable_cat
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=enable_cat)
        evals = [(dtrain, "train")]
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=enable_cat)
            evals.append((dval, "valid"))
            self.model_ = xgb.train(params, dtrain, num_boost_round=n_rounds,
                                    evals=evals, early_stopping_rounds=early_stop,
                                    verbose_eval=False)
        else:
            self.model_ = xgb.train(params, dtrain, num_boost_round=n_rounds,
                                    evals=evals, verbose_eval=False)
        return self

    def predict(self, X):
        d = xgb.DMatrix(X, enable_categorical=True)
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

    def fit(self, X, y, X_val=None, y_val=None, cat_features=None):
        params = {**self.default_params, **self.params}
        cat_features = cat_features or []
        # CatBoost требует int категории передать
        train_pool = Pool(X, y, cat_features=cat_features)
        eval_pool = Pool(X_val, y_val, cat_features=cat_features) if X_val is not None else None
        self.model_ = CatBoostRegressor(**params)
        self.model_.fit(train_pool, eval_set=eval_pool, use_best_model=eval_pool is not None)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def feature_importance(self):
        if self.model_ is None: return None
        return pd.Series(self.model_.get_feature_importance(),
                         index=self.model_.feature_names_).sort_values(ascending=False)


MODEL_REGISTRY = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
}
