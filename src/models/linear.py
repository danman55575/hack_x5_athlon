from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .base import BaseModel


class LinearModel(BaseModel):
    """Ridge/ElasticNet/Huber на лог-таргете. Категории должны быть закодированы заранее."""
    name = "linear"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        kind = (self.params or {}).get("kind", "ridge")
        alpha = (self.params or {}).get("alpha", 1.0)
        seed = (self.params or {}).get("random_state", 42)
        if kind == "ridge":
            reg = Ridge(alpha=alpha, random_state=seed)
        elif kind == "elasticnet":
            l1 = (self.params or {}).get("l1_ratio", 0.5)
            reg = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=seed, max_iter=10000)
        elif kind == "huber":
            reg = HuberRegressor(alpha=alpha, max_iter=500)
        else:
            raise ValueError(f"Unknown linear kind: {kind}")
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", reg),
        ])

    def fit(self, X, y, X_val=None, y_val=None, cat_features=None,
            sample_weight=None, sample_weight_val=None, seed=None, **kwargs):
        fit_params = {}
        if sample_weight is not None:
            # sklearn Pipeline принимает sample_weight через имя_шага__sample_weight
            fit_params["reg__sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
        Xv = X.values if hasattr(X, "values") else X
        self.pipeline.fit(Xv, y, **fit_params)
        self.model_ = self.pipeline
        self.best_iteration_ = None
        return self

    def predict(self, X):
        Xv = X.values if hasattr(X, "values") else X
        return self.pipeline.predict(Xv)
