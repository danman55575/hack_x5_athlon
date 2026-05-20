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
        if kind == "ridge":
            reg = Ridge(alpha=alpha, random_state=42)
        elif kind == "elasticnet":
            l1 = (self.params or {}).get("l1_ratio", 0.5)
            reg = ElasticNet(alpha=alpha, l1_ratio=l1, random_state=42, max_iter=10000)
        elif kind == "huber":
            reg = HuberRegressor(alpha=alpha, max_iter=500)
        else:
            raise ValueError(f"Unknown linear kind: {kind}")
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", reg),
        ])

    def fit(self, X, y, X_val=None, y_val=None, cat_features=None):
        self.pipeline.fit(X.values, y)
        self.model_ = self.pipeline
        return self

    def predict(self, X):
        return self.pipeline.predict(X.values)
