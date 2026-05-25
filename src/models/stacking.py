from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .base import BaseModel


class Blender(BaseModel):
    """Простой блендер: Ridge-комбинация уже посчитанных OOF предсказаний."""
    name = "blender"

    def __init__(self, params=None):
        super().__init__(params)
        self.scaler = StandardScaler()
        alpha = (self.params or {}).get("alpha", 1.0)
        self.model_ = Ridge(alpha=alpha, positive=(self.params or {}).get("positive", True))

    def fit(self, X, y, X_val=None, y_val=None, cat_features=None):
        X_scaled = self.scaler.fit_transform(X.values)
        self.model_.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.fit(X.values)
        return self.model_.predict(X_scaled)


def weighted_blend(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    if total <= 0: raise ValueError("Weights sum must be positive")
    out = None
    for name, pred in predictions.items():
        w = weights.get(name, 0.0) / total
        out = pred * w if out is None else out + pred * w
    return out
