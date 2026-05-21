from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    name: str = "base"

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.model_ = None
        self.best_iteration_ = None

    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None, cat_features=None,
            sample_weight=None, sample_weight_val=None, seed=None): ...

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...

    def feature_importance(self): return None
