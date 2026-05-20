from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    name: str = "base"

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.model_ = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            X_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None,
            cat_features: list[str] | None = None) -> "BaseModel": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def feature_importance(self) -> pd.Series | None:
        return None
