import numpy as np

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), eps))) * 100.0)

def smape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_pred) + np.abs(y_true), eps)) * 100.0)

def mape_to_score(mape_value: float) -> float:
    """Формула баллов хакатона"""
    m = min(mape_value, 100.0)
    return 100.0 * ((100.0 - m) / 100.0) ** 2
