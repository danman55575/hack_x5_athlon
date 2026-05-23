"""Фиксация всех источников случайности.
"""
import os
import random
import numpy as np


def set_seed(seed: int = 2026) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Для детерминированных CUDA-операций (если их случайно вызовут библиотеки)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
            try:
                if torch.backends.mps.is_available():
                    torch.mps.manual_seed(seed)
            except Exception:
                pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except ImportError:
        pass
