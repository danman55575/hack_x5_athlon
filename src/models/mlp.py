from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from .base import BaseModel


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPModel(BaseModel):
    name = "mlp"
    default_params = dict(
        hidden=(256, 128, 64),
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=4096,
        epochs=80,
        patience=15,
        loss="l1",
    )

    def fit(self, X, y, X_val=None, y_val=None, cat_features=None,
            sample_weight=None, sample_weight_val=None, seed=None, **kwargs):
        p = {**self.default_params, **(self.params or {})}
        if seed is not None:
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            np.random.seed(int(seed))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        Xn = self.scaler.fit_transform(self.imputer.fit_transform(X.values))
        Xn_t = torch.from_numpy(Xn.astype(np.float32))
        yn_t = torch.from_numpy(np.asarray(y).astype(np.float32))

        if sample_weight is not None:
            w_t = torch.from_numpy(np.asarray(sample_weight).astype(np.float32))
            ds = TensorDataset(Xn_t, yn_t, w_t)
            use_w = True
        else:
            ds = TensorDataset(Xn_t, yn_t)
            use_w = False
        dl = DataLoader(ds, batch_size=p["batch_size"], shuffle=True)

        if X_val is not None:
            Xv = self.scaler.transform(self.imputer.transform(X_val.values))
            Xv_t = torch.from_numpy(Xv.astype(np.float32)).to(device)
            yv_t = torch.from_numpy(np.asarray(y_val).astype(np.float32)).to(device)
            if sample_weight_val is not None:
                wv_t = torch.from_numpy(np.asarray(sample_weight_val).astype(np.float32)).to(device)
            else:
                wv_t = None
        else:
            Xv_t = yv_t = wv_t = None

        self.model_ = _MLP(Xn_t.shape[1], hidden=tuple(p["hidden"]), dropout=p["dropout"]).to(device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

        def _loss(pred, target, w=None):
            if p["loss"] == "l1":
                err = torch.abs(pred - target)
            else:
                err = (pred - target) ** 2
            if w is not None:
                return (err * w).sum() / (w.sum() + 1e-9)
            return err.mean()

        best, bad, best_state = float("inf"), 0, None
        for ep in range(p["epochs"]):
            self.model_.train()
            for batch in dl:
                if use_w:
                    xb, yb, wb = [t.to(device) for t in batch]
                else:
                    xb, yb = [t.to(device) for t in batch]
                    wb = None
                opt.zero_grad()
                pred = self.model_(xb)
                loss = _loss(pred, yb, wb)
                loss.backward()
                opt.step()
            if Xv_t is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_pred = self.model_(Xv_t)
                    val_loss = _loss(val_pred, yv_t, wv_t).item()
                if val_loss < best - 1e-6:
                    best, bad = val_loss, 0
                    best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                else:
                    bad += 1
                    if bad >= p["patience"]:
                        break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.device = device
        self.best_iteration_ = None
        return self

    def predict(self, X):
        Xn = self.scaler.transform(self.imputer.transform(X.values)).astype(np.float32)
        with torch.no_grad():
            self.model_.eval()
            t = torch.from_numpy(Xn).to(self.device)
            return self.model_(t).cpu().numpy()
