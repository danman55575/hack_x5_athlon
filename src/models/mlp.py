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

    def fit(self, X, y, X_val=None, y_val=None, cat_features=None):
        p = {**self.default_params, **(self.params or {})}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        Xn = self.scaler.fit_transform(self.imputer.fit_transform(X.values))
        Xn = torch.from_numpy(Xn.astype(np.float32))
        yn = torch.from_numpy(np.asarray(y).astype(np.float32))
        dl = DataLoader(TensorDataset(Xn, yn), batch_size=p["batch_size"], shuffle=True)

        if X_val is not None:
            Xv = self.scaler.transform(self.imputer.transform(X_val.values))
            Xv = torch.from_numpy(Xv.astype(np.float32)).to(device)
            yv = torch.from_numpy(np.asarray(y_val).astype(np.float32)).to(device)
        else:
            Xv = yv = None

        self.model_ = _MLP(Xn.shape[1], hidden=tuple(p["hidden"]), dropout=p["dropout"]).to(device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        loss_fn = nn.L1Loss() if p["loss"] == "l1" else nn.MSELoss()

        best, bad, best_state = float("inf"), 0, None
        for ep in range(p["epochs"]):
            self.model_.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward(); opt.step()
            if Xv is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_pred = self.model_(Xv)
                    val_loss = loss_fn(val_pred, yv).item()
                if val_loss < best - 1e-6:
                    best, bad, best_state = val_loss, 0, {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                else:
                    bad += 1
                    if bad >= p["patience"]:
                        break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.device = device
        return self

    def predict(self, X):
        Xn = self.scaler.transform(self.imputer.transform(X.values)).astype(np.float32)
        with torch.no_grad():
            self.model_.eval()
            t = torch.from_numpy(Xn).to(self.device)
            return self.model_(t).cpu().numpy()
