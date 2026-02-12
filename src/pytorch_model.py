# src/pytorch_model.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class MLPBinary(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TorchModelResult:
    model: nn.Module
    train_acc: float
    test_acc: float


def train_mlp_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 40,
    lr: float = 1e-3,
    seed: int = 0,
) -> TorchModelResult:
    """
    Very simple MLP training for demo purposes.
    Keep it stable and easy to reproduce (not fancy).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")

    n_features = X_train.shape[1]
    model = MLPBinary(n_features=n_features).to(device)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    yte = torch.tensor(y_test, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(Xtr)
        loss = criterion(logits, ytr)
        loss.backward()
        optimizer.step()

    def acc_for(X: torch.Tensor, y: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            logits = model(X)
            preds = (torch.sigmoid(logits) >= 0.5).to(torch.int32)
            return float((preds.cpu().numpy() == y.cpu().numpy().astype(int)).mean())

    train_acc = acc_for(Xtr, ytr)
    test_acc = acc_for(Xte, yte)

    return TorchModelResult(model=model, train_acc=train_acc, test_acc=test_acc)


def predict_mlp_binary(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = torch.device("cpu")
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(Xt)
        preds = (torch.sigmoid(logits) >= 0.5).to(torch.int32)
    return preds.cpu().numpy()