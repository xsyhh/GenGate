from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .progress import progress_iter


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> LinearProbe:
    set_seed(seed)
    model = LinearProbe(int(features.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=True)
    total_epochs = int(epochs)
    for epoch in range(total_epochs):
        model.train()
        for batch_x, batch_y in progress_iter(loader, desc=f"probe train epoch {epoch + 1}/{total_epochs}", total=len(loader)):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
    return model


@torch.inference_mode()
def predict_self_probs(model: LinearProbe, features: torch.Tensor, *, batch_size: int, device: str) -> torch.Tensor:
    model.eval()
    probs = []
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    for (batch_x,) in progress_iter(loader, desc="probe predict", total=len(loader)):
        logits = model(batch_x.to(device))
        probs.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu())
    return torch.cat(probs, dim=0)


def save_probe(path: str | Path, model: LinearProbe, config: dict[str, Any]) -> None:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, out / "probe.pt")
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_probe(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[LinearProbe, dict[str, Any]]:
    payload = torch.load(Path(path) / "probe.pt" if Path(path).is_dir() else path, map_location=map_location, weights_only=False)
    config = dict(payload.get("config", {}))
    input_dim = int(config.get("input_dim") or payload["state_dict"]["linear.weight"].shape[1])
    model = LinearProbe(input_dim)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config
