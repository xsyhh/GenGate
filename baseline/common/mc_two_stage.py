from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .probe import LinearProbe, set_seed
from .progress import progress_iter


def compute_task_mean_targets(
    post_metadata: list[dict[str, Any]],
    post_labels: torch.Tensor,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for meta, label in zip(post_metadata, post_labels.tolist()):
        task_id = str(meta["task_id"])
        totals[task_id] += float(label)
        counts[task_id] += 1
    return {task_id: totals[task_id] / counts[task_id] for task_id in counts}


def materialize_pre_mc_targets(
    pre_features: torch.Tensor,
    pre_metadata: list[dict[str, Any]],
    task_targets: dict[str, float],
    *,
    dedupe_by_task: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    out_features = []
    out_targets = []
    out_metadata: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()

    for idx, meta in enumerate(pre_metadata):
        task_id = str(meta["task_id"])
        if task_id not in task_targets:
            continue
        if dedupe_by_task and task_id in seen_tasks:
            continue
        out_features.append(pre_features[idx].to(dtype=torch.float32).clone())
        out_targets.append(float(task_targets[task_id]))
        out_metadata.append(dict(meta))
        seen_tasks.add(task_id)

    if not out_features:
        raise ValueError("No pre-stage rows matched task targets.")
    return (
        torch.stack(out_features, dim=0),
        torch.tensor(out_targets, dtype=torch.float32),
        out_metadata,
    )


def train_probe_soft_targets(
    features: torch.Tensor,
    soft_targets: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    sample_weights: torch.Tensor | None = None,
) -> LinearProbe:
    set_seed(seed)
    model = LinearProbe(int(features.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if sample_weights is None:
        dataset = TensorDataset(features, soft_targets)
    else:
        if sample_weights.shape[0] != soft_targets.shape[0]:
            raise ValueError("sample_weights must have the same length as soft_targets")
        dataset = TensorDataset(features, soft_targets, sample_weights)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_epochs = int(epochs)
    for epoch in range(total_epochs):
        model.train()
        for batch in progress_iter(loader, desc=f"probe train epoch {epoch + 1}/{total_epochs}", total=len(loader)):
            if sample_weights is None:
                batch_x, batch_y = batch
                batch_w = None
            else:
                batch_x, batch_y, batch_w = batch
            optimizer.zero_grad(set_to_none=True)
            batch_logits = model(batch_x.to(device))
            margin = batch_logits[:, 1] - batch_logits[:, 0]
            target_prob = batch_y.to(device=device, dtype=margin.dtype).clamp(0.0, 1.0)
            # Match local_state_pref BCE preference loss (without KL):
            # -t*logsigmoid(m) - (1-t)*logsigmoid(-m)
            per_state_loss = -target_prob * F.logsigmoid(margin) - (1.0 - target_prob) * F.logsigmoid(-margin)
            if batch_w is None:
                loss = per_state_loss.mean()
            else:
                weight = batch_w.to(device=device, dtype=margin.dtype)
                loss = (per_state_loss * weight).sum() / weight.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
    return model
