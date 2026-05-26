from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch


def _load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("format") != "probe_feature_shards_v1":
        raise ValueError(f"Not a probe feature manifest: {path}")
    return payload


def _resolve_shard_path(manifest_path: Path, raw_path: str | Path) -> Path:
    shard = Path(raw_path)
    candidates = []
    if shard.is_absolute():
        candidates.append(shard)
        candidates.append(manifest_path.parent / shard.name)
        candidates.append(manifest_path.parent / f"{manifest_path.stem}_shards" / shard.name)
        candidates.append(manifest_path.parent / f"{manifest_path.stem.replace('_manifest', '')}_shards" / shard.name)
    else:
        candidates.append(manifest_path.parent / shard)
        candidates.append(manifest_path.parent / f"{manifest_path.stem}_shards" / shard.name)
        candidates.append(manifest_path.parent / f"{manifest_path.stem.replace('_manifest', '')}_shards" / shard.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve shard path {raw_path!r} from {manifest_path}")


class FeatureManifest:
    def __init__(self, manifest_path: str | Path):
        self.path = Path(manifest_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.manifest = _load_manifest(self.path)
        self.shard_paths = [_resolve_shard_path(self.path, item["path"]) for item in self.manifest["shards"]]
        self.feature_dim = int(self.manifest["shards"][0]["feature_dim"])

    def iter_payloads(self) -> Iterable[dict[str, Any]]:
        for shard_path in self.shard_paths:
            yield torch.load(shard_path, map_location="cpu", weights_only=False)

    def ratios(self) -> list[float]:
        values = set()
        for payload in self.iter_payloads():
            for meta in payload["metadata"]:
                values.add(float(meta["ratio"]))
        return sorted(values)

    def materialize_ratio(self, ratio: float) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        features = []
        labels = []
        metadata = []
        target_ratio = float(ratio)
        for payload in self.iter_payloads():
            shard_features = payload["features"]
            shard_labels = payload["labels"]
            for idx, meta in enumerate(payload["metadata"]):
                if abs(float(meta["ratio"]) - target_ratio) > 1e-8:
                    continue
                features.append(shard_features[idx].to(dtype=torch.float32).clone())
                labels.append(int(shard_labels[idx]))
                metadata.append(dict(meta))
        if not features:
            raise ValueError(f"No features found for ratio={ratio} in {self.path}")
        return torch.stack(features, dim=0), torch.tensor(labels, dtype=torch.long), metadata


def task_sample_key(meta: dict[str, Any]) -> tuple[str, int]:
    return str(meta["task_id"]), int(meta.get("sample_idx", meta.get("sample_index", 0)))


def align_records(
    metadata: list[dict[str, Any]],
    labels: torch.Tensor,
    probs: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    for meta, label, prob in zip(metadata, labels.tolist(), probs.tolist()):
        rows.append(
            {
                "task_id": str(meta["task_id"]),
                "sample_idx": int(meta.get("sample_idx", meta.get("sample_index", 0))),
                "sample_index": int(meta.get("sample_idx", meta.get("sample_index", 0))),
                "ratio": float(meta.get("ratio", 0.0)),
                "self_passed": int(label),
                "score": float(prob),
            }
        )
    return rows


def index_by_task_sample(records: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(row["task_id"]), int(row.get("sample_idx", row.get("sample_index", 0)))): row for row in records}
