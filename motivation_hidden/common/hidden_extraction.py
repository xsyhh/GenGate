"""Shared memory-safe hidden-state extraction for Motivation step2 scripts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


def iter_jsonl_records(jsonl_path: str | Path, max_records: int | None = None):
    count = 0
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if max_records is not None and count >= max_records:
                break


def count_jsonl_records(jsonl_path: str | Path, max_records: int | None = None) -> int:
    count = 0
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            if max_records is not None and count >= max_records:
                break
    return count


class DraftIterableDataset(IterableDataset):
    def __init__(self, jsonl_path: str | Path, max_records: int | None = None):
        self.jsonl_path = Path(jsonl_path)
        self.max_records = max_records

    def __iter__(self):
        return iter_jsonl_records(self.jsonl_path, self.max_records)


def resolve_manifest_path(out_dir: Path, data_jsonl: Path, out_name: str | None) -> Path:
    if out_name:
        out_path = out_dir / out_name
        if out_path.suffix == ".json":
            return out_path
        return out_path.with_name(f"{out_path.stem}_manifest.json")
    stem = data_jsonl.stem.replace("_sliced_drafts", "")
    return out_dir / f"{stem}_probe_features_manifest.json"


def save_feature_shard(
    shard_dir: Path,
    shard_stem: str,
    shard_idx: int,
    hidden_chunks: list[torch.Tensor],
    label_chunks: list[torch.Tensor],
    metadata: list[dict],
) -> dict:
    if not hidden_chunks:
        raise ValueError("cannot save an empty feature shard")
    shard_dir.mkdir(parents=True, exist_ok=True)
    features = torch.cat(hidden_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    if features.shape[0] != labels.shape[0] or features.shape[0] != len(metadata):
        raise ValueError(
            f"shard size mismatch: features={features.shape[0]} labels={labels.shape[0]} metadata={len(metadata)}"
        )
    path = shard_dir / f"{shard_stem}_shard_{shard_idx:05d}.pt"
    torch.save({"features": features, "labels": labels, "metadata": list(metadata)}, path)
    return {
        "path": str(path.name),
        "n": int(features.shape[0]),
        "feature_dim": int(features.shape[1]) if features.ndim == 2 else None,
    }


def write_manifest(path: Path, shards: list[dict], source_jsonl: Path) -> None:
    payload = {
        "format": "probe_feature_shards_v1",
        "source_jsonl": str(source_jsonl),
        "n_total": sum(int(shard["n"]) for shard in shards),
        "n_shards": len(shards),
        "shards": shards,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def collate_fn(batch, tokenizer, max_length: int):
    texts = []
    labels = []
    metadata = []
    for item in batch:
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(item["prompt_text"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(chat_prompt + str(item["prefix_raw"]))
        labels.append(int(item["y_final"]))
        metadata.append(
            {
                "task_id": item["task_id"],
                "sample_idx": int(item["sample_idx"]),
                "ratio": float(item["ratio"]),
            }
        )
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return inputs, torch.tensor(labels, dtype=torch.long), metadata


def forward_last_hidden_state(model, inputs, device):
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    backbone = getattr(model, "model", None)
    if backbone is not None:
        outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return outputs.last_hidden_state
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return outputs.hidden_states[-1]


def run_extraction(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_manifest_path(args.out_dir, args.data_jsonl, args.out_name)
    shard_dir = args.out_dir / f"{manifest_path.stem}_shards"
    shard_stem = manifest_path.stem.replace("_manifest", "")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_records = count_jsonl_records(args.data_jsonl, max_records=args.max_records)
    n_batches = math.ceil(n_records / args.batch_size) if n_records else 0
    print(
        f"Extraction input: {n_records} rows, batch_size={args.batch_size}, total_batches={n_batches}",
        flush=True,
    )
    print(f"Using device={device}, dtype={args.dtype}, model={args.model}", flush=True)
    print("Importing transformers...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model checkpoint...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
    )
    model.eval()

    dataset = DraftIterableDataset(args.data_jsonl, max_records=args.max_records)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
    )

    hidden_chunks = []
    label_chunks = []
    metadata = []
    shard_infos = []
    pending_rows = 0

    with torch.inference_mode():
        for inputs, labels, batch_meta in tqdm(
            dataloader,
            desc="Extracting hidden states",
            total=n_batches,
            unit="batch",
        ):
            last_hidden = forward_last_hidden_state(model, inputs, device)
            hidden_chunks.append(last_hidden[:, -1, :].cpu().to(torch.float32))
            label_chunks.append(labels)
            metadata.extend(batch_meta)
            pending_rows += int(labels.shape[0])

            if pending_rows >= args.shard_size:
                info = save_feature_shard(
                    shard_dir,
                    shard_stem,
                    len(shard_infos),
                    hidden_chunks,
                    label_chunks,
                    metadata,
                )
                shard_infos.append(info)
                print(f"Saved feature shard {len(shard_infos) - 1}: {info['n']} rows -> {info['path']}")
                hidden_chunks = []
                label_chunks = []
                metadata = []
                pending_rows = 0

    if hidden_chunks:
        info = save_feature_shard(
            shard_dir,
            shard_stem,
            len(shard_infos),
            hidden_chunks,
            label_chunks,
            metadata,
        )
        shard_infos.append(info)
        print(f"Saved feature shard {len(shard_infos) - 1}: {info['n']} rows -> {info['path']}")

    write_manifest(manifest_path, shard_infos, args.data_jsonl)
    print(f"Saved shard manifest to {manifest_path}")
    print(f"Total rows saved: {sum(int(info['n']) for info in shard_infos)}")


def add_extraction_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_jsonl", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=200000)
    return parser
