from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.common.domain_data import read_domain_examples
from baseline.common.metrics import row_from_score, sweep_single_threshold, write_metadata, write_rows_csv
from baseline.common.paths import make_run_paths
from baseline.common.progress import progress_iter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RoBERTa-based learned external router.")
    p.add_argument("--domain", required=True, choices=["code", "math", "mmlu"])
    p.add_argument("--weak_model", required=True)
    p.add_argument("--strong_model", default="roberta-base")
    p.add_argument("--train_data_path", required=True)
    p.add_argument("--train_weak_results_csv", required=True)
    p.add_argument("--eval_data_path", required=True)
    p.add_argument("--eval_weak_results_csv", required=True)
    p.add_argument("--eval_strong_results_csv", default="")
    p.add_argument("--run_name", default="default")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def robust_read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return [{str(k).strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def robust_read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _to_int01(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    return 0


def load_pass_map(path: str | Path, *, examples=None) -> dict[str, int]:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    index_to_task_id = {str(i): str(ex.task_id) for i, ex in enumerate(examples or [])}
    if suffix == ".jsonl":
        rows = robust_read_jsonl(path_obj)
        out: dict[str, int] = {}
        for row in rows:
            task_id_raw = str(row.get("task_id", "")).strip()
            if not task_id_raw:
                continue
            task_id = index_to_task_id.get(task_id_raw, task_id_raw)
            sample_index = row.get("sample_index", None)
            if "self_passed" in row:
                if sample_index not in (0, "0"):
                    continue
                out[task_id] = _to_int01(row.get("self_passed", 0))
                continue
            if "target_prob" in row:
                # local_state_pref train_pairs.jsonl
                if str(row.get("state_type", "")).strip() != "s1":
                    continue
                if sample_index not in (0, "0"):
                    continue
                try:
                    out[task_id] = 1 if float(row.get("target_prob", 0.0)) >= 0.5 else 0
                except (TypeError, ValueError):
                    out[task_id] = 0
                continue
        return out

    rows = robust_read_csv(path_obj)
    out: dict[str, int] = {}
    for row in rows:
        task_id_raw = str(row.get("task_id", "")).strip()
        if not task_id_raw:
            continue
        task_id = index_to_task_id.get(task_id_raw, task_id_raw)
        key = "expert_passed" if "expert_passed" in row else "self_passed"
        out[task_id] = _to_int01(row.get(key, 0))
    return out


def build_router_label(*, weak_correct: int, strong_correct: int, drop_both_wrong: bool) -> int | None:
    # Training target is weak-only:
    # weak correct -> stay weak (0), weak wrong -> route strong/defer (1).
    # strong_correct and drop_both_wrong are ignored for label construction
    # but kept in signature for script/config compatibility.
    _ = strong_correct
    _ = drop_both_wrong
    return 0 if int(weak_correct) == 1 else 1


def build_router_rows(
    examples,
    weak_map: dict[str, int],
    strong_map: dict[str, int] | None = None,
    *,
    assume_expert_correct: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ex in progress_iter(examples, desc="build roberta rows", total=len(examples)):
        task_id = str(ex.task_id)
        if task_id not in weak_map:
            continue
        weak_correct = int(weak_map[task_id])
        if strong_map is None:
            strong_correct = 1 if assume_expert_correct else 0
        else:
            strong_correct = int(strong_map.get(task_id, 1 if assume_expert_correct else 0))
        label = build_router_label(
            weak_correct=weak_correct,
            strong_correct=strong_correct,
            drop_both_wrong=False,
        )
        if label is None:
            continue
        rows.append(
            {
                "task_id": task_id,
                "text": str(ex.problem or ""),
                "label": int(label),
                "weak_correct": weak_correct,
                "strong_correct": strong_correct,
            }
        )
    return rows


def _tokenize_batch(batch, tokenizer, max_length: int):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
    )


def train_router(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    model_name: str,
    output_dir: Path,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    gradient_accumulation_steps: int,
    num_proc: int,
    trust_remote_code: bool,
    local_files_only: bool,
):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    tokenized_train = train_ds.map(
        lambda batch: _tokenize_batch(batch, tokenizer, max_length),
        batched=True,
        num_proc=max(1, int(num_proc)),
        desc="tokenize roberta train",
    )
    tokenized_eval = eval_ds.map(
        lambda batch: _tokenize_batch(batch, tokenizer, max_length),
        batched=True,
        num_proc=max(1, int(num_proc)),
        desc="tokenize roberta eval",
    )
    tokenized_train = tokenized_train.remove_columns([c for c in tokenized_train.column_names if c not in {"input_ids", "attention_mask", "label"}])
    tokenized_eval = tokenized_eval.remove_columns([c for c in tokenized_eval.column_names if c not in {"input_ids", "attention_mask", "label"}])
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_eval = tokenized_eval.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    training_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": int(batch_size),
        "per_device_eval_batch_size": int(batch_size),
        "gradient_accumulation_steps": max(1, int(gradient_accumulation_steps)),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "warmup_ratio": float(warmup_ratio),
        "num_train_epochs": float(epochs),
        "lr_scheduler_type": "cosine",
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "logging_strategy": "steps",
        "logging_steps": 10,
        "dataloader_num_workers": 2,
        "report_to": [],
        "seed": int(seed),
        "data_seed": int(seed),
        "remove_unused_columns": True,
        "disable_tqdm": False,
    }
    ta_sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in ta_sig.parameters:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_sig.parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        raise RuntimeError("Current transformers.TrainingArguments has neither evaluation_strategy nor eval_strategy.")
    args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_eval,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


def predict_p_strong(
    rows: list[dict[str, Any]],
    *,
    ckpt_dir: Path,
    max_length: int,
    batch_size: int,
    trust_remote_code: bool,
    local_files_only: bool,
):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    ds = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(
        str(ckpt_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    tokenized = ds.map(
        lambda batch: _tokenize_batch(batch, tokenizer, max_length),
        batched=True,
        num_proc=1,
        desc="tokenize roberta predict",
    )
    keep_cols = {"input_ids", "attention_mask"}
    tokenized = tokenized.remove_columns([c for c in tokenized.column_names if c not in keep_cols])
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    pred_args = TrainingArguments(
        output_dir=str(ckpt_dir / "predict_tmp"),
        per_device_eval_batch_size=int(batch_size),
        dataloader_num_workers=2,
        report_to=[],
        disable_tqdm=False,
    )
    trainer_sig = inspect.signature(Trainer.__init__)
    trainer_kwargs = {"model": model, "args": pred_args}
    if "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    outputs = trainer.predict(tokenized)
    logits = torch.tensor(outputs.predictions)
    probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
    return [float(p) for p in probs]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    paths = make_run_paths(
        method="Roberta-Router",
        domain=args.domain,
        model=args.weak_model,
        run_name=args.run_name,
    )
    train_examples = read_domain_examples(args.domain, args.train_data_path)
    train_weak_map = load_pass_map(args.train_weak_results_csv, examples=train_examples)
    train_rows = build_router_rows(
        train_examples,
        train_weak_map,
        strong_map=None,
        assume_expert_correct=True,
    )
    if not train_rows:
        raise ValueError("No aligned train rows after task-id join. Check train_data_path and train_weak_results_csv task_id.")

    eval_examples = read_domain_examples(args.domain, args.eval_data_path)
    eval_weak_map = load_pass_map(args.eval_weak_results_csv, examples=eval_examples)
    eval_strong_map = load_pass_map(args.eval_strong_results_csv, examples=eval_examples) if str(args.eval_strong_results_csv).strip() else None
    eval_rows = build_router_rows(
        eval_examples,
        eval_weak_map,
        eval_strong_map,
        assume_expert_correct=eval_strong_map is None,
    )
    if not eval_rows:
        raise ValueError("No aligned eval rows after task-id join. Check eval_data_path and eval_weak_results_csv task_id.")

    train_router(
        train_rows,
        eval_rows,
        model_name=args.strong_model,
        output_dir=paths.ckpt,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_proc=args.num_proc,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    pred_rows = eval_rows
    scores = predict_p_strong(
        pred_rows,
        ckpt_dir=paths.ckpt,
        max_length=args.max_length,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if len(scores) != len(pred_rows):
        raise RuntimeError("Prediction length mismatch.")

    score_rows = []
    metadata_rows = []
    threshold_pstrong = float(args.threshold)
    threshold_pweak = 1.0 - threshold_pstrong
    for row, p_strong in zip(pred_rows, scores):
        p_self = 1.0 - float(p_strong)
        weak_correct = int(row["weak_correct"])
        strong_correct = int(row["strong_correct"])
        score_rows.append(
            {
                "task_id": row["task_id"],
                "text": row["text"],
                "label": int(row["label"]),
                "weak_correct": weak_correct,
                "strong_correct": strong_correct,
                "p_strong": float(p_strong),
                "p_weak": float(p_self),
            }
        )
        metadata_rows.append(
            row_from_score(
                task_id=row["task_id"],
                score=p_self,
                self_passed=weak_correct,
                threshold=threshold_pweak,
                method="Roberta-Router",
                domain=args.domain,
                model_slug=paths.model_slug,
                dataset_slug=paths.dataset_slug,
                expert_passed=strong_correct,
                stage="pre",
                route_defer="early_defer",
            )
        )

    threshold_curve = sweep_single_threshold(
        [
            {
                "score": row["p_weak"],
                "self_passed": row["weak_correct"],
                "expert_passed": row["strong_correct"],
            }
            for row in score_rows
        ]
    )

    write_rows_csv(paths.outputs / "scores.csv", score_rows)
    write_metadata(paths.eval_output / "metadata.csv", metadata_rows)
    write_rows_csv(paths.eval_output / "threshold_curve.csv", threshold_curve)
    with (paths.eval_output / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "domain": args.domain,
                "weak_model": args.weak_model,
                "strong_model": args.strong_model,
                "run_name": args.run_name,
                "train_data_path": args.train_data_path,
                "train_weak_results_csv": args.train_weak_results_csv,
                "eval_data_path": args.eval_data_path,
                "eval_weak_results_csv": args.eval_weak_results_csv,
                "eval_strong_results_csv": args.eval_strong_results_csv,
                "threshold_pstrong": threshold_pstrong,
                "threshold_pweak": threshold_pweak,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "seed": args.seed,
                "assume_expert_correct_in_eval": eval_strong_map is None,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "num_examples_aligned_train": len(train_rows),
                "num_examples_aligned_eval": len(eval_rows),
                "num_train": len(train_rows),
                "num_eval": len(eval_rows),
                "num_pred": len(pred_rows),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved metadata to {paths.eval_output / 'metadata.csv'}")


if __name__ == "__main__":
    main()
