from __future__ import annotations

import argparse
import json
from pathlib import Path

from .features import FeatureManifest, align_records, index_by_task_sample
from .metrics import (
    attach_expert,
    cascade_grid,
    load_expert_map,
    pareto_envelope,
    write_metadata,
    write_rows_csv,
)
from .paths import make_run_paths, manifest_path, split_for
from .probe import load_probe, predict_self_probs, resolve_device, save_probe, train_probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train+evaluate Naive Two-Stage Cascade with pre/post linear probes.")
    p.add_argument("--domain", required=True, choices=["code", "math", "mmlu"])
    p.add_argument("--model", required=True)
    p.add_argument("--run_name", default="default")
    p.add_argument("--pre_probe", default=None)
    p.add_argument("--post_probe", default=None)
    p.add_argument("--train_manifest", default=None)
    p.add_argument("--eval_manifest", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--pre_threshold", type=float, default=None)
    p.add_argument("--post_threshold", type=float, default=None)
    p.add_argument("--target_expert_rate", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--expert_csv", default=None)
    p.add_argument("--assume_expert_correct", action="store_true")
    return p.parse_args()


def _metadata_row(record: dict, pre_t: float, post_t: float, paths) -> dict:
    pre = float(record["pre_score"])
    post = float(record["post_score"])
    self_passed = int(record["self_passed"])
    expert_passed = record.get("expert_passed", "")
    if pre < pre_t:
        route = "early_defer"
        model_decision = "defer"
        p_self = pre
        stage = "pre"
    elif post < post_t:
        route = "post_defer"
        model_decision = "defer"
        p_self = post
        stage = "post"
    else:
        route = "post_self"
        model_decision = "self"
        p_self = post
        stage = "post"
    p_defer = 1.0 - p_self
    return {
        "task_id": record["task_id"],
        "route": route,
        "model_decision": model_decision,
        "first_p_defer": round(1.0 - pre, 6),
        "first_p_self": round(pre, 6),
        "first_margin": "",
        "post_p_defer": round(1.0 - post, 6),
        "post_p_self": round(post, 6),
        "post_margin": "",
        "p_defer": round(p_defer, 6),
        "p_self": round(p_self, 6),
        "margin": "",
        "self_passed": self_passed,
        "expert_passed": expert_passed,
        "answer_len": 0,
        "actual_local_tokens": 0 if route == "early_defer" else int(record.get("answer_len", 0)),
        "method": "Naive-Two-Stage-Cascade",
        "domain": paths.domain,
        "model_slug": paths.model_slug,
        "dataset_slug": paths.dataset_slug,
        "score": round(p_self, 6),
        "pre_score": round(pre, 6),
        "post_score": round(post, 6),
        "decision_stage": stage,
        "pre_threshold": pre_t,
        "post_threshold": post_t,
    }


def main() -> None:
    args = parse_args()
    paths = make_run_paths(method="Naive-Two-Stage-Cascade", domain=args.domain, model=args.model, run_name=args.run_name)
    train_manifest = args.train_manifest or str(manifest_path(paths.model_slug, split_for(args.domain, "train")))
    eval_manifest = args.eval_manifest or str(manifest_path(paths.model_slug, split_for(args.domain, "eval")))

    device = resolve_device(args.device)
    if bool(args.pre_probe) ^ bool(args.post_probe):
        raise ValueError("Please provide both --pre_probe and --post_probe, or neither.")

    if args.pre_probe and args.post_probe:
        pre_probe_path = Path(args.pre_probe)
        post_probe_path = Path(args.post_probe)
        pre_probe, _ = load_probe(pre_probe_path, map_location=device)
        post_probe, _ = load_probe(post_probe_path, map_location=device)
        pre_probe.to(device)
        post_probe.to(device)
    else:
        train_source = FeatureManifest(train_manifest)
        train_pre_x, train_pre_y, _ = train_source.materialize_ratio(0.0)
        train_post_x, train_post_y, _ = train_source.materialize_ratio(1.0)
        pre_probe = train_probe(
            train_pre_x,
            train_pre_y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
        )
        post_probe = train_probe(
            train_post_x,
            train_post_y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
        )
        pre_cfg = {
            "method": "Naive-Two-Stage-Cascade",
            "stage": "pre",
            "domain": args.domain,
            "model_slug": paths.model_slug,
            "dataset_slug": paths.dataset_slug,
            "ratio": 0.0,
            "train_manifest": str(train_manifest),
            "input_dim": int(train_pre_x.shape[1]),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        }
        post_cfg = {
            "method": "Naive-Two-Stage-Cascade",
            "stage": "post",
            "domain": args.domain,
            "model_slug": paths.model_slug,
            "dataset_slug": paths.dataset_slug,
            "ratio": 1.0,
            "train_manifest": str(train_manifest),
            "input_dim": int(train_post_x.shape[1]),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        }
        pre_probe_path = paths.ckpt / "pre_probe"
        post_probe_path = paths.ckpt / "post_probe"
        save_probe(pre_probe_path, pre_probe, pre_cfg)
        save_probe(post_probe_path, post_probe, post_cfg)

    source = FeatureManifest(eval_manifest)
    pre_x, pre_y, pre_meta = source.materialize_ratio(0.0)
    post_x, post_y, post_meta = source.materialize_ratio(1.0)
    pre_records = align_records(pre_meta, pre_y, predict_self_probs(pre_probe, pre_x, batch_size=args.batch_size, device=device))
    post_records = align_records(post_meta, post_y, predict_self_probs(post_probe, post_x, batch_size=args.batch_size, device=device))
    post_index = index_by_task_sample(post_records)

    records = []
    for pre in pre_records:
        key = (str(pre["task_id"]), int(pre["sample_idx"]))
        if key not in post_index:
            continue
        post = post_index[key]
        records.append(
            {
                "task_id": pre["task_id"],
                "sample_idx": pre["sample_idx"],
                "pre_score": pre["score"],
                "post_score": post["score"],
                "self_passed": post["self_passed"],
            }
        )

    expert_map = load_expert_map(args.expert_csv, assume_correct=args.assume_expert_correct)
    records = attach_expert(records, expert_map, assume_correct=args.assume_expert_correct or args.expert_csv is None)
    grid = cascade_grid(records)
    envelope = pareto_envelope(grid)

    if args.pre_threshold is not None and args.post_threshold is not None:
        pre_t = float(args.pre_threshold)
        post_t = float(args.post_threshold)
    else:
        target = float(args.target_expert_rate)
        chosen = min(grid, key=lambda row: (abs(float(row["expert_rate"]) - target), -float(row["accuracy"])))
        pre_t = float(chosen["pre_threshold"])
        post_t = float(chosen["post_threshold"])

    metadata_rows = [_metadata_row(record, pre_t, post_t, paths) for record in records]
    write_metadata(paths.eval_output / "metadata.csv", metadata_rows)
    write_rows_csv(paths.eval_output / "threshold_grid.csv", grid)
    write_rows_csv(paths.eval_output / "pareto_envelope.csv", envelope)
    write_rows_csv(paths.outputs / "scores.csv", records)
    with (paths.eval_output / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "domain": args.domain,
                "model_slug": paths.model_slug,
                "train_manifest": str(train_manifest),
                "eval_manifest": str(eval_manifest),
                "pre_probe": str(pre_probe_path),
                "post_probe": str(post_probe_path),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "pre_threshold": pre_t,
                "post_threshold": post_t,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved cascade metadata to {paths.eval_output / 'metadata.csv'}")
    print(f"Selected thresholds: pre={pre_t:.4f}, post={post_t:.4f}")


if __name__ == "__main__":
    main()
