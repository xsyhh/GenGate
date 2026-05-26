from __future__ import annotations

import argparse
import json

from .features import FeatureManifest, align_records
from .metrics import (
    attach_expert,
    load_expert_map,
    row_from_score,
    select_threshold_for_budget,
    sweep_single_threshold,
    write_metadata,
    write_rows_csv,
)
from .paths import make_run_paths, manifest_path, split_for
from .probe import predict_self_probs, resolve_device, save_probe, train_probe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate Pre/Post Linear baseline.")
    p.add_argument("--method", required=True, choices=["Pre-Linear", "Post-Linear"])
    p.add_argument("--domain", required=True, choices=["code", "math", "mmlu"])
    p.add_argument("--model", required=True)
    p.add_argument("--run_name", default="default")
    p.add_argument("--train_manifest", default=None)
    p.add_argument("--eval_manifest", default=None)
    p.add_argument("--ratio", type=float, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--target_expert_rate", type=float, default=0.5)
    p.add_argument("--expert_csv", default=None)
    p.add_argument("--assume_expert_correct", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ratio = args.ratio
    if ratio is None:
        ratio = 0.0 if args.method == "Pre-Linear" else 1.0

    paths = make_run_paths(method=args.method, domain=args.domain, model=args.model, run_name=args.run_name)
    train_manifest = args.train_manifest or str(manifest_path(paths.model_slug, split_for(args.domain, "train")))
    eval_manifest = args.eval_manifest or str(manifest_path(paths.model_slug, split_for(args.domain, "eval")))
    device = resolve_device(args.device)

    train_source = FeatureManifest(train_manifest)
    eval_source = FeatureManifest(eval_manifest)
    train_x, train_y, _ = train_source.materialize_ratio(ratio)
    eval_x, eval_y, eval_meta = eval_source.materialize_ratio(ratio)

    model = train_probe(
        train_x,
        train_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )
    config = {
        "method": args.method,
        "domain": args.domain,
        "model_slug": paths.model_slug,
        "dataset_slug": paths.dataset_slug,
        "ratio": ratio,
        "train_manifest": str(train_manifest),
        "eval_manifest": str(eval_manifest),
        "input_dim": int(train_x.shape[1]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    save_probe(paths.ckpt, model, config)

    probs = predict_self_probs(model, eval_x, batch_size=args.batch_size, device=device)
    records = align_records(eval_meta, eval_y, probs)
    expert_map = load_expert_map(args.expert_csv, assume_correct=args.assume_expert_correct)
    records = attach_expert(records, expert_map, assume_correct=args.assume_expert_correct or args.expert_csv is None)
    threshold = args.threshold if args.threshold is not None else select_threshold_for_budget(records, args.target_expert_rate)
    curve = sweep_single_threshold(records)

    metadata_rows = [
        row_from_score(
            task_id=row["task_id"],
            score=row["score"],
            self_passed=row["self_passed"],
            threshold=threshold,
            method=args.method,
            domain=args.domain,
            model_slug=paths.model_slug,
            dataset_slug=paths.dataset_slug,
            route_self="post_self" if args.method == "Post-Linear" else "post_self",
            route_defer="post_defer" if args.method == "Post-Linear" else "early_defer",
            expert_passed=row.get("expert_passed", ""),
            stage="post" if args.method == "Post-Linear" else "pre",
        )
        for row in records
    ]

    write_metadata(paths.eval_output / "metadata.csv", metadata_rows)
    write_rows_csv(paths.eval_output / "threshold_curve.csv", curve)
    write_rows_csv(paths.outputs / "scores.csv", records)
    with (paths.eval_output / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({**config, "threshold": threshold}, f, ensure_ascii=False, indent=2)
    print(f"Saved probe to {paths.ckpt}")
    print(f"Saved metadata to {paths.eval_output / 'metadata.csv'}")
    print(f"Selected threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
