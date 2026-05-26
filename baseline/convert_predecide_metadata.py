#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def robust_read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{str(k).strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def map_route(route: str, model_decision: str, p_defer: float) -> tuple[str, str]:
    route_norm = str(route or "").strip().lower()
    decision = str(model_decision or "").strip().lower()

    if decision not in {"self", "defer"}:
        decision = "defer" if p_defer > 0.5 else "self"

    if route_norm in {"pre_defer", "early_defer", "post_defer"}:
        return "early_defer", "defer"
    if route_norm in {"pre_self", "post_self"}:
        return "post_self", "self"

    if decision == "defer":
        return "early_defer", "defer"
    return "post_self", "self"


def convert_rows(
    rows: list[dict[str, str]],
    *,
    method: str,
    domain: str,
    model_slug: str,
    dataset_slug: str,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        p_self = to_float(row.get("p_self"), 1.0 - to_float(row.get("p_defer"), 0.0))
        p_defer = to_float(row.get("p_defer"), 1.0 - p_self)
        margin = to_float(row.get("margin"), 0.0)
        route, model_decision = map_route(row.get("route", ""), row.get("model_decision", ""), p_defer)

        converted = {
            "task_id": str(row.get("task_id", "")).strip(),
            "route": route,
            "model_decision": model_decision,
            "first_p_defer": round(p_defer, 6),
            "first_p_self": round(p_self, 6),
            "first_margin": round(margin, 4),
            "post_p_defer": "",
            "post_p_self": "",
            "post_margin": "",
            "p_defer": round(p_defer, 6),
            "p_self": round(p_self, 6),
            "margin": round(margin, 4),
            "self_passed": int(to_float(row.get("self_passed"), 0.0) > 0.5),
            "expert_passed": row.get("expert_passed", ""),
            "answer_len": int(to_float(row.get("answer_len"), 0.0)),
            "actual_local_tokens": int(to_float(row.get("actual_local_tokens"), 0.0)),
            "method": method,
            "domain": domain,
            "model_slug": model_slug,
            "dataset_slug": dataset_slug,
            "score": round(p_self, 6),
        }
        out.append(converted)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "route",
        "model_decision",
        "first_p_defer",
        "first_p_self",
        "first_margin",
        "post_p_defer",
        "post_p_self",
        "post_margin",
        "p_defer",
        "p_self",
        "margin",
        "self_passed",
        "expert_passed",
        "answer_len",
        "actual_local_tokens",
        "method",
        "domain",
        "model_slug",
        "dataset_slug",
        "score",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert baseline_pre_decide metadata.csv to current unified schema.")
    p.add_argument("--input", required=True, help="Path to original metadata.csv")
    p.add_argument("--output", required=True, help="Path to converted metadata.csv")
    p.add_argument("--method", default="Pre-Decide", help="Method name to write into output metadata.")
    p.add_argument("--domain", default="", help="Domain to write (code/math/mmlu).")
    p.add_argument("--model_slug", default="", help="Model slug to write (e.g., qwen2.5-coder-3b-instruct).")
    p.add_argument("--dataset_slug", default="", help="Dataset slug to write.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = robust_read_csv(input_path)
    converted = convert_rows(
        rows,
        method=args.method,
        domain=args.domain,
        model_slug=args.model_slug,
        dataset_slug=args.dataset_slug,
    )
    write_csv(output_path, converted)
    print(f"Converted {len(converted)} rows")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
