#!/usr/bin/env python3
# Joint-cost analysis at matched-quality operating points.
#
# C_alpha = ExpertRate + alpha * AttemptRate
#
# AttemptRate is method-kind dependent:
#   * two_stage (PBDD, MC Two-Stage Probe):
#       100 - (#early_defer rows in the top-k deferred set) / total * 100
#   * post-generation routers (Self-REF, Post Linear, Answer Prob, Prompt Router):
#       100  (every query pays a local generation before the router decides)
#   * pre-generation routers (Pre Linear, Roberta Router):
#       100 - ExpertRate  (queries routed to the expert never run a local attempt)
#
# Example:
# python build_joint_cost_report.py \
#   --metadata PBDD=/path/to/metadata.csv \
#   --metadata Self-REF=/path/to/metadata.csv \
#   --expert_csv /path/to/expert_results.csv \
#   --target_fractions 95 --alphas 0.05,0.1,0.2 \
#   --title "Joint Cost | Code | Qwen" \
#   --out_md figures/joint_cost/code_qwen.md

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from build_route_markdown_report import (
    DISPLAY_LABEL_MAP,
    METHOD_ORDER,
    compute_full_budget_curve,
    display_label,
    is_self_ref_run,
    load_eval_rows,
    normalize_self_ref_rows_to_score_decision,
    ordered,
    parse_kv_spec,
    parse_target_fractions,
)


DEFAULT_METHOD_KIND = {
    "PBDD": "two_stage",
    "MC-Two-Stage-Probe": "two_stage",
    "Naive-Two-Stage-Cascade": "two_stage",
    "Self-REF": "post",
    "Post-Linear": "post",
    "Answer-Probability": "post",
    "External-Prompt-Router": "post",
    "Pre-Linear": "pre",
    "Roberta-Router": "pre",
}

KIND_DISPLAY = {
    "two_stage": "two-stage",
    "post": "post-gen",
    "pre": "pre-gen",
}


def parse_kind_spec(spec: str) -> tuple[str, str]:
    text = str(spec).strip()
    if "=" not in text:
        raise ValueError(f"--method_kind expects LABEL=KIND, got: {spec}")
    label, kind = text.split("=", 1)
    label = label.strip()
    kind = kind.strip().lower()
    if kind not in {"two_stage", "post", "pre"}:
        raise ValueError(f"Unknown kind '{kind}' for {label}; use one of two_stage|post|pre")
    return label, kind


def parse_alpha_list(text: str) -> list[float]:
    values: list[float] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if value < 0:
            raise ValueError(f"alpha must be non-negative: {token}")
        values.append(value)
    if not values:
        raise ValueError("No valid alpha parsed.")
    return values


def compute_target_breakdown(
    rows,
    target_accuracy: float,
) -> dict[str, float]:
    """At the smallest counterfactual route_rate that meets target_accuracy,
    report (k, route_rate, achieved_acc, early_in_top_k_count, post_defer_in_top_k_count)."""
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "k": 0,
            "route_rate": float("nan"),
            "achieved_accuracy": float("nan"),
            "early_in_top_k": 0,
            "post_defer_in_top_k": 0,
        }

    prepared = sorted(rows, key=lambda row: (row.score, row.task_id))
    self_passed = np.asarray([row.self_passed for row in prepared], dtype=float)
    expert_passed = np.asarray([row.expert_passed for row in prepared], dtype=float)
    self_prefix = np.concatenate(([0.0], np.cumsum(self_passed)))
    expert_prefix = np.concatenate(([0.0], np.cumsum(expert_passed)))
    total_self = float(self_passed.sum())

    accuracies = np.asarray(
        [
            (expert_prefix[k] + (total_self - self_prefix[k])) / total * 100.0
            for k in range(total + 1)
        ],
        dtype=float,
    )
    hit = np.flatnonzero(accuracies >= target_accuracy)
    if len(hit) == 0:
        k = int(np.argmax(accuracies))
    else:
        k = int(hit[0])

    deferred = prepared[:k]
    early_in_top_k = sum(1 for row in deferred if row.route == "early_defer")
    post_defer_in_top_k = sum(1 for row in deferred if row.route == "post_defer")

    return {
        "total": total,
        "k": k,
        "route_rate": k / total * 100.0,
        "achieved_accuracy": float(accuracies[k]),
        "early_in_top_k": early_in_top_k,
        "post_defer_in_top_k": post_defer_in_top_k,
    }


def attempt_rate_for_kind(kind: str, expert_rate: float, breakdown: dict[str, float]) -> float:
    if kind == "two_stage":
        total = int(breakdown["total"])
        early = int(breakdown["early_in_top_k"])
        if total <= 0:
            return float("nan")
        return 100.0 - (early / total) * 100.0
    if kind == "post":
        return 100.0
    if kind == "pre":
        return max(0.0, 100.0 - expert_rate)
    raise ValueError(f"Unknown kind: {kind}")


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "NA"
    return f"{value:.{digits}f}"


def build_markdown(
    *,
    title: str,
    expert_only_accuracy: float,
    reference_method: str,
    target_specs: list[dict],
    alphas: list[float],
    results: list[dict],
) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"- Reference method for target ACC: {display_label(reference_method)} "
        f"(expert-only ACC = {fmt(expert_only_accuracy, 2)}%)"
    )
    lines.append(
        "- Target fractions: "
        + ", ".join(f"{fmt(spec['target_fraction'] * 100.0, 0)}%" for spec in target_specs)
    )
    lines.append("- Alphas (relative cost of one local attempt vs. one expert call): "
                 + ", ".join(f"{a:g}" for a in alphas))
    lines.append("- AttemptRate convention: two-stage = 100 - pre-defer@cutoff; "
                 "post-gen = 100; pre-gen = 100 - ExpertRate.")
    lines.append("- Cost unit: expert-call-equivalent count per 100 queries (not a probability). "
                 "Values can exceed 100 because a post-gen router may pay both a local attempt "
                 "and an expert call on the same query.")
    lines.append("")

    lines.append("## Source Metadata")
    lines.append("")
    lines.append("| Method | Kind | Metadata Path |")
    lines.append("|---|---|---|")
    for row in results:
        lines.append(
            f"| {display_label(row['method'])} "
            f"| {KIND_DISPLAY.get(row['kind'], row['kind'])} "
            f"| `{row['path']}` |"
        )
    lines.append("")

    for table_idx, spec in enumerate(target_specs, start=1):
        target_fraction = float(spec["target_fraction"])
        target_accuracy = float(spec["target_accuracy"])
        lines.append("")
        lines.append(
            f"## Table {table_idx}: Joint Cost at Route@{fmt(target_fraction * 100.0, 0)}% "
            f"(target ACC = {fmt(target_accuracy, 2)}%)"
        )
        lines.append("")
        cost_headers = " | ".join(f"Cost@α={a:g}" for a in alphas)
        cost_sep = "---:|" * len(alphas)
        lines.append(
            "| Method | Kind | ExpertRate (%) | AttemptRate (%) | "
            f"{cost_headers} | Achieved ACC (%) |"
        )
        lines.append("|---|---|---:|---:|" + cost_sep + "---:|")
        for row in results:
            entry = row["targets"][table_idx - 1]
            cost_cells = " | ".join(fmt(entry["costs"][a], 2) for a in alphas)
            lines.append(
                "| "
                + f"{display_label(row['method'])} | "
                + f"{KIND_DISPLAY.get(row['kind'], row['kind'])} | "
                + f"{fmt(entry['expert_rate'], 2)} | "
                + f"{fmt(entry['attempt_rate'], 2)} | "
                + f"{cost_cells} | "
                + f"{fmt(entry['achieved_acc'], 2)} |"
            )
        lines.append("")

        # Per-target ranking summary at the middle alpha for narrative anchor.
        mid_alpha = alphas[len(alphas) // 2]
        sorted_for_alpha = sorted(
            results,
            key=lambda r: r["targets"][table_idx - 1]["costs"][mid_alpha],
        )
        ranked = ", ".join(
            f"{display_label(r['method'])} ({fmt(r['targets'][table_idx - 1]['costs'][mid_alpha], 2)})"
            for r in sorted_for_alpha
        )
        lines.append(f"- Ranking by Cost@α={mid_alpha:g}: {ranked}")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        f"- ExpertRate is the smallest counterfactual route rate (sweep over the method's score) "
        f"that meets the target accuracy."
    )
    lines.append(
        "- For two-stage methods, AttemptRate counts queries that pay the local-generation cost: "
        "this excludes pre-deferred queries that get routed to the expert at the chosen cutoff, "
        "and includes pre-deferred queries that fall back to a local attempt because they did not "
        "clear the cutoff."
    )
    lines.append(
        "- Post-gen routers always run a local attempt before deciding to defer, so AttemptRate = 100."
    )
    lines.append(
        "- Pre-gen routers decide before generating, so AttemptRate = 100 - ExpertRate."
    )
    lines.append(
        "- Cost is reported in expert-call-equivalent units per 100 queries (not a probability), "
        "so values can exceed 100: a post-gen router pays one local attempt plus one expert call "
        "on every routed query, contributing 1 + α expert-call-equivalent units."
    )
    lines.append(
        "- Cost is a derived deployment metric; no extra model calls are required."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate joint-cost markdown report at matched-quality operating points."
    )
    parser.add_argument(
        "--metadata",
        action="append",
        required=True,
        help="Repeatable. LABEL=/path/to/metadata.csv.",
    )
    parser.add_argument("--expert_csv", default=None)
    parser.add_argument("--assume_expert_correct", action="store_true")
    parser.add_argument(
        "--method_kind",
        action="append",
        default=[],
        help="Override default kind. Repeatable. e.g. PBDD=two_stage Self-REF=post Pre-Linear=pre",
    )
    parser.add_argument("--target_reference_method", default="PBDD")
    parser.add_argument("--target_fractions", default="95")
    parser.add_argument("--alphas", default="0.05,0.1,0.2")
    parser.add_argument("--title", default="Joint Cost Report")
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    target_fractions = parse_target_fractions(args.target_fractions)
    alphas = parse_alpha_list(args.alphas)

    overrides: dict[str, str] = {}
    for spec in args.method_kind:
        label, kind = parse_kind_spec(spec)
        overrides[label] = kind

    parsed_entries: list[tuple[str, Path]] = []
    labels_seen: set[str] = set()
    for spec in args.metadata:
        label, path = parse_kv_spec(spec)
        if not path.exists():
            print(f"[skip] file not found: {path}")
            continue
        if not label:
            label = path.parent.name or path.stem
        parsed_entries.append((label, path))
        labels_seen.add(label)
    if not parsed_entries:
        raise SystemExit("No valid metadata paths.")

    method_order = ordered(labels_seen, METHOD_ORDER)
    rank = {method: idx for idx, method in enumerate(method_order)}

    loaded: list[dict] = []
    for method, path in parsed_entries:
        rows = load_eval_rows(
            path,
            expert_csv=args.expert_csv,
            assume_expert_correct=args.assume_expert_correct,
        )
        if is_self_ref_run(method, path):
            rows = normalize_self_ref_rows_to_score_decision(rows)
        if not rows:
            print(f"[skip] empty rows: {path}")
            continue
        kind = overrides.get(method, DEFAULT_METHOD_KIND.get(method))
        if kind is None:
            print(f"[warn] no kind for {method}; defaulting to post")
            kind = "post"
        loaded.append(
            {
                "method": method,
                "path": str(path),
                "rows": rows,
                "kind": kind,
                "full_curve": compute_full_budget_curve(rows),
            }
        )
    if not loaded:
        raise SystemExit("No valid loaded results.")

    loaded.sort(key=lambda item: rank.get(item["method"], 10**9))

    reference_entry = next(
        (item for item in loaded if item["method"] == args.target_reference_method),
        loaded[0],
    )
    expert_only_accuracy = float(reference_entry["full_curve"]["expert_only_accuracy"])
    target_specs = [
        {"target_fraction": tf, "target_accuracy": expert_only_accuracy * tf}
        for tf in target_fractions
    ]

    for item in loaded:
        target_entries: list[dict] = []
        for spec in target_specs:
            breakdown = compute_target_breakdown(item["rows"], float(spec["target_accuracy"]))
            expert_rate = breakdown["route_rate"]
            attempt_rate = attempt_rate_for_kind(item["kind"], expert_rate, breakdown)
            costs = {a: expert_rate + a * attempt_rate for a in alphas}
            target_entries.append(
                {
                    "expert_rate": expert_rate,
                    "attempt_rate": attempt_rate,
                    "achieved_acc": breakdown["achieved_accuracy"],
                    "k": breakdown["k"],
                    "costs": costs,
                }
            )
        item["targets"] = target_entries

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(
        title=args.title,
        expert_only_accuracy=expert_only_accuracy,
        reference_method=reference_entry["method"],
        target_specs=target_specs,
        alphas=alphas,
        results=loaded,
    )
    out_md.write_text(markdown + "\n", encoding="utf-8")
    print(f"Saved markdown: {out_md}")


if __name__ == "__main__":
    main()
