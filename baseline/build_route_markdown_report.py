#!/usr/bin/env python3
# Example:
# python build_route_markdown_report.py \
#   --metadata PBDD=/path/to/pbdd_metadata.csv \
#   --metadata Self-REF=/path/to/self_ref_metadata.csv \
#   --expert_csv /path/to/expert_results.csv \
#   --out_md figures/route_tables/report_code_qwen.md \
#   --title "Route Table | Code | Qwen"

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


METHOD_ORDER = [
    "PBDD",
    "Self-REF",
    "MC-Two-Stage-Probe",
    "Naive-Two-Stage-Cascade",
    "Post-Linear",
    "Pre-Linear",
    "Roberta-Router",
    "External-Prompt-Router",
    "Answer-Probability",
]
DISPLAY_LABEL_MAP = {
    "MC-Two-Stage-Probe": "MC Two-Stage Probe",
    "Naive-Two-Stage-Cascade": "Two-Stage Cascade",
    "External-Prompt-Router": "Prompt Router",
    "Answer-Probability": "Answer Prob",
    "Roberta-Router": "RoBERTa Router",
    "Post-Linear": "Post Linear",
    "Pre-Linear": "Pre Linear",
}


@dataclass
class EvalRow:
    task_id: str
    route: str
    self_passed: float
    expert_passed: float
    p_defer: float
    p_self: float
    score: float
    first_p_defer: float | None
    post_p_defer: float | None


def robust_read_csv(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        return [{str(k).strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_optional_float(value) -> float | None:
    text = str(value if value is not None else "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_route(route: str | None) -> str:
    value = str(route or "").strip().lower()
    return value or "other"


def is_deferred_route(route: str) -> bool:
    return route in {"early_defer", "post_defer"}


def is_self_ref_run(label: str | None, metadata_path: str | Path) -> bool:
    text = f"{label or ''} {metadata_path}".lower().replace("_", "-")
    return "self-ref" in text


def normalize_self_ref_rows_to_score_decision(rows: list[EvalRow]) -> list[EvalRow]:
    normalized: list[EvalRow] = []
    for row in rows:
        model_decision = "defer" if row.p_defer >= row.p_self else "self"
        normalized.append(
            EvalRow(
                task_id=row.task_id,
                route="post_defer" if model_decision == "defer" else "post_self",
                self_passed=row.self_passed,
                expert_passed=row.expert_passed,
                p_defer=row.p_defer,
                p_self=row.p_self,
                score=row.score,
                first_p_defer=row.first_p_defer,
                post_p_defer=row.post_p_defer,
            )
        )
    return normalized


def parse_kv_spec(spec: str) -> tuple[str, Path]:
    text = str(spec).strip()
    if "=" in text:
        key, value = text.split("=", 1)
        if key.strip() and value.strip():
            return key.strip(), Path(value.strip())
    return "", Path(text)


def display_label(label: str) -> str:
    return DISPLAY_LABEL_MAP.get(label, label)


def ordered(values: set[str], priority: list[str]) -> list[str]:
    front = [item for item in priority if item in values]
    tail = sorted(item for item in values if item not in set(priority))
    return front + tail


def load_expert_map(expert_csv: str | Path | None) -> dict[str, float]:
    if expert_csv is None:
        return {}
    out: dict[str, float] = {}
    for row in robust_read_csv(expert_csv):
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        out[task_id] = to_float(row.get("expert_passed", row.get("self_passed", 0.0)), 0.0)
    return out


def confidence_score(row: EvalRow) -> float:
    return row.score


def load_eval_rows(
    metadata_path: Path,
    *,
    expert_csv: str | Path | None,
    assume_expert_correct: bool,
) -> list[EvalRow]:
    rows = robust_read_csv(metadata_path)
    expert_map = load_expert_map(expert_csv)
    merged: list[EvalRow] = []
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        route = normalize_route(row.get("route"))
        self_passed = to_float(row.get("self_passed"), 0.0)
        p_self = to_float(row.get("p_self"), 0.0)
        p_defer = to_float(row.get("p_defer"), 1.0 - p_self)
        score = to_float(row.get("score"), p_self)
        first_p_defer = parse_optional_float(row.get("first_p_defer"))
        post_p_defer = parse_optional_float(row.get("post_p_defer"))

        if expert_csv is not None:
            expert_passed = expert_map.get(task_id, 0.0)
        else:
            raw = str(row.get("expert_passed", "")).strip()
            if raw != "":
                expert_passed = to_float(raw, 0.0)
            else:
                expert_passed = 1.0 if assume_expert_correct else 0.0

        merged.append(
            EvalRow(
                task_id=task_id,
                route=route,
                self_passed=self_passed,
                expert_passed=expert_passed,
                p_defer=p_defer,
                p_self=p_self,
                score=score,
                first_p_defer=first_p_defer,
                post_p_defer=post_p_defer,
            )
        )
    return merged


def compute_ece(probs: list[float], labels: list[float], n_bins: int) -> float:
    if not probs:
        return float("nan")
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == 0:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p > left) & (p <= right)
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += abs(conf - acc) * (count / n)
    return float(ece)


def compute_budget_stats(rows: list[EvalRow], budgets: list[int]) -> dict[int, dict[str, float]]:
    total = len(rows)
    if total == 0:
        return {
            budget: {
                "accuracy": 0.0,
                "pre_defer_rate": 0.0,
                "post_defer_rate": 0.0,
                "pre_defer_share": 0.0,
            }
            for budget in budgets
        }

    prepared = sorted(rows, key=lambda row: (confidence_score(row), row.task_id))
    out: dict[int, dict[str, float]] = {}
    for budget in budgets:
        n_defer = int(round((budget / 100.0) * total))
        deferred = prepared[:n_defer]
        local = prepared[n_defer:]
        success = sum(row.expert_passed for row in deferred) + sum(row.self_passed for row in local)
        early_cnt = sum(1 for row in deferred if row.route == "early_defer")
        post_cnt = sum(1 for row in deferred if row.route == "post_defer")

        out[budget] = {
            "accuracy": (success / total) * 100.0,
            "pre_defer_rate": (early_cnt / total) * 100.0,
            "post_defer_rate": (post_cnt / total) * 100.0,
            "pre_defer_share": (early_cnt / n_defer) * 100.0 if n_defer > 0 else 0.0,
        }
    return out


def compute_auc_from_budget_stats(budget_stats: dict[int, dict[str, float]], budgets: list[int]) -> float:
    x = np.asarray([b / 100.0 for b in budgets], dtype=float)
    y = np.asarray([budget_stats[b]["accuracy"] for b in budgets], dtype=float)
    if len(x) <= 1:
        return float("nan")
    return float(np.trapz(y, x))


def compute_full_budget_curve(rows: list[EvalRow]) -> dict[str, float | np.ndarray]:
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "route_rates": np.asarray([0.0], dtype=float),
            "accuracies": np.asarray([0.0], dtype=float),
            "expert_only_accuracy": 0.0,
            "model_only_accuracy": 0.0,
        }

    prepared = sorted(rows, key=lambda row: (confidence_score(row), row.task_id))
    self_passed = np.asarray([row.self_passed for row in prepared], dtype=float)
    expert_passed = np.asarray([row.expert_passed for row in prepared], dtype=float)

    self_prefix = np.concatenate(([0.0], np.cumsum(self_passed)))
    expert_prefix = np.concatenate(([0.0], np.cumsum(expert_passed)))
    total_self = float(self_passed.sum())

    route_rates = np.arange(total + 1, dtype=float) / total * 100.0
    accuracies = []
    for k in range(total + 1):
        success = expert_prefix[k] + (total_self - self_prefix[k])
        accuracies.append(success / total * 100.0)

    return {
        "total": total,
        "route_rates": route_rates,
        "accuracies": np.asarray(accuracies, dtype=float),
        "expert_only_accuracy": float(expert_passed.mean() * 100.0),
        "model_only_accuracy": float(self_passed.mean() * 100.0),
    }


def accuracy_at_budget(curve: dict[str, float | np.ndarray], budget: float) -> dict[str, float]:
    route_rates = np.asarray(curve["route_rates"], dtype=float)
    accuracies = np.asarray(curve["accuracies"], dtype=float)
    total = int(curve["total"])
    if total <= 0:
        return {"route_rate": 0.0, "accuracy": 0.0}
    idx = int(round(float(budget) / 100.0 * total))
    idx = max(0, min(total, idx))
    return {"route_rate": float(route_rates[idx]), "accuracy": float(accuracies[idx])}


def compute_target_accuracy_stats_from_curve(
    curve: dict[str, float | np.ndarray],
    target_accuracy: float,
) -> dict[str, float]:
    route_rates = np.asarray(curve["route_rates"], dtype=float)
    accuracies = np.asarray(curve["accuracies"], dtype=float)
    expert_only_accuracy = float(curve["expert_only_accuracy"])
    model_only_accuracy = float(curve["model_only_accuracy"])

    hit_indices = np.flatnonzero(accuracies >= target_accuracy)
    if len(hit_indices) == 0:
        best_idx = int(np.argmax(accuracies))
        return {
            "target_accuracy": target_accuracy,
            "route_rate": float("nan"),
            "achieved_accuracy": float(accuracies[best_idx]),
            "model_only_accuracy": model_only_accuracy,
            "expert_only_accuracy": expert_only_accuracy,
        }

    first_hit = int(hit_indices[0])
    return {
        "target_accuracy": target_accuracy,
        "route_rate": float(route_rates[first_hit]),
        "achieved_accuracy": float(accuracies[first_hit]),
        "model_only_accuracy": model_only_accuracy,
        "expert_only_accuracy": expert_only_accuracy,
    }


def compute_reference_target_specs(
    reference_result: dict,
    target_fractions: list[float],
) -> list[dict[str, float | str]]:
    curve = reference_result["full_curve"]
    expert_only_accuracy = float(curve["expert_only_accuracy"])
    return [
        {
            "reference_method": reference_result["method"],
            "target_fraction": target_fraction,
            "target_accuracy": expert_only_accuracy * target_fraction,
            "expert_only_accuracy": expert_only_accuracy,
        }
        for target_fraction in target_fractions
    ]


def compute_natural_stats(rows: list[EvalRow]) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {
            "natural_budget": 0.0,
            "natural_accuracy": 0.0,
            "natural_route_acc": 0.0,
            "natural_pre_defer_rate": 0.0,
            "natural_post_defer_rate": 0.0,
            "natural_pre_defer_share": 0.0,
        }

    defer_rows = [row for row in rows if is_deferred_route(row.route)]
    early_rows = [row for row in defer_rows if row.route == "early_defer"]
    post_rows = [row for row in defer_rows if row.route == "post_defer"]

    success = 0.0
    route_correct = 0.0
    for row in rows:
        should_defer = row.self_passed < 0.5
        did_defer = is_deferred_route(row.route)
        route_correct += 1.0 if (should_defer == did_defer) else 0.0
        if did_defer:
            success += row.expert_passed
        else:
            success += row.self_passed

    n_defer = len(defer_rows)
    return {
        "natural_budget": (n_defer / total) * 100.0,
        "natural_accuracy": (success / total) * 100.0,
        "natural_route_acc": (route_correct / total) * 100.0,
        "natural_pre_defer_rate": (len(early_rows) / total) * 100.0,
        "natural_post_defer_rate": (len(post_rows) / total) * 100.0,
        "natural_pre_defer_share": (len(early_rows) / n_defer) * 100.0 if n_defer > 0 else 0.0,
    }


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and np.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def build_markdown(
    *,
    title: str,
    budgets: list[int],
    target_specs: list[dict[str, float | str]],
    results: list[dict],
    ece_bins: int,
) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Budgets: {', '.join(str(b) + '%' for b in budgets)}")
    lines.append(
        "- Table 1 targets: "
        + ", ".join(
            f"{fmt(float(spec['target_fraction']) * 100.0, 0)}% of expert-only ACC"
            for spec in target_specs
        )
    )
    lines.append(f"- ECE bins: {ece_bins}")
    lines.append("")
    lines.append("## Source Metadata")
    lines.append("")
    lines.append("| Method | Metadata Path |")
    lines.append("|---|---|")
    for row in results:
        lines.append(f"| {display_label(row['method'])} | `{row['path']}` |")
    lines.append("")

    for table_idx, target_spec in enumerate(target_specs, start=1):
        target_fraction = float(target_spec["target_fraction"])
        target_accuracy = float(target_spec["target_accuracy"])
        expert_only_accuracy = float(target_spec["expert_only_accuracy"])
        lines.append("")
        lines.append(
            f"## Table 1.{table_idx}: Route Rate to Reach "
            f"{fmt(target_fraction * 100.0, 0)}% Expert Accuracy"
        )
        lines.append("")
        lines.append(
            f"- Target ACC: {fmt(target_accuracy, 2)}% "
            f"({fmt(target_fraction * 100.0, 0)}% of expert-only ACC {fmt(expert_only_accuracy, 2)}%)"
        )
        lines.append("")
        lines.append(
            "| Method | Route Rate (%) | Achieved ACC (%) | AUC(Acc-Budget, %) | ECE(defer) |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for row in results:
            target_stats = row["target_stats"][table_idx - 1]
            lines.append(
                "| "
                + f"{display_label(row['method'])} | "
                + f"{fmt(target_stats['route_rate'], 2)} | "
                + f"{fmt(target_stats['achieved_accuracy'], 2)} | "
                + f"{fmt(row['auc'], 3)} | {fmt(row['ece'], 4)} |"
            )

    lines.append("")
    lines.append("## Table 2: Fixed-Budget System ACC (Counterfactual Sweep)")
    lines.append("")
    header = "| Method | AUC(Acc-Budget, %) | ECE(defer) | " + " | ".join(f"ACC@{b}% (%)" for b in budgets) + " |"
    sep = "|---|---:|---:|" + "---:|" * len(budgets)
    lines.append(header)
    lines.append(sep)
    for row in results:
        acc_cols = " | ".join(fmt(row["budget_stats"][b]["accuracy"], 2) for b in budgets)
        lines.append(
            f"| {display_label(row['method'])} | {fmt(row['auc'], 3)} | {fmt(row['ece'], 4)} | {acc_cols} |"
        )

    two_stage = [row for row in results if row["is_two_stage"]]
    if two_stage:
        lines.append("")
        lines.append("## Table 3: Two-Stage Pre-Defer Rate")
        lines.append("")
        header3 = "| Method | " + " | ".join(f"Pre-Defer@{b}% (%)" for b in budgets) + " |"
        sep3 = "|---|" + "---:|" * len(budgets)
        lines.append(header3)
        lines.append(sep3)
        for row in two_stage:
            pre_cols = " | ".join(fmt(row["budget_stats"][b]["pre_defer_rate"], 2) for b in budgets)
            lines.append(f"| {display_label(row['method'])} | {pre_cols} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Routing target for ECE(defer): self wrong => defer, self correct => self.")
    lines.append("- AUC integrates counterfactual accuracy over budgets in [0, 100%].")
    lines.append("- Table 1.x reports the minimum counterfactual route rate needed to match the target accuracy.")
    lines.append("- Table 2 and Table 3 use threshold-sweep counterfactual routing under fixed budget.")
    lines.append("")
    return "\n".join(lines)


def parse_budgets(text: str) -> list[int]:
    budgets: list[int] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value < 0 or value > 100:
            raise ValueError(f"Budget out of range: {value}")
        budgets.append(value)
    if not budgets:
        raise ValueError("No valid budgets parsed.")
    return budgets


def parse_target_fractions(text: str) -> list[float]:
    values: list[float] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if value > 1.0:
            value = value / 100.0
        if value <= 0.0 or value > 1.0:
            raise ValueError(f"Target fraction out of range: {token}")
        values.append(value)
    if not values:
        raise ValueError("No valid target fractions parsed.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown routing report (AUC/ECE/budget table).")
    parser.add_argument(
        "--metadata",
        action="append",
        required=True,
        help="Repeatable. Use LABEL=/path/to/metadata.csv or /path/to/metadata.csv.",
    )
    parser.add_argument("--expert_csv", default=None, help="Optional expert csv for expert_passed.")
    parser.add_argument("--assume_expert_correct", action="store_true")
    parser.add_argument("--budgets", default="0,20,40,60,80,100")
    parser.add_argument("--target_reference_method", default="PBDD")
    parser.add_argument("--target_fractions", default="0.8,0.9")
    parser.add_argument("--pre_threshold", type=float, default=0.5,
                        help="Stage-0 defer threshold for two-stage methods. Queries with first_p_defer > this are early-deferred.")
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--title", default="Routing Report")
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    budgets = parse_budgets(args.budgets)
    target_fractions = parse_target_fractions(args.target_fractions)
    labels_seen: set[str] = set()
    parsed_entries: list[tuple[str, Path]] = []
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

    results: list[dict] = []
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

        # Recompute final score/route for two-stage methods based on --pre_threshold
        has_two_stage_scores = any(row.first_p_defer is not None and row.post_p_defer is not None for row in rows)
        if has_two_stage_scores and not is_self_ref_run(method, path):
            recomputed: list[EvalRow] = []
            for row in rows:
                if row.first_p_defer is not None and row.post_p_defer is not None:
                    if row.first_p_defer > args.pre_threshold:
                        new_route = "early_defer"
                        new_p_defer = row.first_p_defer
                        new_p_self = 1.0 - row.first_p_defer
                    else:
                        new_p_defer = row.post_p_defer
                        new_p_self = 1.0 - row.post_p_defer
                        new_route = "post_defer" if new_p_defer >= new_p_self else "post_self"
                    recomputed.append(EvalRow(
                        task_id=row.task_id,
                        route=new_route,
                        self_passed=row.self_passed,
                        expert_passed=row.expert_passed,
                        p_defer=new_p_defer,
                        p_self=new_p_self,
                        score=new_p_self,
                        first_p_defer=row.first_p_defer,
                        post_p_defer=row.post_p_defer,
                    ))
                else:
                    recomputed.append(row)
            rows = recomputed

        should_defer = [1.0 if row.self_passed < 0.5 else 0.0 for row in rows]
        ece = compute_ece([row.p_defer for row in rows], should_defer, args.ece_bins)

        budget_stats = compute_budget_stats(rows, budgets)
        natural_stats = compute_natural_stats(rows)
        auc = compute_auc_from_budget_stats(budget_stats, budgets)
        full_curve = compute_full_budget_curve(rows)
        routes = {row.route for row in rows}
        is_two_stage = ("early_defer" in routes) and ("post_defer" in routes)

        results.append(
            {
                "method": method,
                "path": str(path),
                "auc": auc,
                "ece": ece,
                "budget_stats": budget_stats,
                "natural_stats": natural_stats,
                "full_curve": full_curve,
                "is_two_stage": is_two_stage,
            }
        )

    if not results:
        raise SystemExit("No valid results to report.")

    results.sort(key=lambda item: rank.get(item["method"], 10**9))
    reference_result = next(
        (row for row in results if row["method"] == args.target_reference_method),
        results[0],
    )
    target_specs = compute_reference_target_specs(reference_result, target_fractions)

    for row in results:
        row["target_stats"] = [
            compute_target_accuracy_stats_from_curve(row["full_curve"], float(spec["target_accuracy"]))
            for spec in target_specs
        ]

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(
        title=args.title,
        budgets=budgets,
        target_specs=target_specs,
        results=results,
        ece_bins=args.ece_bins,
    )
    out_md.write_text(markdown + "\n", encoding="utf-8")
    print(f"Saved markdown: {out_md}")


if __name__ == "__main__":
    main()
