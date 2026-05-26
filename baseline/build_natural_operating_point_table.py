#!/usr/bin/env python3
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
    "Post-Linear",
    "Pre-Linear",
    "Roberta-Router",
    "External-Prompt-Router",
    "Answer-Probability",
]
DISPLAY_LABEL_MAP = {
    "MC-Two-Stage-Probe": "MC Two-Stage Probe",
    "External-Prompt-Router": "Prompt Router",
    "Answer-Probability": "Answer Prob",
    "Roberta-Router": "RoBERTa Router",
    "Post-Linear": "Post Linear",
    "Pre-Linear": "Pre Linear",
}
DOMAIN_ORDER = ["code", "math", "mmlu"]
DOMAIN_TITLE = {"code": "Code", "math": "Math", "mmlu": "MMLU"}


@dataclass
class EvalRow:
    task_id: str
    route: str
    model_decision: str
    self_passed: float
    expert_passed: float
    p_defer: float
    p_self: float
    score: float


def robust_read_csv(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        return [{str(k).strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_route(route: str | None) -> str:
    value = str(route or "").strip().lower()
    return value or "other"


def is_deferred_route(route: str) -> bool:
    return route in {"early_defer", "post_defer"}


def parse_kv_spec(spec: str) -> tuple[str, Path]:
    text = str(spec).strip()
    if "=" in text:
        key, value = text.split("=", 1)
        if key.strip() and value.strip():
            return key.strip(), Path(value.strip())
    return "", Path(text)


def ordered(values: set[str], priority: list[str]) -> list[str]:
    front = [item for item in priority if item in values]
    tail = sorted(item for item in values if item not in set(priority))
    return front + tail


def display_label(label: str) -> str:
    return DISPLAY_LABEL_MAP.get(label, label)


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
        model_decision = str(row.get("model_decision", "")).strip().lower()

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
                model_decision=model_decision,
                self_passed=self_passed,
                expert_passed=expert_passed,
                p_defer=p_defer,
                p_self=p_self,
                score=score,
            )
        )
    return merged


def normalize_self_ref_rows_to_score_decision(rows: list[EvalRow]) -> list[EvalRow]:
    out: list[EvalRow] = []
    for row in rows:
        model_decision = "defer" if row.p_defer >= row.p_self else "self"
        out.append(
            EvalRow(
                task_id=row.task_id,
                route="post_defer" if model_decision == "defer" else "post_self",
                model_decision=model_decision,
                self_passed=row.self_passed,
                expert_passed=row.expert_passed,
                p_defer=row.p_defer,
                p_self=row.p_self,
                score=row.score,
            )
        )
    return out


def apply_answer_probability_threshold(rows: list[EvalRow]) -> list[EvalRow]:
    if not rows:
        return rows

    scores = np.asarray([row.score for row in rows], dtype=float)
    self_passed = np.asarray([row.self_passed for row in rows], dtype=float)
    should_defer = (self_passed < 0.5).astype(float)
    unique_scores = np.unique(scores)
    candidates: list[float] = [-1e-6]
    candidates.extend(float(x) for x in unique_scores.tolist())
    candidates.append(1.000001)

    wrong_total = float(should_defer.sum())
    correct_total = float(len(rows) - wrong_total)
    target_wrong_rate = wrong_total / max(float(len(rows)), 1.0)

    best_threshold = candidates[0]
    best_balanced_acc = -1.0
    best_accuracy = -1.0
    best_rate_gap = float("inf")

    for threshold in candidates:
        predicted_defer = (scores < threshold).astype(float)
        tp = float(((predicted_defer == 1.0) & (should_defer == 1.0)).sum())
        tn = float(((predicted_defer == 0.0) & (should_defer == 0.0)).sum())
        tpr = tp / wrong_total if wrong_total > 0 else 1.0
        tnr = tn / correct_total if correct_total > 0 else 1.0
        balanced_acc = 0.5 * (tpr + tnr)
        route_acc = float((predicted_defer == should_defer).mean())
        defer_rate = float(predicted_defer.mean())
        rate_gap = abs(defer_rate - target_wrong_rate)

        better = False
        if balanced_acc > best_balanced_acc + 1e-12:
            better = True
        elif abs(balanced_acc - best_balanced_acc) <= 1e-12 and route_acc > best_accuracy + 1e-12:
            better = True
        elif (
            abs(balanced_acc - best_balanced_acc) <= 1e-12
            and abs(route_acc - best_accuracy) <= 1e-12
            and rate_gap < best_rate_gap - 1e-12
        ):
            better = True

        if better:
            best_balanced_acc = balanced_acc
            best_accuracy = route_acc
            best_threshold = threshold
            best_rate_gap = rate_gap

    out: list[EvalRow] = []
    for row in rows:
        model_decision = "defer" if row.score < best_threshold else "self"
        out.append(
            EvalRow(
                task_id=row.task_id,
                route="post_defer" if model_decision == "defer" else "post_self",
                model_decision=model_decision,
                self_passed=row.self_passed,
                expert_passed=row.expert_passed,
                p_defer=row.p_defer,
                p_self=row.p_self,
                score=row.score,
            )
        )
    return out


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


def compute_full_budget_curve(rows: list[EvalRow]) -> dict[str, float | np.ndarray]:
    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "route_rates": np.asarray([0.0], dtype=float),
            "accuracies": np.asarray([0.0], dtype=float),
        }

    prepared = sorted(rows, key=lambda row: (row.score, row.task_id))
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
    }


def compute_auc(rows: list[EvalRow]) -> float:
    curve = compute_full_budget_curve(rows)
    x = np.asarray(curve["route_rates"], dtype=float) / 100.0
    y = np.asarray(curve["accuracies"], dtype=float)
    if len(x) <= 1:
        return float("nan")
    return float(np.trapezoid(y, x))


def compute_natural_stats(rows: list[EvalRow]) -> dict[str, float]:
    total = len(rows)
    if total == 0:
        return {"accuracy": 0.0, "defer": 0.0}

    n_defer = 0
    success = 0.0
    for row in rows:
        did_defer = is_deferred_route(row.route)
        if did_defer:
            n_defer += 1
            success += row.expert_passed
        else:
            success += row.self_passed

    return {
        "accuracy": (success / total) * 100.0,
        "defer": (n_defer / total) * 100.0,
    }


def fmt(value: float | None, digits: int) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and np.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def build_markdown(*, title: str, results: dict[str, dict[str, dict[str, float]]], method_order: list[str]) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- Defer is the expert-call rate at each method's natural operating point.")
    lines.append("")

    header = (
        "| Method | "
        "Code AUC | Code ECE | Code ACC | Code Defer | "
        "Math AUC | Math ECE | Math ACC | Math Defer | "
        "MMLU AUC | MMLU ECE | MMLU ACC | MMLU Defer |"
    )
    sep = "|---|" + "---:|" * 12
    lines.append(header)
    lines.append(sep)

    for method in method_order:
        row_cells = [display_label(method)]
        for domain in DOMAIN_ORDER:
            metrics = results.get(method, {}).get(domain)
            if not metrics:
                row_cells.extend(["NA", "NA", "NA", "NA"])
                continue
            row_cells.extend(
                [
                    fmt(metrics["auc"], 3),
                    fmt(metrics["ece"], 4),
                    fmt(metrics["acc"], 2),
                    fmt(metrics["defer"], 2),
                ]
            )
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full natural operating-point table across Code/Math/MMLU.")
    parser.add_argument("--code_metadata", action="append", required=True, help="Repeatable. LABEL=/path/to/metadata.csv")
    parser.add_argument("--math_metadata", action="append", required=True, help="Repeatable. LABEL=/path/to/metadata.csv")
    parser.add_argument("--mmlu_metadata", action="append", required=True, help="Repeatable. LABEL=/path/to/metadata.csv")
    parser.add_argument("--code_expert_csv", required=True)
    parser.add_argument("--math_expert_csv", required=True)
    parser.add_argument("--mmlu_expert_csv", required=True)
    parser.add_argument("--title", default="Full Natural Operating-Point Results")
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--assume_expert_correct", action="store_true")
    parser.add_argument("--ece_bins", type=int, default=10)
    parser.add_argument("--answer_prob_target_expert_rate", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    domain_specs = {
        "code": (args.code_metadata, args.code_expert_csv),
        "math": (args.math_metadata, args.math_expert_csv),
        "mmlu": (args.mmlu_metadata, args.mmlu_expert_csv),
    }

    seen_methods: set[str] = set()
    per_method_domain: dict[str, dict[str, dict[str, float]]] = {}

    for domain, (metadata_specs, expert_csv) in domain_specs.items():
        for spec in metadata_specs:
            method, path = parse_kv_spec(spec)
            if not path.exists():
                print(f"[skip] file not found: {path}")
                continue
            if not method:
                method = path.parent.name or path.stem

            rows = load_eval_rows(
                path,
                expert_csv=expert_csv,
                assume_expert_correct=args.assume_expert_correct,
            )
            if not rows:
                print(f"[skip] empty rows: {path}")
                continue

            if method == "Self-REF":
                rows = normalize_self_ref_rows_to_score_decision(rows)
            if method == "Answer-Probability":
                rows = apply_answer_probability_threshold(rows)

            should_defer = [1.0 if row.self_passed < 0.5 else 0.0 for row in rows]
            ece = compute_ece([row.p_defer for row in rows], should_defer, args.ece_bins)
            auc = compute_auc(rows)
            natural = compute_natural_stats(rows)

            per_method_domain.setdefault(method, {})[domain] = {
                "auc": auc,
                "ece": ece,
                "acc": natural["accuracy"],
                "defer": natural["defer"],
            }
            seen_methods.add(method)

    if not per_method_domain:
        raise SystemExit("No valid metadata rows found.")

    method_order = ordered(seen_methods, METHOD_ORDER)
    markdown = build_markdown(title=args.title, results=per_method_domain, method_order=method_order)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown + "\n", encoding="utf-8")
    print(f"Saved markdown: {out_md}")


if __name__ == "__main__":
    main()
