#!/usr/bin/env python3
# Example 1:
# python plot_paper_main_figure.py \
#   --metadata PBDD=/path/to/pbdd_metadata.csv \
#   --metadata Self-REF=/path/to/self_ref_metadata.csv \
#   --metadata AnsProb=/path/to/answer_prob_metadata.csv \
#   --metadata Roberta=/path/to/roberta_router_metadata.csv \
#   --out figures/main_route_figure.png
#
# Example 2 (mixed domains and models; missing combinations are skipped automatically):
# python plot_paper_main_figure.py \
#   --metadata PBDD-Qwen-Code=/path/to/pbdd_qwen_code_metadata.csv \
#   --metadata PBDD-Llama-Code=/path/to/pbdd_llama_code_metadata.csv \
#   --metadata PBDD-Qwen-Math=/path/to/pbdd_qwen_math_metadata.csv \
#   --metadata PBDD-Llama-Math=/path/to/pbdd_llama_math_metadata.csv \
#   --metadata PBDD-Qwen-MMLU=/path/to/pbdd_qwen_mmlu_metadata.csv \
#   --metadata PBDD-Llama-MMLU=/path/to/pbdd_llama_mmlu_metadata.csv \
#   --out figures/main_route_figure_all.png

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DOMAIN_ORDER = ["code", "math", "mmlu"]
MODEL_ORDER = ["qwen", "llama", "other"]
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
METHOD_COLOR_MAP = {
    "PBDD": "#1f77b4",
    "Self-REF": "#d62728",
    "MC-Two-Stage-Probe": "#ff9896",
    "Naive-Two-Stage-Cascade": "#2ca02c",
    "Post-Linear": "#9467bd",
    "Pre-Linear": "#ff7f0e",
    "Roberta-Router": "#8c564b",
    "External-Prompt-Router": "#17becf",
    "Answer-Probability": "#e377c2",
}
FALLBACK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#17becf",
]
DISPLAY_LABEL_MAP = {
    "MC-Two-Stage-Probe": "MC Two-Stage Probe",
    "Naive-Two-Stage-Cascade": "Two-Stage Cascade",
    "External-Prompt-Router": "Prompt Router",
    "Answer-Probability": "Answer Prob",
    "Roberta-Router": "RoBERTa Router",
    "Post-Linear": "Post Linear",
    "Pre-Linear": "Pre Linear",
    "Always-Self": "Always self",
    "Always-Defer": "Always defer",
}


def robust_read_csv(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        return [{str(k).strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def optional_float(value) -> float | None:
    text = str(value if value is not None else "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_kv_spec(spec: str) -> tuple[str, Path]:
    text = str(spec).strip()
    if "=" in text:
        key, value = text.split("=", 1)
        if key.strip() and value.strip():
            return key.strip(), Path(value.strip())
    return "", Path(text)


def resolve_curve_path(label: str, metadata_path: Path) -> Path:
    if label == "Naive-Two-Stage-Cascade" and metadata_path.name == "metadata.csv":
        pareto_path = metadata_path.with_name("pareto_envelope.csv")
        if pareto_path.is_file():
            return pareto_path
        threshold_path = metadata_path.with_name("threshold_grid.csv")
        if threshold_path.is_file():
            return threshold_path
    return metadata_path


def normalize_domain(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if "mmlu" in text:
        return "mmlu"
    if "math" in text:
        return "math"
    if "code" in text or "humaneval" in text or "mbpp" in text or "leetcode" in text:
        return "code"
    return text


def normalize_model_family(value: str) -> str:
    text = str(value or "").strip().lower()
    if "qwen" in text:
        return "qwen"
    if "llama" in text:
        return "llama"
    return "other"


def infer_domain(rows: list[dict[str, str]], metadata_path: Path) -> str:
    for row in rows:
        domain = normalize_domain(row.get("domain", ""))
        if domain:
            return domain
        domain = normalize_domain(row.get("dataset_slug", ""))
        if domain:
            return domain

    task_ids = [str(row.get("task_id", "")).strip() for row in rows[:30]]
    if any(task_id.startswith("mmlu/") for task_id in task_ids):
        return "mmlu"
    if any(task_id.startswith("hendrycks_math/") for task_id in task_ids):
        return "math"
    if any(task_id.startswith("math/") for task_id in task_ids):
        return "math"
    if task_ids and all(task_id.isdigit() for task_id in task_ids if task_id):
        return "code"

    path_text = str(metadata_path).lower()
    if "mmlu" in path_text:
        return "mmlu"
    if "math" in path_text:
        return "math"
    if "code" in path_text:
        return "code"
    return "other"


def infer_model_family(rows: list[dict[str, str]], metadata_path: Path) -> str:
    for row in rows:
        family = normalize_model_family(row.get("model_slug", ""))
        if family != "other":
            return family

    path_text = str(metadata_path).lower()
    if "qwen" in path_text:
        return "qwen"
    if "llama" in path_text:
        return "llama"
    return "other"


def infer_label(rows: list[dict[str, str]], metadata_path: Path, label_override: str) -> str:
    if label_override:
        return label_override
    for row in rows:
        method = str(row.get("method", "")).strip()
        if method:
            return method
    return metadata_path.parent.name or metadata_path.stem


def load_expert_map(expert_csv: str | Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in robust_read_csv(expert_csv):
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        out[task_id] = to_float(row.get("expert_passed", row.get("self_passed", 0.0)), 0.0)
    return out


def normalize_route(route: str | None) -> str:
    value = str(route or "").strip().lower()
    return value or "other"


def is_deferred_row(row: dict[str, str | float]) -> bool:
    route = normalize_route(str(row.get("route", "")))
    if route in {"early_defer", "post_defer"}:
        return True
    if route == "post_self":
        return False
    decision = str(row.get("model_decision", "")).strip().lower()
    if decision:
        return decision == "defer"
    return to_float(row.get("p_defer"), 0.0) > 0.5


def normalize_rows_to_score_decision(rows: list[dict[str, str | float]]) -> None:
    for row in rows:
        p_defer = optional_float(row.get("p_defer"))
        p_self = optional_float(row.get("p_self"))
        if p_defer is None and p_self is None:
            continue
        if p_defer is None:
            p_defer = 1.0 - float(p_self)
        if p_self is None:
            p_self = 1.0 - float(p_defer)
        model_decision = "defer" if p_defer >= p_self else "self"
        row["model_decision_raw"] = row.get("model_decision", "")
        row["route_raw"] = row.get("route", "")
        row["model_decision"] = model_decision
        row["route"] = "post_defer" if model_decision == "defer" else "post_self"


def confidence_score(row: dict[str, str | float]) -> float:
    if str(row.get("score", "")).strip() != "":
        return to_float(row.get("score"), 0.0)
    return to_float(row.get("p_self"), 0.0)


def apply_answer_probability_threshold(
    rows: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    if not rows:
        return rows

    scores = np.asarray([confidence_score(row) for row in rows], dtype=float)
    self_passed = np.asarray([to_float(row.get("self_passed"), 0.0) for row in rows], dtype=float)
    should_defer = (self_passed < 0.5).astype(float)  # 1 means model is wrong -> should defer

    unique_scores = np.unique(scores)
    candidates: list[float] = [-1e-6]
    candidates.extend(float(x) for x in unique_scores.tolist())
    candidates.append(1.000001)

    wrong_total = float(should_defer.sum())
    correct_total = float(len(rows) - wrong_total)
    target_wrong_rate = wrong_total / max(float(len(rows)), 1.0)

    best_threshold = candidates[0]
    best_balanced_acc = -1.0
    best_route_acc = -1.0
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
        elif abs(balanced_acc - best_balanced_acc) <= 1e-12 and route_acc > best_route_acc + 1e-12:
            better = True
        elif (
            abs(balanced_acc - best_balanced_acc) <= 1e-12
            and abs(route_acc - best_route_acc) <= 1e-12
            and rate_gap < best_rate_gap - 1e-12
        ):
            better = True
        if better:
            best_threshold = threshold
            best_balanced_acc = balanced_acc
            best_route_acc = route_acc
            best_rate_gap = rate_gap

    updated: list[dict[str, str | float]] = []
    for row in rows:
        new_row = dict(row)
        score = confidence_score(new_row)
        model_decision = "defer" if score < best_threshold else "self"
        new_row["model_decision"] = model_decision
        new_row["route"] = "post_defer" if model_decision == "defer" else "post_self"
        updated.append(new_row)
    return updated


def compute_system_success_rate(rows: list[dict[str, str | float]]) -> float:
    if not rows:
        return 0.0
    correct = 0.0
    for row in rows:
        if is_deferred_row(row):
            correct += to_float(row.get("expert_passed"), 0.0)
        else:
            correct += to_float(row.get("self_passed"), 0.0)
    return correct / len(rows)


def compute_budget_curve(rows: list[dict[str, str | float]]) -> dict[str, np.ndarray | float]:
    total = len(rows)
    actual_defer_count = sum(1 for row in rows if is_deferred_row(row))
    base_budgets = list(np.linspace(0, 1, 101))
    if total > 0:
        base_budgets.append(actual_defer_count / total)
    budgets = np.asarray(sorted(set(round(float(budget), 12) for budget in base_budgets)), dtype=float)
    if total == 0:
        return {
            "budgets": budgets,
            "routing_rates": np.zeros_like(budgets),
            "model_only_rate": 0.0,
            "expert_only_rate": 0.0,
            "actual_budget": 0.0,
            "actual_rate": 0.0,
        }

    prepared = [dict(row) for row in rows]
    prepared.sort(key=lambda row: (confidence_score(row), str(row.get("task_id", ""))))
    self_passed = np.array([to_float(row.get("self_passed"), 0.0) for row in prepared], dtype=float)
    expert_passed = np.array([to_float(row.get("expert_passed"), 0.0) for row in prepared], dtype=float)

    routing_rates = []
    for budget in budgets:
        n_defer = int(round(float(budget) * total))
        success = expert_passed[:n_defer].sum() + self_passed[n_defer:].sum()
        routing_rates.append(success / total * 100.0)

    return {
        "budgets": budgets,
        "routing_rates": np.asarray(routing_rates, dtype=float),
        "model_only_rate": float(self_passed.mean() * 100.0),
        "expert_only_rate": float(expert_passed.mean() * 100.0),
        "actual_budget": float(100.0 * actual_defer_count / total),
        "actual_rate": float(100.0 * compute_system_success_rate(rows)),
    }


def build_curve_from_threshold_csv(rows: list[dict[str, str]]) -> dict[str, np.ndarray | float]:
    parsed: list[tuple[float, float]] = []
    for row in rows:
        expert_rate = to_float(row.get("expert_rate"), np.nan)
        accuracy = to_float(row.get("accuracy"), np.nan)
        if np.isnan(expert_rate) or np.isnan(accuracy):
            continue
        parsed.append((expert_rate, accuracy))

    if not parsed:
        raise ValueError("threshold csv has no valid (expert_rate, accuracy) rows")

    parsed.sort(key=lambda item: item[0])
    budgets = np.asarray([item[0] * 100.0 for item in parsed], dtype=float)
    routing_rates = np.asarray([item[1] * 100.0 for item in parsed], dtype=float)

    idx_model = int(np.argmin(np.abs(np.asarray([item[0] for item in parsed], dtype=float) - 0.0)))
    idx_expert = int(np.argmin(np.abs(np.asarray([item[0] for item in parsed], dtype=float) - 1.0)))

    return {
        "budgets": budgets / 100.0,
        "routing_rates": routing_rates,
        "model_only_rate": float(routing_rates[idx_model]),
        "expert_only_rate": float(routing_rates[idx_expert]),
        "actual_budget": float("nan"),
        "actual_rate": float("nan"),
    }


def load_rows_for_curve(
    metadata_path: Path,
    *,
    expert_csv: str | Path | None = None,
    assume_expert_correct: bool = True,
) -> list[dict[str, str | float]]:
    rows = robust_read_csv(metadata_path)
    expert_map = load_expert_map(expert_csv) if expert_csv else {}
    merged_rows: list[dict[str, str | float]] = []

    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        merged = dict(row)
        merged["task_id"] = task_id
        merged["route"] = normalize_route(row.get("route"))
        merged["model_decision"] = str(row.get("model_decision", "")).strip().lower()
        merged["self_passed"] = to_float(row.get("self_passed"), 0.0)
        merged["p_defer"] = to_float(row.get("p_defer"), 0.0)
        merged["p_self"] = to_float(row.get("p_self"), 0.0)
        merged["score"] = to_float(row.get("score", row.get("p_self", 0.0)), 0.0)

        if expert_map:
            merged["expert_passed"] = expert_map.get(task_id, 0.0)
        else:
            raw_expert = str(row.get("expert_passed", "")).strip()
            if raw_expert:
                merged["expert_passed"] = to_float(raw_expert, 0.0)
            else:
                merged["expert_passed"] = 1.0 if assume_expert_correct else 0.0

        merged_rows.append(merged)
    return merged_rows


def ordered(values: set[str], priority: list[str]) -> list[str]:
    front = [item for item in priority if item in values]
    tail = sorted(item for item in values if item not in set(priority))
    return front + tail


def make_group_title(domain: str, model_family: str) -> str:
    domain_text = domain.upper() if domain in {"code", "math", "mmlu"} else domain
    model_text = model_family.upper() if model_family in {"qwen", "llama"} else model_family
    return f"{domain_text} | {model_text}"


def display_label(label: str) -> str:
    return DISPLAY_LABEL_MAP.get(label, label)


def color_for_label(label: str) -> str:
    if label in METHOD_COLOR_MAP:
        return METHOD_COLOR_MAP[label]
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def save_standalone_legend(
    legend_handles: dict[str, object],
    labels: list[str],
    out_path: Path,
    *,
    dpi: int,
) -> None:
    ordered_legend_labels = [label for label in labels if label in legend_handles]
    if not ordered_legend_labels:
        return
    legend_height = max(1.8, 0.34 * len(ordered_legend_labels) + 0.35)
    legend_fig = plt.figure(figsize=(3.0, legend_height))
    legend_fig.legend(
        [legend_handles[label] for label in ordered_legend_labels],
        [display_label(label) for label in ordered_legend_labels],
        loc="center left",
        ncol=1,
        frameon=False,
        fontsize=9.0,
        handlelength=2.0,
        handletextpad=0.55,
        labelspacing=0.45,
        borderaxespad=0.0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    legend_fig.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(legend_fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot paper-style route curves on one figure (grouped by domain x model)."
    )
    parser.add_argument(
        "--metadata",
        action="append",
        required=True,
        help="Repeatable. Use LABEL=/path/to/metadata.csv or /path/to/metadata.csv.",
    )
    parser.add_argument("--expert_csv", default=None, help="Optional global expert results csv.")
    parser.add_argument(
        "--assume_expert_correct",
        action="store_true",
        help="If expert_passed is missing and no expert csv is provided, treat deferred expert as always correct.",
    )
    parser.add_argument(
        "--out",
        default="figures/main_route_figure.png",
        help="Output figure path.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--show_natural_points",
        action="store_true",
        help="Overlay saved natural routing decisions as markers.",
    )
    parser.add_argument(
        "--hide_natural_points",
        action="store_true",
        help="Hide saved natural routing decision markers.",
    )
    parser.add_argument(
        "--force_model_family",
        default=None,
        choices=["qwen", "llama", "other"],
        help="Force all input runs to this model family (useful when some metadata lacks model_slug).",
    )
    parser.add_argument(
        "--force_domain",
        default=None,
        choices=["code", "math", "mmlu", "other"],
        help="Force all input runs to this domain (useful for cross-domain generalization plots).",
    )
    parser.add_argument(
        "--show_legend",
        action="store_true",
        help="Draw the legend on the main figure. This is the default unless --hide_legend is set.",
    )
    parser.add_argument("--hide_legend", action="store_true", help="Do not draw the legend on the main figure.")
    parser.add_argument("--legend_out", default=None, help="Optional standalone vertical legend output path.")
    parser.add_argument("--pre_threshold", type=float, default=0.5,
                        help="Stage-0 defer threshold for two-stage methods. Recomputes final score from first_p_defer/post_p_defer.")
    args = parser.parse_args()

    entries: list[dict[str, object]] = []
    for spec in args.metadata:
        label_override, path = parse_kv_spec(spec)
        if not path.exists():
            print(f"[skip] file not found: {path}")
            continue
        try:
            label = label_override or ""
            if not label:
                try:
                    label = infer_label(robust_read_csv(path), path, label_override)
                except Exception:
                    label = ""
            curve_path = resolve_curve_path(label, path)
            raw_rows = robust_read_csv(curve_path)
            if not raw_rows:
                print(f"[skip] empty csv: {curve_path}")
                continue
            row_keys = set(raw_rows[0].keys())
            is_threshold_curve = {"threshold", "expert_rate", "accuracy"}.issubset(row_keys)
            is_cascade_curve = {"pre_threshold", "post_threshold", "expert_rate", "accuracy"}.issubset(row_keys)

            metadata_rows = robust_read_csv(path) if curve_path != path else raw_rows
            domain = infer_domain(metadata_rows, path)
            model_family = infer_model_family(metadata_rows, path)
            if args.force_domain is not None:
                domain = args.force_domain
            if args.force_model_family is not None:
                model_family = args.force_model_family
            label = infer_label(metadata_rows, path, label_override)
            if is_threshold_curve or is_cascade_curve:
                curve = build_curve_from_threshold_csv(raw_rows)
            else:
                rows = load_rows_for_curve(
                    path,
                    expert_csv=args.expert_csv,
                    assume_expert_correct=args.assume_expert_correct or args.expert_csv is None,
                )
                if label == "Self-REF":
                    normalize_rows_to_score_decision(rows)
                if label == "Answer-Probability":
                    rows = apply_answer_probability_threshold(rows)
                # Recompute final score for two-stage methods based on --pre_threshold
                if args.pre_threshold != 0.5 and label not in ("Self-REF", "Answer-Probability"):
                    has_two_stage = any(
                        str(row.get("first_p_defer", "")).strip() and str(row.get("post_p_defer", "")).strip()
                        for row in rows
                    )
                    if has_two_stage:
                        for row in rows:
                            fp = str(row.get("first_p_defer", "")).strip()
                            pp = str(row.get("post_p_defer", "")).strip()
                            if fp and pp:
                                first_p = to_float(fp, 0.0)
                                post_p = to_float(pp, 0.0)
                                if first_p > args.pre_threshold:
                                    row["p_defer"] = first_p
                                    row["p_self"] = 1.0 - first_p
                                    row["score"] = 1.0 - first_p
                                else:
                                    row["p_defer"] = post_p
                                    row["p_self"] = 1.0 - post_p
                                    row["score"] = 1.0 - post_p
                curve = compute_budget_curve(rows)
            entries.append(
                {
                    "label": label,
                    "metadata_path": str(path),
                    "curve_path": str(curve_path),
                    "domain": domain,
                    "model_family": model_family,
                    "curve": curve,
                }
            )
        except Exception as exc:
            print(f"[skip] failed to parse {path}: {exc}")

    if not entries:
        raise SystemExit("No valid metadata csv provided.")

    domains = ordered({str(entry["domain"]) for entry in entries}, DOMAIN_ORDER)
    models = ordered({str(entry["model_family"]) for entry in entries}, MODEL_ORDER)
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for entry in entries:
        key = (str(entry["domain"]), str(entry["model_family"]))
        groups.setdefault(key, []).append(entry)
    for key in list(groups.keys()):
        groups[key].sort(key=lambda entry: str(entry["label"]).lower())

    labels = ordered({str(entry["label"]) for entry in entries}, METHOD_ORDER)
    label_to_color = {label: color_for_label(label) for label in labels}

    fig, axes = plt.subplots(
        len(domains),
        len(models),
        figsize=(7.2 * len(models), 4.4 * len(domains)),
        squeeze=False,
    )
    legend_handles: dict[str, object] = {}
    has_any_axis = False

    for row_idx, domain in enumerate(domains):
        for col_idx, model_family in enumerate(models):
            ax = axes[row_idx][col_idx]
            key = (domain, model_family)
            cell_entries = groups.get(key, [])
            if not cell_entries:
                ax.axis("off")
                continue

            has_any_axis = True
            y_values: list[float] = []
            for entry in cell_entries:
                label = str(entry["label"])
                curve = entry["curve"]
                budgets = np.asarray(curve["budgets"]) * 100.0
                routing_rates = np.asarray(curve["routing_rates"])
                line_zorder = 12 if label == "PBDD" else 4
                line, = ax.plot(
                    budgets,
                    routing_rates,
                    linewidth=1.4,
                    color=label_to_color[label],
                    label=label,
                    alpha=0.96,
                    zorder=line_zorder,
                )
                y_values.extend(float(x) for x in routing_rates.tolist())
                actual_rate = float(curve["actual_rate"])
                actual_budget = float(curve["actual_budget"])
                show_natural_points = args.show_natural_points or not args.hide_natural_points
                if show_natural_points and np.isfinite(actual_rate):
                    y_values.append(actual_rate)
                if label not in legend_handles:
                    legend_handles[label] = line

                if show_natural_points and np.isfinite(actual_budget) and np.isfinite(actual_rate):
                    ax.scatter(
                        [actual_budget],
                        [actual_rate],
                        color=label_to_color[label],
                        s=16,
                        marker="o",
                        alpha=0.9,
                        zorder=(line_zorder + 1),
                    )

            ref = cell_entries[0]["curve"]
            y_values.extend([float(ref["model_only_rate"]), float(ref["expert_only_rate"])])
            model_only_line = ax.axhline(
                float(ref["model_only_rate"]),
                color="#666666",
                linestyle="--",
                linewidth=0.9,
                alpha=0.85,
            )
            expert_only_line = ax.axhline(
                float(ref["expert_only_rate"]),
                color="#9a9a9a",
                linestyle=":",
                linewidth=0.9,
                alpha=0.85,
            )
            legend_handles.setdefault("Always-Self", model_only_line)
            legend_handles.setdefault("Always-Defer", expert_only_line)

            ax.set_xlim(0, 100)
            if y_values:
                lower = max(0.0, min(y_values) - 2.0)
                upper = min(100.0, max(y_values) + 2.0)
                if upper - lower < 4.0:
                    center = (upper + lower) / 2.0
                    lower = max(0.0, center - 2.0)
                    upper = min(100.0, center + 2.0)
                ax.set_ylim(lower, upper)
            ax.set_xticks(np.arange(0, 101, 20))
            ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.28)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)
            ax.tick_params(axis="both", labelsize=10)
            ax.set_xlabel("Expert budget (%)", fontsize=11)
            ax.set_ylabel("Accuracy (%)", fontsize=11)

    if not has_any_axis:
        raise SystemExit("All groups were empty after filtering.")

    ordered_legend_labels = [label for label in labels if label in legend_handles]
    for baseline_label in ["Always-Self", "Always-Defer"]:
        if baseline_label in legend_handles:
            ordered_legend_labels.append(baseline_label)
    ordered_legend_text = [display_label(label) for label in ordered_legend_labels]
    if args.legend_out:
        save_standalone_legend(legend_handles, ordered_legend_labels, Path(args.legend_out), dpi=int(args.dpi))
        print(f"Saved legend to: {args.legend_out}")

    if not args.hide_legend:
        fig.legend(
            [legend_handles[label] for label in ordered_legend_labels],
            ordered_legend_text,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(4, max(1, len(ordered_legend_labels))),
            frameon=True,
            framealpha=0.92,
            fancybox=False,
            fontsize=8.0,
            handlelength=1.5,
            handletextpad=0.35,
            columnspacing=0.55,
            borderaxespad=0.25,
            borderpad=0.25,
            labelspacing=0.22,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    else:
        fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {out_path}")
    print(f"Loaded runs: {len(entries)}")
    print(f"Domains: {', '.join(domains)}")
    print(f"Models: {', '.join(models)}")


if __name__ == "__main__":
    main()
