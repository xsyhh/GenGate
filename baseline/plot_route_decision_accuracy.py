#!/usr/bin/env python3
# Example:
# python plot_route_decision_accuracy.py \
#   --metadata PBDD=/path/to/pbdd_metadata.csv \
#   --metadata Self-REF=/path/to/self_ref_metadata.csv \
#   --out figures/route_decision_accuracy/route_acc_code_qwen.png

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
METHOD_COLOR_MAP = {
    "PBDD": "#1f77b4",
    "Self-REF": "#d62728",
    "MC-Two-Stage-Probe": "#ff9896",
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
    "External-Prompt-Router": "Prompt Router",
    "Answer-Probability": "Answer Prob",
    "Roberta-Router": "RoBERTa Router",
    "Post-Linear": "Post Linear",
    "Pre-Linear": "Pre Linear",
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


def ordered(values: set[str], priority: list[str]) -> list[str]:
    front = [item for item in priority if item in values]
    tail = sorted(item for item in values if item not in set(priority))
    return front + tail


def display_label(label: str) -> str:
    return DISPLAY_LABEL_MAP.get(label, label)


def color_for_label(label: str) -> str:
    if label in METHOD_COLOR_MAP:
        return METHOD_COLOR_MAP[label]
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def normalize_route(route: str | None) -> str:
    value = str(route or "").strip().lower()
    return value or "other"


def is_self_ref_run(label: str | None, metadata_path: str | Path) -> bool:
    text = f"{label or ''} {metadata_path}".lower().replace("_", "-")
    return "self-ref" in text


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


def confidence_score(row: dict[str, str | float]) -> float:
    score_raw = str(row.get("score", "")).strip()
    if score_raw:
        return to_float(score_raw, 0.5)
    p_self_raw = str(row.get("p_self", "")).strip()
    if p_self_raw:
        return to_float(p_self_raw, 0.5)
    p_defer_raw = str(row.get("p_defer", "")).strip()
    if p_defer_raw:
        return 1.0 - to_float(p_defer_raw, 0.5)
    return 0.5


def apply_answer_probability_threshold(
    rows: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    if not rows:
        return rows

    scores = np.asarray([confidence_score(row) for row in rows], dtype=float)
    self_passed = np.asarray([to_float(row.get("self_passed"), 0.0) for row in rows], dtype=float)
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


def load_rows(metadata_path: Path) -> list[dict[str, str | float]]:
    rows = robust_read_csv(metadata_path)
    out: list[dict[str, str | float]] = []
    for row in rows:
        self_raw = str(row.get("self_passed", "")).strip()
        if not self_raw:
            continue
        out.append(
            {
                "task_id": str(row.get("task_id", "")).strip(),
                "route": normalize_route(row.get("route")),
                "model_decision": str(row.get("model_decision", "")).strip().lower(),
                "self_passed": to_float(self_raw, 0.0),
                "p_defer": to_float(row.get("p_defer"), 0.0),
                "p_self": to_float(row.get("p_self"), 0.0),
                "score": to_float(row.get("score", row.get("p_self", 0.5)), 0.5),
            }
        )
    return out


def compute_route_accuracy_curve(rows: list[dict[str, str | float]]) -> dict[str, np.ndarray | float]:
    total = len(rows)
    actual_defer_count = sum(1 for row in rows if is_deferred_row(row))
    base_budgets = list(np.linspace(0, 1, 101))
    if total > 0:
        base_budgets.append(actual_defer_count / total)
    budgets = np.asarray(sorted(set(round(float(budget), 12) for budget in base_budgets)), dtype=float)
    if total == 0:
        return {
            "budgets": budgets,
            "route_acc": np.zeros_like(budgets),
            "actual_budget": 0.0,
            "actual_route_acc": 0.0,
        }

    prepared = [dict(row) for row in rows]
    prepared.sort(key=lambda row: (confidence_score(row), str(row.get("task_id", ""))))
    self_passed = np.array([to_float(row.get("self_passed"), 0.0) for row in prepared], dtype=float)
    should_defer = (self_passed < 0.5).astype(float)

    route_acc = []
    for budget in budgets:
        n_defer = int(round(float(budget) * total))
        correct = should_defer[:n_defer].sum() + (1.0 - should_defer[n_defer:]).sum()
        route_acc.append(correct / total * 100.0)

    actual_defer = np.array([1.0 if is_deferred_row(row) else 0.0 for row in rows], dtype=float)
    actual_correct = ((actual_defer == 1.0) & (np.array([to_float(row.get("self_passed"), 0.0) for row in rows]) < 0.5)) | (
        (actual_defer == 0.0) & (np.array([to_float(row.get("self_passed"), 0.0) for row in rows]) >= 0.5)
    )

    return {
        "budgets": budgets,
        "route_acc": np.asarray(route_acc, dtype=float),
        "actual_budget": float(actual_defer_count / total * 100.0),
        "actual_route_acc": float(actual_correct.mean() * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot routing decision accuracy vs expert budget.")
    parser.add_argument(
        "--metadata",
        action="append",
        required=True,
        help="Repeatable. Use LABEL=/path/to/metadata.csv or /path/to/metadata.csv.",
    )
    parser.add_argument("--out", required=True, help="Output figure path.")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--title", default="", help="Optional title. Empty means no title.")
    parser.add_argument("--show_legend", action="store_true", help="Draw a legend.")
    parser.add_argument("--hide_legend", action="store_true", help="Do not draw a legend.")
    args = parser.parse_args()

    entries: list[dict[str, object]] = []
    for spec in args.metadata:
        label_override, path = parse_kv_spec(spec)
        if not path.exists():
            print(f"[skip] file not found: {path}")
            continue
        try:
            rows = load_rows(path)
            if not rows:
                print(f"[skip] no usable rows (missing self_passed): {path}")
                continue
            label = label_override if label_override else (path.parent.name or path.stem)
            if is_self_ref_run(label, path):
                normalize_rows_to_score_decision(rows)
            if label == "Answer-Probability":
                rows = apply_answer_probability_threshold(rows)
            curve = compute_route_accuracy_curve(rows)
            entries.append({"label": label, "curve": curve})
        except Exception as exc:
            print(f"[skip] failed to parse {path}: {exc}")

    if not entries:
        raise SystemExit("No valid metadata rows found.")

    labels = ordered({str(entry["label"]) for entry in entries}, METHOD_ORDER)
    label_to_color = {label: color_for_label(label) for label in labels}
    entries.sort(key=lambda item: labels.index(str(item["label"])) if str(item["label"]) in labels else 10**9)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    y_values: list[float] = []
    handles = []
    legend_text = []
    for entry in entries:
        label = str(entry["label"])
        curve = entry["curve"]
        budgets = np.asarray(curve["budgets"]) * 100.0
        route_acc = np.asarray(curve["route_acc"])
        line_zorder = 12 if label == "PBDD" else 4
        line, = ax.plot(
            budgets,
            route_acc,
            linewidth=1.5,
            color=label_to_color[label],
            alpha=0.96,
            zorder=line_zorder,
        )
        handles.append(line)
        legend_text.append(display_label(label))
        y_values.extend(float(x) for x in route_acc.tolist())

        actual_budget = float(curve["actual_budget"])
        actual_route_acc = float(curve["actual_route_acc"])
        if np.isfinite(actual_budget) and np.isfinite(actual_route_acc):
            ax.scatter(
                [actual_budget],
                [actual_route_acc],
                s=14,
                color=label_to_color[label],
                alpha=0.9,
                zorder=(line_zorder + 1),
            )
            y_values.append(actual_route_acc)

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    if y_values:
        lower = max(0.0, min(y_values) - 2.0)
        upper = min(100.0, max(y_values) + 2.0)
        if upper - lower < 6.0:
            center = (upper + lower) / 2.0
            lower = max(0.0, center - 3.0)
            upper = min(100.0, center + 3.0)
        ax.set_ylim(lower, upper)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("Expert Budget (%)", fontsize=11)
    ax.set_ylabel("Routing Accuracy (%)", fontsize=11)
    if args.title.strip():
        ax.set_title(args.title.strip(), fontsize=11)

    if args.show_legend and not args.hide_legend:
        fig.legend(
            handles,
            legend_text,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(4, max(1, len(legend_text))),
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


if __name__ == "__main__":
    main()
