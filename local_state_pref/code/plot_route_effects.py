from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


"""
python plot_route_effects.py \
  --metadata BCE=/path/to/bce_metadata.csv \
  --metadata DPO=/path/to/dpo_metadata.csv \
  --metadata SFT=/path/to/sft_metadata.csv \
  --expert_csv /path/to/expert_results.csv \
  --oracle_source SFT \
  --out_dir /path/to/route_curve_with_sft
"""

ROUTE_ORDER = ["early_defer", "post_self", "post_defer"]
ROUTE_DISPLAY = {
    "early_defer": "Early Defer",
    "post_self": "Post Self",
    "post_defer": "Post Defer",
}
ROUTE_COLORS = {
    "early_defer": "#d95f02",
    "post_self": "#1b9e77",
    "post_defer": "#7570b3",
}


def parse_metadata_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid metadata spec: {spec!r}")
        return label, path

    path = spec.strip()
    if not path:
        raise ValueError("Empty metadata spec")
    label = Path(path).parent.name or Path(path).stem
    return label, path


def robust_read_csv(csv_path: str | Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [{str(k).strip(): v for k, v in row.items()} for row in reader]


def normalize_route(route: str | None) -> str:
    s = str(route or "").strip().lower()
    if s in ROUTE_ORDER:
        return s
    return s or "other"


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_expert_map(expert_csv: str | Path) -> dict[str, float]:
    expert_rows = robust_read_csv(expert_csv)
    expert_map: dict[str, float] = {}
    for row in expert_rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        if "expert_passed" in row:
            expert_map[task_id] = to_float(row.get("expert_passed"), 0.0)
        else:
            expert_map[task_id] = to_float(row.get("self_passed"), 0.0)
    return expert_map


def load_metadata_rows(
    metadata_csv: str | Path,
    expert_csv: str | Path | None = None,
    *,
    assume_expert_correct: bool = False,
) -> list[dict[str, str | float]]:
    rows = robust_read_csv(metadata_csv)
    expert_map = load_expert_map(expert_csv) if expert_csv is not None else {}
    if expert_csv is not None:
        metadata_ids = {str(row.get("task_id", "")).strip() for row in rows if str(row.get("task_id", "")).strip()}
        overlap = metadata_ids.intersection(expert_map)
        if metadata_ids and not overlap:
            raise ValueError(
                f"Expert csv has no overlapping task_id values with metadata: {expert_csv}. "
                "Use the matching split/domain expert csv, or omit --expert_csv and pass --assume_expert_correct."
            )

    merged_rows: list[dict[str, str | float]] = []
    for row in rows:
        merged = dict(row)
        merged["task_id"] = str(row.get("task_id", "")).strip()
        merged["route"] = normalize_route(row.get("route"))
        merged["self_passed"] = to_float(row.get("self_passed"), 0.0)
        merged["p_defer"] = to_float(row.get("p_defer"), 0.0)
        merged["margin"] = to_float(row.get("margin"), 0.0)
        if expert_csv is not None:
            merged["expert_passed"] = expert_map.get(str(merged["task_id"]), 0.0)
        elif assume_expert_correct:
            merged["expert_passed"] = 1.0
        merged_rows.append(merged)
    return merged_rows


def is_deferred_row(row: dict[str, str | float]) -> bool:
    route = normalize_route(str(row.get("route", "")))
    if route in {"early_defer", "post_defer"}:
        return True
    if route == "post_self":
        return False

    model_decision = str(row.get("model_decision", "")).strip().lower()
    if model_decision:
        return model_decision == "defer"

    if "p_defer" in row:
        return to_float(row.get("p_defer"), 0.0) > 0.5
    return to_float(row.get("margin"), 0.0) > 0.0


def compute_system_success_rate(rows: list[dict[str, str | float]]) -> float:
    total = len(rows)
    if total == 0:
        return 0.0
    success = 0.0
    for row in rows:
        if is_deferred_row(row):
            success += to_float(row.get("expert_passed"), 0.0)
        else:
            success += to_float(row.get("self_passed"), 0.0)
    return success / total


def compute_budget_curve(
    rows: list[dict[str, str | float]],
    oracle: str | None = None,
    *,
    assume_expert_correct: bool = False,
) -> dict[str, np.ndarray | float]:
    total = len(rows)
    budgets = np.linspace(0, 1, 101)
    if total == 0:
        return {
            "budgets": budgets,
            "routing_rates": np.zeros_like(budgets),
            "model_only_rate": 0.0,
            "actual_budget": 0.0,
            "actual_rate": 0.0,
        }

    prepared = [dict(row) for row in rows]
    for row in prepared:
        row["self_passed"] = to_float(row.get("self_passed"), 0.0)
        row["expert_passed"] = 1.0 if assume_expert_correct and "expert_passed" not in row else to_float(row.get("expert_passed"), 0.0)
        row["p_defer"] = to_float(row.get("p_defer"), 0.0)
        row["task_id"] = str(row.get("task_id", ""))

    if oracle == "naive":
        prepared.sort(key=lambda row: (row["self_passed"], row["task_id"]))
    elif oracle == "knowledge":
        prepared.sort(
            key=lambda row: (
                -int((row["self_passed"] == 0.0) and (row["expert_passed"] > 0.0)),
                row["task_id"],
            )
        )
    else:
        prepared.sort(key=lambda row: (-row["p_defer"], row["task_id"]))

    self_passed = np.array([float(row["self_passed"]) for row in prepared], dtype=float)
    expert_passed = np.array([float(row["expert_passed"]) for row in prepared], dtype=float)
    model_only_rate = self_passed.mean() * 100.0

    routing_rates = []
    for budget in budgets:
        n_defer = int(round(budget * total))
        success = expert_passed[:n_defer].sum() + self_passed[n_defer:].sum()
        routing_rates.append(success / total * 100.0)
    routing_rates = np.array(routing_rates, dtype=float)

    actual_budget = 100.0 * sum(1 for row in prepared if is_deferred_row(row)) / total
    actual_rate = 100.0 * compute_system_success_rate(prepared)
    return {
        "budgets": budgets,
        "routing_rates": routing_rates,
        "model_only_rate": model_only_rate,
        "actual_budget": actual_budget,
        "actual_rate": actual_rate,
    }


def select_oracle_comparison(comparisons: list[dict[str, object]], oracle_source: str | None) -> dict[str, object]:
    if not comparisons:
        raise ValueError("No comparisons available for oracle selection")
    if not oracle_source:
        return comparisons[0]

    for comparison in comparisons:
        if str(comparison.get("label", "")).strip() == oracle_source:
            return comparison
    raise ValueError(f"Unknown oracle source label: {oracle_source}")


def compute_plot_ylim(curves: list[dict[str, object]], padding: float = 2.0) -> tuple[float, float]:
    values: list[float] = []
    for curve in curves:
        values.extend(float(x) for x in curve.get("routing_rates", []))
        if "model_only_rate" in curve:
            values.append(float(curve["model_only_rate"]))
        if "actual_rate" in curve:
            values.append(float(curve["actual_rate"]))

    if not values:
        return 0.0, 100.0

    lower = max(0.0, min(values) - padding)
    upper = min(100.0, max(values) + padding)
    if upper - lower < 5.0:
        center = (upper + lower) / 2.0
        lower = max(0.0, center - 2.5)
        upper = min(100.0, center + 2.5)
    return lower, upper


def compute_route_summary(
    metadata_csv: str | Path,
    label: str,
    expert_csv: str | Path | None = None,
    *,
    assume_expert_correct: bool = False,
) -> dict[str, float | int | str]:
    rows = load_metadata_rows(metadata_csv, expert_csv=expert_csv, assume_expert_correct=assume_expert_correct)
    total = len(rows)
    summary: dict[str, float | int | str] = {
        "label": label,
        "metadata_csv": str(metadata_csv),
        "total_tasks": total,
    }

    route_counts = {route: 0 for route in ROUTE_ORDER}
    route_pass_sums = {route: 0.0 for route in ROUTE_ORDER}

    for row in rows:
        route = normalize_route(str(row.get("route", "")))
        passed = to_float(row.get("self_passed"), 0.0)
        if route in route_counts:
            route_counts[route] += 1
            route_pass_sums[route] += passed

    overall_cf_pass = sum(to_float(row.get("self_passed"), 0.0) for row in rows)
    deferred_count = route_counts["early_defer"] + route_counts["post_defer"]
    deferred_pass_sum = route_pass_sums["early_defer"] + route_pass_sums["post_defer"]

    for route in ROUTE_ORDER:
        count = route_counts[route]
        summary[f"route_count_{route}"] = count
        summary[f"route_rate_{route}"] = (count / total) if total else 0.0
        summary[f"cf_pass_rate_{route}"] = (route_pass_sums[route] / count) if count else 0.0

    post_self_count = route_counts["post_self"]
    summary["accepted_self_accuracy"] = (route_pass_sums["post_self"] / post_self_count) if post_self_count else 0.0
    summary["deferred_cf_pass_rate"] = (deferred_pass_sum / deferred_count) if deferred_count else 0.0
    summary["overall_cf_pass_rate"] = (overall_cf_pass / total) if total else 0.0
    summary["actual_budget"] = (deferred_count / total) if total else 0.0

    if expert_csv is not None or assume_expert_correct:
        summary["system_success_rate"] = compute_system_success_rate(rows)

    return summary


def _summary_fieldnames(summaries: list[dict[str, float | int | str]]) -> list[str]:
    fieldnames = ["label", "metadata_csv", "total_tasks"]
    for route in ROUTE_ORDER:
        fieldnames.extend(
            [
                f"route_count_{route}",
                f"route_rate_{route}",
                f"cf_pass_rate_{route}",
            ]
        )
    fieldnames.extend(["accepted_self_accuracy", "deferred_cf_pass_rate", "overall_cf_pass_rate"])
    fieldnames.append("actual_budget")
    if any("system_success_rate" in summary for summary in summaries):
        fieldnames.append("system_success_rate")
    return fieldnames


def write_summary_csv(summaries: list[dict[str, float | int | str]], out_path: Path) -> None:
    fieldnames = _summary_fieldnames(summaries)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_summary_report(summaries: list[dict[str, float | int | str]], out_path: Path) -> None:
    lines = []
    for summary in summaries:
        lines.append("=" * 72)
        lines.append(str(summary["label"]))
        lines.append("=" * 72)
        total = int(summary["total_tasks"])
        lines.append(f"metadata: {summary['metadata_csv']}")
        lines.append(f"total tasks: {total}")
        for route in ROUTE_ORDER:
            count = int(summary[f"route_count_{route}"])
            rate = float(summary[f"route_rate_{route}"]) * 100
            cf_pass = float(summary[f"cf_pass_rate_{route}"]) * 100
            lines.append(f"{route:>12}: count={count:>4}  rate={rate:>6.2f}%  cf_self_pass={cf_pass:>6.2f}%")
        lines.append(f"accepted_self_accuracy: {float(summary['accepted_self_accuracy']) * 100:.2f}%")
        lines.append(f"deferred_cf_pass_rate: {float(summary['deferred_cf_pass_rate']) * 100:.2f}%")
        lines.append(f"overall_cf_pass_rate: {float(summary['overall_cf_pass_rate']) * 100:.2f}%")
        lines.append(f"actual_budget: {float(summary['actual_budget']) * 100:.2f}%")
        if "system_success_rate" in summary:
            lines.append(f"system_success_rate: {float(summary['system_success_rate']) * 100:.2f}%")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_route_distribution(summaries: list[dict[str, float | int | str]], out_path: Path) -> None:
    labels = [str(summary["label"]) for summary in summaries]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for route in ROUTE_ORDER:
        values = np.array([float(summary[f"route_rate_{route}"]) * 100 for summary in summaries], dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=ROUTE_DISPLAY[route],
            color=ROUTE_COLORS[route],
            alpha=0.9,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Route Share (%)")
    ax.set_title("Route Distribution")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_route_outcomes(summaries: list[dict[str, float | int | str]], out_path: Path) -> None:
    labels = [str(summary["label"]) for summary in summaries]
    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for idx, route in enumerate(ROUTE_ORDER):
        values = np.array([float(summary[f"cf_pass_rate_{route}"]) * 100 for summary in summaries], dtype=float)
        ax.bar(
            x + (idx - 1) * width,
            values,
            width=width,
            label=ROUTE_DISPLAY[route],
            color=ROUTE_COLORS[route],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Counterfactual Self Pass Rate (%)")
    ax.set_title("Route Outcome Breakdown")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_overall_metrics(summaries: list[dict[str, float | int | str]], out_path: Path) -> None:
    labels = [str(summary["label"]) for summary in summaries]
    x = np.arange(len(labels))
    width = 0.22

    metric_specs = [
        ("overall_cf_pass_rate", "Overall CF Pass", "#4c78a8"),
        ("accepted_self_accuracy", "Accepted Self Acc", "#59a14f"),
        ("deferred_cf_pass_rate", "Deferred CF Pass", "#e15759"),
    ]
    if any("system_success_rate" in summary for summary in summaries):
        metric_specs.append(("system_success_rate", "System Success", "#f28e2b"))

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    center = (len(metric_specs) - 1) / 2
    for idx, (key, title, color) in enumerate(metric_specs):
        values = np.array([float(summary.get(key, 0.0)) * 100 for summary in summaries], dtype=float)
        ax.bar(x + (idx - center) * width, values, width=width, label=title, color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Overall Routing Metrics")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_budget_curves(
    comparisons: list[dict[str, object]],
    out_path: Path,
    oracle_source: str | None = None,
    *,
    assume_expert_correct: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    base_colors = plt.get_cmap("tab10")
    plotted_curves: list[dict[str, object]] = []
    for idx, comparison in enumerate(comparisons):
        label = str(comparison["label"])
        rows = comparison["rows"]
        curve = compute_budget_curve(rows, assume_expert_correct=assume_expert_correct)
        plotted_curves.append(curve)
        color = base_colors(idx % 10)

        ax.plot(
            curve["budgets"] * 100.0,
            curve["routing_rates"],
            color=color,
            linewidth=2,
            label=f"{label} (routing)",
        )
        ax.axhline(
            y=float(curve["model_only_rate"]),
            color=color,
            linestyle=":",
            linewidth=1,
            label=f"{label} model-only ({float(curve['model_only_rate']):.1f}%)",
        )
        ax.plot(
            float(curve["actual_budget"]),
            float(curve["actual_rate"]),
            "*",
            color=color,
            markersize=12,
            zorder=5,
            label=f"{label} decision ({float(curve['actual_budget']):.0f}%, {float(curve['actual_rate']):.1f}%)",
        )

    oracle_comparison = select_oracle_comparison(comparisons, oracle_source)
    oracle_label = str(oracle_comparison["label"])
    rows = oracle_comparison["rows"]
    for oracle_mode, color, title in [
        ("naive", "#59a14f", "Oracle-Naive"),
        ("knowledge", "#e15759", "Oracle-Knowledge"),
    ]:
        curve = compute_budget_curve(rows, oracle=oracle_mode, assume_expert_correct=assume_expert_correct)
        plotted_curves.append(curve)
        ax.plot(
            curve["budgets"] * 100.0,
            curve["routing_rates"],
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"{title} ({oracle_label})",
        )

    ax.set_xlabel("Defer Budget (%)")
    ax.set_ylabel("Total Success Rate (%)")
    ax.set_title("Route Curve with Expert")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(*compute_plot_ylim(plotted_curves))
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot route-level comparison figures from local-state-pref metadata.csv files.")
    parser.add_argument(
        "--metadata",
        action="append",
        required=True,
        help="Metadata spec in LABEL=/path/to/metadata.csv format. Can be passed multiple times.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--expert_csv", default=None, help="Optional expert results csv with task_id + expert_passed/self_passed.")
    parser.add_argument(
        "--assume_expert_correct",
        action="store_true",
        help="If no expert csv is available, treat every expert-routed example as correct.",
    )
    parser.add_argument("--oracle_source", default=None, help="Optional label used to draw the two oracle lines. Defaults to the first metadata entry.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    comparisons: list[dict[str, object]] = []
    for spec in args.metadata:
        label, metadata_csv = parse_metadata_spec(spec)
        rows = load_metadata_rows(
            metadata_csv,
            expert_csv=args.expert_csv,
            assume_expert_correct=args.assume_expert_correct,
        )
        comparisons.append({"label": label, "metadata_csv": metadata_csv, "rows": rows})
        summaries.append(
            compute_route_summary(
                metadata_csv,
                label=label,
                expert_csv=args.expert_csv,
                assume_expert_correct=args.assume_expert_correct,
            )
        )

    write_summary_csv(summaries, out_dir / "route_effect_summary.csv")
    write_summary_report(summaries, out_dir / "route_effect_summary.txt")
    plot_route_distribution(summaries, out_dir / "route_distribution.png")
    plot_route_outcomes(summaries, out_dir / "route_outcomes.png")
    plot_overall_metrics(summaries, out_dir / "route_overall_metrics.png")
    has_expert_scores = bool(args.expert_csv or args.assume_expert_correct)
    if has_expert_scores:
        plot_budget_curves(
            comparisons,
            out_dir / "budget_curve.png",
            oracle_source=args.oracle_source,
            assume_expert_correct=args.assume_expert_correct,
        )

    print(f"Saved summary csv to: {out_dir / 'route_effect_summary.csv'}")
    print(f"Saved summary report to: {out_dir / 'route_effect_summary.txt'}")
    print(f"Saved route distribution to: {out_dir / 'route_distribution.png'}")
    print(f"Saved route outcomes to: {out_dir / 'route_outcomes.png'}")
    print(f"Saved overall metrics to: {out_dir / 'route_overall_metrics.png'}")
    if has_expert_scores:
        print(f"Saved budget curve to: {out_dir / 'budget_curve.png'}")


if __name__ == "__main__":
    main()
