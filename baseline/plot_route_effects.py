
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
  --metadata /path/to/code_metadata.csv \
  --expert_csv /path/to/code_expert_results.csv

python plot_route_effects.py \
  --metadata /path/to/math_metadata.csv \
  --expert_csv /path/to/math_expert_results.csv

python plot_route_effects.py \
  --metadata /path/to/mmlu_metadata.csv \
  --expert_csv /path/to/mmlu_expert_results.csv

python plot_paper_main_figure.py \
  --force_domain mmlu \
  --force_model_family llama \
  --metadata 'Math → MMLU=/path/to/math2mmlu_metadata.csv' \
  --metadata 'MMLU → MMLU=/path/to/mmlu_metadata.csv' \
  --expert_csv /path/to/mmlu_expert_results.csv \
  --hide_legend \
  --legend_out /path/to/llama_math2mmlu_legend.png \
  --out /path/to/llama_math2mmlu.png

python plot_paper_main_figure.py \
  --force_domain mmlu \
  --force_model_family qwen \
  --metadata 'Math → MMLU=/path/to/math2mmlu_metadata.csv' \
  --metadata 'MMLU → MMLU=/path/to/mmlu_metadata.csv' \
  --expert_csv /path/to/mmlu_expert_results.csv \
  --hide_legend \
  --legend_out /path/to/qwen_math2mmlu_legend.png \
  --out /path/to/qwen_math2mmlu.png
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


def ensure_axes_frame() -> None:
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)


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


def normalize_route(route: str | None) -> str:
    value = str(route or "").strip().lower()
    return value or "other"


def is_self_ref_run(label: str | None, metadata_csv: str | Path) -> bool:
    text = f"{label or ''} {metadata_csv}".lower().replace("_", "-")
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


def load_expert_map(expert_csv: str | Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in robust_read_csv(expert_csv):
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        out[task_id] = to_float(row.get("expert_passed", row.get("self_passed", 0.0)), 0.0)
    return out


def load_metadata_rows(
    metadata_csv: str | Path,
    *,
    expert_csv: str | Path | None = None,
    assume_expert_correct: bool = False,
) -> list[dict[str, str | float]]:
    rows = robust_read_csv(metadata_csv)
    expert_map = load_expert_map(expert_csv) if expert_csv is not None else {}
    if expert_csv is not None:
        metadata_ids = {str(row.get("task_id", "")).strip() for row in rows if str(row.get("task_id", "")).strip()}
        if metadata_ids and not metadata_ids.intersection(expert_map):
            raise ValueError(f"Expert csv has no overlapping task_id values with metadata: {expert_csv}")

    out: list[dict[str, str | float]] = []
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        merged = dict(row)
        merged["task_id"] = task_id
        merged["route"] = normalize_route(row.get("route"))
        merged["model_decision"] = str(row.get("model_decision", "")).strip().lower()
        merged["self_passed"] = to_float(row.get("self_passed"), 0.0)
        merged["p_defer"] = to_float(row.get("p_defer"), 0.0)
        merged["p_self"] = to_float(row.get("p_self"), 0.0)
        merged["score"] = to_float(row.get("score", row.get("p_self")), 0.0)
        if expert_csv is not None:
            merged["expert_passed"] = expert_map.get(task_id, 0.0)
        elif "expert_passed" in row and str(row.get("expert_passed", "")).strip() != "":
            merged["expert_passed"] = to_float(row.get("expert_passed"), 0.0)
        elif assume_expert_correct:
            merged["expert_passed"] = 1.0
        else:
            merged["expert_passed"] = 0.0
        out.append(merged)
    return out


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
    if str(row.get("score", "")).strip() != "":
        return to_float(row.get("score"), 0.0)
    return to_float(row.get("p_self"), 0.0)


def compute_system_success_rate(rows: list[dict[str, str | float]]) -> float:
    if not rows:
        return 0.0
    correct = 0.0
    for row in rows:
        correct += to_float(row.get("expert_passed"), 0.0) if is_deferred_row(row) else to_float(row.get("self_passed"), 0.0)
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


def compute_route_summary(rows: list[dict[str, str | float]], *, label: str, metadata_csv: str | Path) -> dict[str, float | int | str]:
    total = len(rows)
    summary: dict[str, float | int | str] = {
        "label": label,
        "metadata_csv": str(metadata_csv),
        "total_tasks": total,
        "model_only_accuracy": 0.0,
        "expert_only_accuracy": 0.0,
        "actual_expert_rate": 0.0,
        "actual_accuracy": 0.0,
    }
    if total == 0:
        return summary

    route_counts = {route: 0 for route in ROUTE_ORDER}
    route_pass_sums = {route: 0.0 for route in ROUTE_ORDER}
    for row in rows:
        route = normalize_route(str(row.get("route", "")))
        if route in route_counts:
            route_counts[route] += 1
            route_pass_sums[route] += to_float(row.get("self_passed"), 0.0)

    for route in ROUTE_ORDER:
        count = route_counts[route]
        summary[f"route_count_{route}"] = count
        summary[f"route_rate_{route}"] = count / total
        summary[f"cf_pass_rate_{route}"] = route_pass_sums[route] / count if count else 0.0

    summary["model_only_accuracy"] = sum(to_float(row.get("self_passed"), 0.0) for row in rows) / total
    summary["expert_only_accuracy"] = sum(to_float(row.get("expert_passed"), 0.0) for row in rows) / total
    summary["actual_expert_rate"] = sum(1 for row in rows if is_deferred_row(row)) / total
    summary["actual_accuracy"] = compute_system_success_rate(rows)
    return summary


def write_summary(summary: dict[str, float | int | str], out_dir: Path) -> None:
    csv_path = out_dir / "route_effect_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    lines = []
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    (out_dir / "route_effect_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_route_distribution(summary: dict[str, float | int | str], out_path: Path) -> None:
    labels = [ROUTE_DISPLAY[route] for route in ROUTE_ORDER]
    values = [float(summary.get(f"route_rate_{route}", 0.0)) * 100.0 for route in ROUTE_ORDER]
    colors = [ROUTE_COLORS[route] for route in ROUTE_ORDER]
    plt.figure(figsize=(6.5, 4.0))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Examples (%)")
    plt.title("Route Distribution")
    plt.ylim(0, max(100.0, max(values, default=0.0) + 5.0))
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ensure_axes_frame()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_budget_curve(rows: list[dict[str, str | float]], out_path: Path, *, show_legend: bool = False) -> None:
    curve = compute_budget_curve(rows)
    plt.figure(figsize=(6.5, 4.3))
    plt.plot(curve["budgets"] * 100.0, curve["routing_rates"], linewidth=2.0, label="Threshold sweep")
    plt.axhline(curve["model_only_rate"], color="#555555", linestyle="--", linewidth=1.2, label="Always local")
    plt.axhline(curve["expert_only_rate"], color="#999999", linestyle=":", linewidth=1.2, label="Always expert")
    plt.scatter([curve["actual_budget"]], [curve["actual_rate"]], color="#d62728", zorder=3, label="Saved threshold")
    plt.xlabel("Expert budget (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Expert Budget")
    plt.xlim(0, 100)
    values = list(float(x) for x in curve["routing_rates"])
    values.extend([float(curve["model_only_rate"]), float(curve["expert_only_rate"]), float(curve["actual_rate"])])
    lower = max(0.0, min(values) - 2.0)
    upper = min(100.0, max(values) + 2.0)
    if upper - lower < 5.0:
        center = (upper + lower) / 2.0
        lower = max(0.0, center - 2.5)
        upper = min(100.0, center + 2.5)
    plt.ylim(lower, upper)
    plt.grid(True, axis="y", alpha=0.25)
    ensure_axes_frame()
    if show_legend:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline route curves from a metadata.csv file.")
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv.")
    parser.add_argument("--expert_csv", default=None, help="Optional expert results CSV. Overrides expert_passed in metadata.")
    parser.add_argument("--assume_expert_correct", action="store_true", help="Use expert_passed=1 when no expert CSV/column is available.")
    parser.add_argument("--label", default=None, help="Optional label for summaries. Defaults to metadata parent folder name.")
    parser.add_argument("--out_dir", default=None, help="Output directory. Defaults to metadata parent / route_curve.")
    parser.add_argument("--show_legend", action="store_true", help="Draw legends. Default is no legend.")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    out_dir = Path(args.out_dir) if args.out_dir else metadata_path.parent / "route_curve"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or metadata_path.parent.name
    rows = load_metadata_rows(
        metadata_path,
        expert_csv=args.expert_csv,
        assume_expert_correct=args.assume_expert_correct,
    )
    if is_self_ref_run(label, metadata_path):
        normalize_rows_to_score_decision(rows)
    summary = compute_route_summary(rows, label=label, metadata_csv=metadata_path)
    write_summary(summary, out_dir)
    plot_route_distribution(summary, out_dir / "route_distribution.png")
    plot_budget_curve(rows, out_dir / "budget_curve.png", show_legend=args.show_legend)
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
