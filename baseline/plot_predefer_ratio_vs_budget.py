#!/usr/bin/env python3
# Example:
# python plot_predefer_ratio_vs_budget.py \
#   --metadata PBDD=/path/to/pbdd_metadata.csv \
#   --metadata Naive-Two-Stage-Cascade=/path/to/two_stage_metadata.csv \
#   --out figures/predefer_ratio/ratio_code_qwen.png

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
    "Naive-Two-Stage-Cascade",
]
METHOD_COLOR_MAP = {
    "PBDD": "#1f77b4",
    "Naive-Two-Stage-Cascade": "#2ca02c",
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
    "Naive-Two-Stage-Cascade": "Two-Stage Cascade",
}


def robust_read_csv(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        return [{str(k).strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def is_deferred_route(route: str) -> bool:
    return route in {"early_defer", "post_defer"}


def load_rows(metadata_path: Path) -> list[dict[str, str | float]]:
    rows = robust_read_csv(metadata_path)
    out: list[dict[str, str | float]] = []
    for row in rows:
        out.append(
            {
                "task_id": str(row.get("task_id", "")).strip(),
                "route": normalize_route(row.get("route")),
                "first_p_defer": to_float(row.get("first_p_defer"), 0.0),
                "post_p_defer": to_float(row.get("post_p_defer"), 0.0),
                "p_defer": to_float(row.get("p_defer"), 0.0),
            }
        )
    return out


def sort_key_for_budget(row: dict[str, str | float]) -> tuple[float, str]:
    # Same policy as routing budget sweep: less confident self (higher defer tendency) gets deferred first.
    # Here we use p_self proxy = 1 - p_defer.
    p_defer = to_float(row.get("p_defer"), 0.0)
    p_self_proxy = 1.0 - p_defer
    return (p_self_proxy, str(row.get("task_id", "")))


def compute_predefer_ratio_curve(rows: list[dict[str, str | float]]) -> dict[str, np.ndarray | float]:
    budgets = np.linspace(0, 1, 101)
    total = len(rows)
    if total == 0:
        return {
            "budgets": budgets,
            "predefer_ratio": np.zeros_like(budgets),
            "actual_budget": 0.0,
            "actual_ratio": 0.0,
        }

    prepared = [dict(row) for row in rows]
    prepared.sort(key=sort_key_for_budget)
    routes = [normalize_route(str(row.get("route", ""))) for row in prepared]

    ratios = []
    for budget in budgets:
        n_defer = int(round(float(budget) * total))
        if n_defer <= 0:
            ratios.append(0.0)
            continue
        selected = routes[:n_defer]
        early = sum(1 for route in selected if route == "early_defer")
        ratios.append(early / n_defer * 100.0)

    actual_defer_rows = [normalize_route(str(row.get("route", ""))) for row in rows if is_deferred_route(normalize_route(str(row.get("route", ""))))]
    actual_defer = len(actual_defer_rows)
    if actual_defer > 0:
        actual_ratio = sum(1 for route in actual_defer_rows if route == "early_defer") / actual_defer * 100.0
    else:
        actual_ratio = 0.0
    actual_budget = actual_defer / total * 100.0

    return {
        "budgets": budgets,
        "predefer_ratio": np.asarray(ratios, dtype=float),
        "actual_budget": float(actual_budget),
        "actual_ratio": float(actual_ratio),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pre-defer ratio under different defer budgets.")
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
                print(f"[skip] empty csv: {path}")
                continue
            label = label_override if label_override else (path.parent.name or path.stem)
            curve = compute_predefer_ratio_curve(rows)
            entries.append({"label": label, "curve": curve})
        except Exception as exc:
            print(f"[skip] failed to parse {path}: {exc}")

    if not entries:
        raise SystemExit("No valid metadata rows found.")

    labels = ordered({str(entry["label"]) for entry in entries}, METHOD_ORDER)
    label_to_color = {label: color_for_label(label) for label in labels}
    entries.sort(key=lambda item: labels.index(str(item["label"])) if str(item["label"]) in labels else 10**9)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    handles = []
    legend_text = []
    y_values: list[float] = []
    for entry in entries:
        label = str(entry["label"])
        curve = entry["curve"]
        budgets = np.asarray(curve["budgets"]) * 100.0
        ratios = np.asarray(curve["predefer_ratio"])
        line_zorder = 12 if label == "PBDD" else 4
        line, = ax.plot(
            budgets,
            ratios,
            linewidth=1.6,
            color=label_to_color[label],
            alpha=0.96,
            zorder=line_zorder,
        )
        handles.append(line)
        legend_text.append(display_label(label))
        y_values.extend(float(x) for x in ratios.tolist())

        actual_budget = float(curve["actual_budget"])
        actual_ratio = float(curve["actual_ratio"])
        ax.scatter(
            [actual_budget],
            [actual_ratio],
            s=16,
            color=label_to_color[label],
            alpha=0.9,
            zorder=line_zorder + 1,
        )
        y_values.append(actual_ratio)

    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    if y_values:
        lower = max(0.0, min(y_values) - 3.0)
        upper = min(100.0, max(y_values) + 3.0)
        if upper - lower < 8.0:
            center = (upper + lower) / 2.0
            lower = max(0.0, center - 4.0)
            upper = min(100.0, center + 4.0)
        ax.set_ylim(lower, upper)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("Total Defer Budget (%)", fontsize=11)
    ax.set_ylabel("Pre-Defer Ratio (%)", fontsize=11)
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
