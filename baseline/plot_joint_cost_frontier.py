#!/usr/bin/env python3
"""Plot accuracy under a joint-cost budget (x = C_alpha, y = accuracy).

For each method, sweeps the routing threshold and computes:
  C_alpha = ExpertRate + alpha * AttemptRate
at each operating point, then plots accuracy vs C_alpha.

Method types determine AttemptRate behavior:
  - Post-generation (Self-REF, Answer Prob, Post Linear, Prompt Router): AttemptRate = 100%
  - Pre-generation (Pre Linear, RoBERTa): AttemptRate = 1 - ExpertRate
  - Two-stage (GenGate, MC Two-Stage): AttemptRate = fixed (fraction passing first gate)
"""

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
MODEL_ORDER = ["qwen", "llama"]
METHOD_ORDER = [
    "GenGate",
    "Self-REF",
    "MC-Two-Stage-Probe",
    "Post-Linear",
    "Pre-Linear",
    "Roberta-Router",
    "External-Prompt-Router",
    "Answer-Probability",
]

METHOD_COLOR_MAP = {
    "GenGate": "#1f77b4",
    "Self-REF": "#d62728",
    "MC-Two-Stage-Probe": "#ff9896",
    "Post-Linear": "#9467bd",
    "Pre-Linear": "#ff7f0e",
    "Roberta-Router": "#8c564b",
    "External-Prompt-Router": "#17becf",
    "Answer-Probability": "#e377c2",
}
DISPLAY_LABEL_MAP = {
    "MC-Two-Stage-Probe": "MC Two-Stage Probe",
    "External-Prompt-Router": "Prompt Router",
    "Answer-Probability": "Answer Prob",
    "Roberta-Router": "RoBERTa Router",
    "Post-Linear": "Post Linear",
    "Pre-Linear": "Pre Linear",
    "Always-Self": "Always self",
    "Always-Defer": "Always defer",
}

POST_GEN_METHODS = {"Self-REF", "Answer-Probability", "Post-Linear", "External-Prompt-Router"}
PRE_GEN_METHODS = {"Pre-Linear", "Roberta-Router"}
TWO_STAGE_METHODS = {"GenGate", "MC-Two-Stage-Probe"}


def robust_read_csv(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        return [{str(k).strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def display_label(label: str) -> str:
    return DISPLAY_LABEL_MAP.get(label, label)


def color_for_label(label: str) -> str:
    if label in METHOD_COLOR_MAP:
        return METHOD_COLOR_MAP[label]
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    return METHOD_COLOR_MAP.get(label, f"#{digest[:6]}")


def load_expert_map(expert_csv: str | Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in robust_read_csv(expert_csv):
        task_id = str(row.get("task_id", "")).strip()
        if task_id:
            out[task_id] = to_float(row.get("expert_passed", row.get("self_passed", 0.0)), 0.0)
    return out

def compute_joint_cost_frontier(
    rows: list[dict],
    label: str,
    alpha: float,
    pre_threshold: float,
    expert_map: dict[str, float],
) -> dict[str, np.ndarray]:
    """Compute accuracy and attempt_rate as functions of joint cost C_alpha."""
    total = len(rows)
    if total == 0:
        return {"costs": np.array([]), "accuracies": np.array([]), "attempt_rates": np.array([])}

    # Determine scores and self/expert passed
    scored_rows = []
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        self_passed = to_float(row.get("self_passed"), 0.0)
        expert_passed = expert_map.get(task_id, to_float(row.get("expert_passed"), 1.0))

        # Compute score based on method type
        first_p = str(row.get("first_p_defer", "")).strip()
        post_p = str(row.get("post_p_defer", "")).strip()

        if label in TWO_STAGE_METHODS and first_p and post_p:
            fp = to_float(first_p, 0.0)
            pp = to_float(post_p, 0.0)
            if fp > pre_threshold:
                score = 1.0 - fp  # low score = likely to defer
                attempted = False
            else:
                score = 1.0 - pp
                attempted = True
        elif label == "Self-REF":
            p_self = to_float(row.get("p_self"), 0.0)
            score = p_self
            attempted = True
        elif label == "Answer-Probability":
            score = to_float(row.get("score", row.get("p_self", 0.0)), 0.0)
            attempted = True
        else:
            score = to_float(row.get("score", row.get("p_self", 0.0)), 0.0)
            if label in PRE_GEN_METHODS:
                attempted = False  # will be determined by threshold
            else:
                attempted = True

        scored_rows.append({
            "score": score,
            "self_passed": self_passed,
            "expert_passed": expert_passed,
            "attempted": attempted,
            "task_id": task_id,
        })

    # Sort by score ascending (lowest score = most likely to defer first)
    scored_rows.sort(key=lambda r: (r["score"], r["task_id"]))
    self_passed = np.array([r["self_passed"] for r in scored_rows], dtype=float)
    expert_passed = np.array([r["expert_passed"] for r in scored_rows], dtype=float)
    attempted_flags = np.array([r["attempted"] for r in scored_rows], dtype=bool)

    # Sweep budget from 0 to 1
    budgets = np.linspace(0, 1, 201)
    costs = []
    accuracies = []
    attempt_rates = []

    for budget in budgets:
        n_defer = int(round(budget * total))
        # Accuracy: deferred use expert, kept use self
        acc = (expert_passed[:n_defer].sum() + self_passed[n_defer:].sum()) / total * 100.0

        # Attempt rate depends on method type
        if label in POST_GEN_METHODS:
            att_rate = 100.0
        elif label in PRE_GEN_METHODS:
            att_rate = (1.0 - budget) * 100.0
        elif label in TWO_STAGE_METHODS:
            # Attempted queries always pay generation cost.
            # Pre-deferred queries NOT sent to expert must fall back to local attempt.
            fallback_count = int((~attempted_flags[n_defer:]).sum())
            att_count = int(attempted_flags.sum()) + fallback_count
            att_rate = att_count / total * 100.0
        else:
            att_rate = 100.0

        joint_cost = budget * 100.0 + alpha * att_rate
        costs.append(joint_cost)
        accuracies.append(acc)
        attempt_rates.append(att_rate)

    return {
        "costs": np.array(costs),
        "accuracies": np.array(accuracies),
        "attempt_rates": np.array(attempt_rates),
    }

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy under a joint-cost budget.")
    parser.add_argument("--metadata", action="append", required=True,
                        help="LABEL=/path/to/metadata.csv")
    parser.add_argument("--expert_csv", action="append", default=None,
                        help="DOMAIN=/path/to/expert.csv (repeatable)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Relative cost of local attempt to expert call")
    parser.add_argument("--pre_threshold", type=float, default=0.7,
                        help="Stage-0 defer threshold for two-stage methods")
    parser.add_argument("--out", default="figures/joint_cost_frontier/joint_cost_budget.pdf")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--layout", choices=["full", "main-zoom"], default="full",
                        help="full: 2x3 complete grid; main-zoom: enlarged 2x2 panels where GenGate has the clearest advantage")
    args = parser.parse_args()

    # Parse expert CSVs per domain
    expert_maps: dict[str, dict[str, float]] = {}
    if args.expert_csv:
        for spec in args.expert_csv:
            if "=" in spec:
                domain_key, path = spec.split("=", 1)
                expert_maps[domain_key.strip().lower()] = load_expert_map(path.strip())

    # Parse metadata entries: expect LABEL_DOMAIN_MODEL=path
    # Format: "GenGate_code_qwen=/path/to/metadata.csv"
    entries = []
    for spec in args.metadata:
        if "=" not in spec:
            continue
        key, path_str = spec.split("=", 1)
        path = Path(path_str.strip())
        if not path.exists():
            print(f"[skip] {path}")
            continue
        parts = key.strip().split("_", 2)
        if len(parts) != 3:
            print(f"[skip] bad key format '{key}', expect LABEL_DOMAIN_MODEL")
            continue
        label, domain, model = parts[0], parts[1].lower(), parts[2].lower()

        rows = robust_read_csv(path)
        if not rows:
            continue

        domain_expert = expert_maps.get(domain, {})
        curve = compute_joint_cost_frontier(rows, label, args.alpha, args.pre_threshold, domain_expert)

        entries.append({
            "label": label,
            "domain": domain,
            "model": model,
            "curve": curve,
        })

    if not entries:
        raise SystemExit("No valid entries loaded.")

    # The main paper version enlarges the settings where GenGate separates most
    # clearly from the strongest baseline; the appendix keeps the complete grid.
    if args.layout == "main-zoom":
        panels = [
            ("code", "qwen", None, (45, 84)),
            ("math", "qwen", None, (75, 95)),
            ("mmlu", "qwen", None, (70, 93)),
            ("code", "llama", None, (38, 84)),
            ("math", "llama", None, (55, 94)),
            ("mmlu", "llama", None, (72, 94)),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2), squeeze=False)
        domain_col = {}
        model_row = {}
        panel_pos = {(domain, model): (idx // 3, idx % 3) for idx, (domain, model, _, _) in enumerate(panels)}
        label_size = 12
        title_size = 13
        tick_size = 10.5
        legend_size = 9.5
        pbdd_lw = 3.0
        two_stage_lw = 2.0
        other_lw = 1.75
    else:
        panels = [(domain, model, None, None) for model in MODEL_ORDER for domain in DOMAIN_ORDER]
        fig, axes = plt.subplots(2, 3, figsize=(14, 6), squeeze=False)
        domain_col = {"code": 0, "math": 1, "mmlu": 2}
        model_row = {"qwen": 0, "llama": 1}
        panel_pos = {}
        label_size = 10
        title_size = 11
        tick_size = 9
        legend_size = 8.5
        pbdd_lw = 2.2
        two_stage_lw = 1.5
        other_lw = 1.3

    domain_titles = {"code": "Code", "math": "Math", "mmlu": "MMLU"}
    model_titles = {"qwen": "Qwen", "llama": "LLaMA"}

    def get_linestyle(label: str) -> str:
        if label == "GenGate":
            return "-"
        if label in POST_GEN_METHODS:
            return "--"
        if label in PRE_GEN_METHODS:
            return ":"
        if label in TWO_STAGE_METHODS:
            return "-."
        return "-"

    legend_handles = {}
    for entry in entries:
        if args.layout == "main-zoom":
            pos = panel_pos.get((entry["domain"], entry["model"]))
            if pos is None:
                continue
            row, col = pos
        else:
            col = domain_col.get(entry["domain"])
            row = model_row.get(entry["model"])
            if col is None or row is None:
                continue
        if row is None or col is None:
            continue
        ax = axes[row][col]
        label = entry["label"]
        curve = entry["curve"]
        if len(curve["costs"]) == 0:
            continue

        color = color_for_label(label)
        if label == "GenGate":
            lw, alpha_val, zorder = pbdd_lw, 1.0, 12
        elif label in TWO_STAGE_METHODS:
            lw, alpha_val, zorder = two_stage_lw, 0.9, 8
        else:
            lw, alpha_val, zorder = other_lw, 0.82, 4
        line, = ax.plot(
            curve["costs"], curve["accuracies"],
            linewidth=lw, color=color, alpha=alpha_val, zorder=zorder,
            linestyle=get_linestyle(label),
        )
        if label not in legend_handles:
            legend_handles[label] = line

    # Format axes
    for idx, (domain, model, xlim, ylim) in enumerate(panels):
        row_idx, col_idx = divmod(idx, axes.shape[1])
        ax = axes[row_idx][col_idx]
        ax.set_xlabel(r"Cost budget ($\mathrm{Cost}_{" + f"{args.alpha}" + r"}$)", fontsize=label_size)
        if col_idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=label_size)
        ax.set_title(f"{domain_titles[domain]} | {model_titles[model]}", fontsize=title_size, fontweight="bold")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        ax.tick_params(labelsize=tick_size)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # Legend — vertical, in the bottom-right of both left-column subplots
    ordered_labels = [l for l in METHOD_ORDER if l in legend_handles]
    legend_rows = range(2)
    for row_idx in legend_rows:
        axes[row_idx][0].legend(
            [legend_handles[l] for l in ordered_labels],
            [display_label(l) for l in ordered_labels],
            loc="lower right",
            ncol=1,
            frameon=True, framealpha=0.92, fontsize=legend_size,
            handlelength=2.0,
        )
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {out_path}")
    print(f"Saved figure to: {png_path}")


if __name__ == "__main__":
    main()
