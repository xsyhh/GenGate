#!/usr/bin/env python3
"""Build Table 1 joint-cost results and a machine-readable CSV.

This script follows the same accounting convention as build_joint_cost_report.py,
with one added option: two-stage methods can be recomputed with a shared
stage-0 threshold tau0 before the final routing cutoff is swept.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from build_route_markdown_report import (
    EvalRow,
    compute_full_budget_curve,
    display_label,
    is_self_ref_run,
    load_eval_rows,
    normalize_self_ref_rows_to_score_decision,
)


DOMAIN_ORDER = ["code", "math", "mmlu"]
MODEL_ORDER = ["qwen", "llama"]
METHOD_ORDER = [
    "Self-REF",
    "Answer-Probability",
    "Post-Linear",
    "External-Prompt-Router",
    "MC-Two-Stage-Probe",
    "Pre-Linear",
    "Roberta-Router",
    "GenGate",
]

DEFAULT_METHOD_KIND = {
    "GenGate": "two_stage",
    "PBDD": "two_stage",
    "MC-Two-Stage-Probe": "two_stage",
    "Self-REF": "post",
    "Post-Linear": "post",
    "Answer-Probability": "post",
    "External-Prompt-Router": "post",
    "Pre-Linear": "pre",
    "Roberta-Router": "pre",
}

TWO_STAGE_METHODS = {"GenGate", "PBDD", "MC-Two-Stage-Probe"}
ALPHA_TO_FIELD = {
    0.05: "cost_0.05",
    0.1: "cost_0.1",
    0.2: "cost_0.2",
}


@dataclass(frozen=True)
class MetadataEntry:
    method: str
    domain: str
    model: str
    path: Path


def fmt(value: float, digits: int = 2) -> str:
    if value is None or np.isnan(value) or np.isinf(value):
        return "NA"
    return f"{value:.{digits}f}"


def parse_metadata_spec(spec: str) -> MetadataEntry:
    if "=" not in spec:
        raise ValueError(f"--metadata expects METHOD_DOMAIN_MODEL=/path, got: {spec}")
    key, raw_path = spec.split("=", 1)
    parts = key.strip().rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"metadata key must be METHOD_DOMAIN_MODEL, got: {key}")
    method, domain, model = parts
    return MetadataEntry(
        method=method.strip(),
        domain=domain.strip().lower(),
        model=model.strip().lower(),
        path=Path(raw_path.strip()),
    )


def parse_expert_csv(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"--expert_csv expects DOMAIN=/path, got: {spec}")
    domain, path = spec.split("=", 1)
    return domain.strip().lower(), Path(path.strip())


def parse_alpha_list(text: str) -> list[float]:
    alphas: list[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            alphas.append(float(token))
    if not alphas:
        raise ValueError("No alpha values were provided.")
    return alphas


def recompute_two_stage(rows: list[EvalRow], tau0: float) -> list[EvalRow]:
    """Merge prior/posterior defer scores using a shared stage-0 threshold."""
    if not any(row.first_p_defer is not None and row.post_p_defer is not None for row in rows):
        return rows

    out: list[EvalRow] = []
    for row in rows:
        if row.first_p_defer is None or row.post_p_defer is None:
            out.append(row)
            continue

        if row.first_p_defer > tau0:
            route = "early_defer"
            p_defer = row.first_p_defer
        else:
            p_defer = row.post_p_defer
            route = "post_defer" if p_defer >= 0.5 else "post_self"

        out.append(
            EvalRow(
                task_id=row.task_id,
                route=route,
                self_passed=row.self_passed,
                expert_passed=row.expert_passed,
                p_defer=p_defer,
                p_self=1.0 - p_defer,
                score=1.0 - p_defer,
                first_p_defer=row.first_p_defer,
                post_p_defer=row.post_p_defer,
            )
        )
    return out


def compute_target_breakdown(rows: list[EvalRow], target_accuracy: float) -> dict[str, float]:
    total = len(rows)
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
    hits = np.flatnonzero(accuracies >= target_accuracy)
    k = int(hits[0]) if len(hits) else int(np.argmax(accuracies))
    deferred = prepared[:k]
    early_in_top_k = sum(1 for row in deferred if row.route == "early_defer")
    post_defer_in_top_k = sum(1 for row in deferred if row.route == "post_defer")

    return {
        "k": k,
        "total": total,
        "expert_rate": k / total * 100.0,
        "attempt_rate_two_stage": 100.0 - early_in_top_k / total * 100.0,
        "achieved_acc": float(accuracies[k]),
        "early_in_top_k": early_in_top_k,
        "post_defer_in_top_k": post_defer_in_top_k,
    }


def attempt_rate_for_kind(kind: str, expert_rate: float, breakdown: dict[str, float]) -> float:
    if kind == "two_stage":
        return float(breakdown["attempt_rate_two_stage"])
    if kind == "post":
        return 100.0
    if kind == "pre":
        return max(0.0, 100.0 - expert_rate)
    raise ValueError(f"Unknown method kind: {kind}")


def load_rows(entry: MetadataEntry, expert_csvs: dict[str, Path], tau0: float) -> list[EvalRow]:
    if entry.domain not in expert_csvs:
        raise ValueError(f"No expert CSV provided for domain '{entry.domain}'")
    rows = load_eval_rows(
        entry.path,
        expert_csv=expert_csvs[entry.domain],
        assume_expert_correct=False,
    )
    if is_self_ref_run(entry.method, entry.path):
        rows = normalize_self_ref_rows_to_score_decision(rows)
    elif entry.method in TWO_STAGE_METHODS:
        rows = recompute_two_stage(rows, tau0)
    return rows


def build_results(
    entries: list[MetadataEntry],
    expert_csvs: dict[str, Path],
    alphas: list[float],
    tau0: float,
    target_fraction: float,
    target_reference_method: str,
) -> list[dict[str, object]]:
    loaded: dict[tuple[str, str, str], dict[str, object]] = {}
    for entry in entries:
        rows = load_rows(entry, expert_csvs, tau0)
        kind = DEFAULT_METHOD_KIND.get(entry.method)
        if kind is None:
            raise ValueError(f"No method kind configured for {entry.method}")
        loaded[(entry.model, entry.domain, entry.method)] = {
            "entry": entry,
            "rows": rows,
            "kind": kind,
            "curve": compute_full_budget_curve(rows),
        }

    results: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for domain in DOMAIN_ORDER:
            reference_key = (model, domain, target_reference_method)
            if reference_key not in loaded:
                raise ValueError(f"Missing target reference: {reference_key}")
            expert_only = float(loaded[reference_key]["curve"]["expert_only_accuracy"])
            target_accuracy = expert_only * target_fraction

            for method in METHOD_ORDER:
                key = (model, domain, method)
                if key not in loaded:
                    raise ValueError(f"Missing metadata for {key}")
                item = loaded[key]
                entry = item["entry"]
                rows = item["rows"]
                kind = str(item["kind"])
                breakdown = compute_target_breakdown(rows, target_accuracy)
                expert_rate = float(breakdown["expert_rate"])
                attempt_rate = attempt_rate_for_kind(kind, expert_rate, breakdown)
                costs = {alpha: expert_rate + alpha * attempt_rate for alpha in alphas}

                result = {
                    "model": model,
                    "domain": domain,
                    "method": method,
                    "method_display": display_label(method),
                    "kind": kind,
                    "tau0": tau0 if method in TWO_STAGE_METHODS else "",
                    "expert_rate": expert_rate,
                    "attempt_rate": attempt_rate,
                    "achieved_acc": float(breakdown["achieved_acc"]),
                    "expert_only_acc": expert_only,
                    "target_acc": target_accuracy,
                    "target_fraction": target_fraction,
                    "metadata_path": str(entry.path),
                }
                for alpha, value in costs.items():
                    field = ALPHA_TO_FIELD.get(alpha, f"cost_{alpha:g}")
                    result[field] = value
                results.append(result)
    return results


def write_csv(results: list[dict[str, object]], out_csv: Path, alphas: list[float]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cost_fields = [ALPHA_TO_FIELD.get(alpha, f"cost_{alpha:g}") for alpha in alphas]
    fields = [
        "model",
        "domain",
        "method",
        "method_display",
        "kind",
        "tau0",
        "expert_rate",
        "attempt_rate",
        *cost_fields,
        "achieved_acc",
        "expert_only_acc",
        "target_acc",
        "target_fraction",
        "metadata_path",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    field: fmt(float(row[field]), 6)
                    if isinstance(row.get(field), float)
                    else row.get(field, "")
                    for field in fields
                }
            )


def rank_cells(results: list[dict[str, object]], alphas: list[float]) -> dict[tuple[str, str, str, float], str]:
    marks: dict[tuple[str, str, str, float], str] = {}
    for model in MODEL_ORDER:
        for domain in DOMAIN_ORDER:
            for alpha in alphas:
                field = ALPHA_TO_FIELD.get(alpha, f"cost_{alpha:g}")
                candidates = [
                    (str(row["method"]), float(row[field]))
                    for row in results
                    if row["model"] == model and row["domain"] == domain
                ]
                candidates.sort(key=lambda item: (item[1], METHOD_ORDER.index(item[0])))
                if candidates:
                    marks[(model, domain, candidates[0][0], alpha)] = "best"
                if len(candidates) > 1:
                    marks[(model, domain, candidates[1][0], alpha)] = "second"
    return marks


def cell_text(value: float, mark: str | None) -> str:
    text = fmt(value, 2)
    if mark == "best":
        return rf"\textbf{{{text}}}"
    if mark == "second":
        return rf"\underline{{{text}}}"
    return text


def alpha_header(alpha: float) -> str:
    suffix = f"{alpha:.2f}".split(".", 1)[1]
    return rf"$\text{{Cost}}_{{.{suffix}}}$"


def write_tex(results: list[dict[str, object]], out_tex: Path, alphas: list[float], tau0: float) -> None:
    by_key = {
        (str(row["model"]), str(row["domain"]), str(row["method"])): row
        for row in results
    }
    marks = rank_cells(results, alphas)
    alpha_headers = " & ".join(alpha_header(alpha) for alpha in alphas)
    lines: list[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Joint deployment cost at Route@$95\%$.",
        r"$\text{Cost}_\alpha = \text{ExpertRate} + \alpha\cdot\text{AttemptRate}$ is the joint cost at three attempt-cost ratios. ExpertRate is the minimum expert-call rate (\%) to reach $95\%$ of expert-only accuracy.",
        rf"For GenGate and MC Two-Stage Probe, $\tau_0={tau0:g}$. \textbf{{Bold}}/\underline{{underline}} mark the best/second-best value per column within each model block.}}",
        r"\label{tab:main-results}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l rrr rrr rrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Code} & \multicolumn{3}{c}{Math} & \multicolumn{3}{c}{MMLU} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        rf"Method & {alpha_headers} & {alpha_headers} & {alpha_headers} \\",
        r"\midrule",
    ]

    for model in MODEL_ORDER:
        model_label = "Qwen" if model == "qwen" else "LLaMA"
        lines.append(rf"\multicolumn{{10}}{{l}}{{\textbf{{{model_label} local model}}}} \\")
        for method in METHOD_ORDER:
            label = display_label(method)
            row_label = rf"\textbf{{{label}}}" if method == "GenGate" else label
            cells: list[str] = []
            for domain in DOMAIN_ORDER:
                row = by_key[(model, domain, method)]
                for alpha in alphas:
                    field = ALPHA_TO_FIELD.get(alpha, f"cost_{alpha:g}")
                    mark = marks.get((model, domain, method, alpha))
                    cells.append(cell_text(float(row[field]), mark))
            prefix = r"\rowcolor{gray!10}" + "\n" if method == "GenGate" else ""
            lines.append(prefix + row_label + " & " + " & ".join(cells) + r" \\")
        if model != MODEL_ORDER[-1]:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(results: list[dict[str, object]], out_md: Path, alphas: list[float], tau0: float) -> None:
    fields = [ALPHA_TO_FIELD.get(alpha, f"cost_{alpha:g}") for alpha in alphas]
    lines = [
        "# Table 1 Joint Deployment Cost",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Target: Route@95%",
        f"- Shared two-stage tau0: {tau0:g} for GenGate and MC Two-Stage Probe",
        "",
    ]
    for model in MODEL_ORDER:
        model_label = "Qwen" if model == "qwen" else "LLaMA"
        lines.extend([f"## {model_label} local model", ""])
        header = "| Method | " + " | ".join(
            f"{domain.upper()} Cost@{alpha:g}" for domain in DOMAIN_ORDER for alpha in alphas
        ) + " |"
        lines.append(header)
        lines.append("|---|" + "---:|" * (len(DOMAIN_ORDER) * len(alphas)))
        for method in METHOD_ORDER:
            cells = []
            for domain in DOMAIN_ORDER:
                row = next(
                    r for r in results
                    if r["model"] == model and r["domain"] == domain and r["method"] == method
                )
                cells.extend(fmt(float(row[field]), 2) for field in fields)
            lines.append(f"| {display_label(method)} | " + " | ".join(cells) + " |")
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Table 1 joint-cost CSV/TEX outputs.")
    parser.add_argument("--metadata", action="append", required=True,
                        help="METHOD_DOMAIN_MODEL=/path/to/metadata.csv")
    parser.add_argument("--expert_csv", action="append", required=True,
                        help="DOMAIN=/path/to/expert_results.csv")
    parser.add_argument("--tau0", type=float, default=0.7,
                        help="Shared stage-0 threshold for GenGate and MC Two-Stage Probe.")
    parser.add_argument("--target_fraction", type=float, default=0.95)
    parser.add_argument("--target_reference_method", default="GenGate")
    parser.add_argument("--alphas", default="0.05,0.1,0.2")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_tex", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    entries = [parse_metadata_spec(spec) for spec in args.metadata]
    for entry in entries:
        if not entry.path.exists():
            raise FileNotFoundError(entry.path)
    expert_csvs = dict(parse_expert_csv(spec) for spec in args.expert_csv)
    for path in expert_csvs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    alphas = parse_alpha_list(args.alphas)

    results = build_results(
        entries=entries,
        expert_csvs=expert_csvs,
        alphas=alphas,
        tau0=args.tau0,
        target_fraction=args.target_fraction,
        target_reference_method=args.target_reference_method,
    )
    write_csv(results, Path(args.out_csv), alphas)
    write_tex(results, Path(args.out_tex), alphas, args.tau0)
    write_markdown(results, Path(args.out_md), alphas, args.tau0)
    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved TeX: {args.out_tex}")
    print(f"Saved Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
