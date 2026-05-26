from __future__ import annotations

import random
from typing import Dict, List, Optional


def _is_self_record(record: Dict) -> bool:
    if record.get("_decision") == "<CN>":
        return True
    return int(record.get("_self_passed", 0)) == 1


def add_sampling_args(parser) -> None:
    parser.add_argument(
        "--max_total_samples",
        type=int,
        default=None,
        help="Maximum number of SFT records to write after balancing. Default keeps all records.",
    )
    parser.add_argument(
        "--self_ratio",
        type=float,
        default=None,
        help="Target fraction of <CN> records in the output. Default keeps the original ratio.",
    )


def balance_decision_records(
    records: List[Dict],
    *,
    max_total_samples: Optional[int],
    self_ratio: Optional[float],
    seed: int,
) -> List[Dict]:
    if max_total_samples is not None and max_total_samples <= 0:
        raise ValueError("--max_total_samples must be positive when provided")
    if self_ratio is not None and not 0.0 <= self_ratio <= 1.0:
        raise ValueError("--self_ratio must be in [0, 1] when provided")
    if max_total_samples is None and self_ratio is None:
        return records

    rng = random.Random(seed)
    self_records: List[Dict] = []
    defer_records: List[Dict] = []
    for record in records:
        if _is_self_record(record):
            self_records.append(record)
        else:
            defer_records.append(record)

    if self_ratio is None:
        total = len(records) if max_total_samples is None else min(max_total_samples, len(records))
        target_self = min(len(self_records), total)
        target_defer = total - target_self
        if target_defer > len(defer_records):
            target_defer = len(defer_records)
            target_self = min(len(self_records), total - target_defer)
    else:
        if self_ratio == 0.0:
            target_self = 0
            target_defer = len(defer_records)
        elif self_ratio == 1.0:
            target_self = len(self_records)
            target_defer = 0
        else:
            max_by_self = len(self_records) / self_ratio
            max_by_defer = len(defer_records) / (1.0 - self_ratio)
            total_float = min(float(len(records)), max_by_self, max_by_defer)
            if max_total_samples is not None:
                total_float = min(total_float, float(max_total_samples))

            target_self = int(total_float * self_ratio)
            target_defer = int(total_float * (1.0 - self_ratio))

            # Prefer using all scarce self records when it remains ratio-feasible.
            if target_self < len(self_records):
                possible_defer = round(target_self * (1.0 - self_ratio) / self_ratio)
                if possible_defer <= len(defer_records):
                    target_defer = possible_defer

            # Round down can leave one feasible pair on the table for common ratios like 0.5.
            while (
                max_total_samples is None or target_self + target_defer + 1 <= max_total_samples
            ) and target_self < len(self_records) and target_defer < len(defer_records):
                next_total = target_self + target_defer + 1
                next_self = round(next_total * self_ratio)
                next_defer = next_total - next_self
                if next_self > len(self_records) or next_defer > len(defer_records):
                    break
                if next_self == target_self and next_defer == target_defer:
                    break
                target_self, target_defer = next_self, next_defer

        if max_total_samples is not None and target_self + target_defer > max_total_samples:
            scale = max_total_samples / float(target_self + target_defer)
            target_self = int(target_self * scale)
            target_defer = int(target_defer * scale)

    selected = rng.sample(self_records, target_self) + rng.sample(defer_records, target_defer)
    rng.shuffle(selected)
    return selected


def describe_decision_records(records: List[Dict]) -> str:
    n_self = sum(1 for r in records if _is_self_record(r))
    n_defer = len(records) - n_self
    ratio = n_self / len(records) if records else 0.0
    return f"rows={len(records)}, self={n_self}, defer={n_defer}, self_ratio={ratio:.4f}"
