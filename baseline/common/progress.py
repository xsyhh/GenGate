from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TypeVar


T = TypeVar("T")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal envs
    tqdm = None


def progress_iter(iterable: Iterable[T], *, desc: str, total: int | None = None) -> Iterable[T]:
    if os.environ.get("DISABLE_TQDM", "").strip() in {"1", "true", "TRUE", "yes"}:
        return iterable
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
