from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_math_eval_on_path() -> Path:
    candidates = []

    env_path = os.environ.get("SELF_REF_MATH_EVAL_DIR")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    self_ref_root = Path(__file__).resolve().parent
    candidates.extend(
        [
            self_ref_root / "third_party" / "Qwen2.5-Math" / "evaluation",
            self_ref_root.parent / "Qwen2.5-Math" / "evaluation",
            self_ref_root.parent.parent / "Qwen2.5-Math" / "evaluation",
        ]
    )

    for candidate in candidates:
        if candidate.is_dir():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return candidate

    raise ImportError(
        "Math evaluation helpers not found. Set SELF_REF_MATH_EVAL_DIR or place "
        "the evaluation package under third_party/Qwen2.5-Math/evaluation."
    )
