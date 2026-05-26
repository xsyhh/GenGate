from __future__ import annotations

import sys
from pathlib import Path


SELF_REF_ROOT = Path(__file__).resolve().parent


def ensure_self_ref_root_on_path() -> Path:
    root_text = str(SELF_REF_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return SELF_REF_ROOT
