"""MATH step2: extract hidden states from math sliced drafts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motivation_hidden.common.hidden_extraction import add_extraction_args, run_extraction  # noqa: E402


def main() -> None:
    parser = add_extraction_args(argparse.ArgumentParser())
    run_extraction(parser.parse_args())


if __name__ == "__main__":
    main()
