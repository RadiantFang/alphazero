from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    # Make `go_zero` importable when running `python scripts/train.py`
    # without requiring `PYTHONPATH=.`
    project_root = Path(__file__).resolve().parents[1]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_project_root_on_path()

from go_zero.train import main


if __name__ == "__main__":
    main()
