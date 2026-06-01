"""Launch the bundled TA run_generation.py from the repository root."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TA_KIT = Path(__file__).resolve().parent / "ta" / "eval_kit"


def main() -> None:
    sys.path.insert(0, str(_TA_KIT))
    runpy.run_path(str(_TA_KIT / "run_generation.py"), run_name="__main__")


if __name__ == "__main__":
    main()
