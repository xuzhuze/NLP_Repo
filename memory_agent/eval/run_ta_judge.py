"""Launch the bundled TA judge while mapping package-local JUDGE_* settings."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
_TA_KIT = Path(__file__).resolve().parent / "ta" / "eval_kit"


def _load_judge_environment() -> None:
    load_dotenv(_ROOT / ".env")
    for suffix in ["BASE_URL", "API_KEY", "MODEL"]:
        value = os.environ.get(f"JUDGE_{suffix}")
        if value:
            os.environ[f"LLM_{suffix}"] = value


def main() -> None:
    _load_judge_environment()
    sys.path.insert(0, str(_TA_KIT))
    runpy.run_path(str(_TA_KIT / "run_judge.py"), run_name="__main__")


if __name__ == "__main__":
    main()
