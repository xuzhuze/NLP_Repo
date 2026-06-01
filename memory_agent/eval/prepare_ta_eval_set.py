"""Build the TA-format stratified eval_set.json from a local LoCoMo file."""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[1]
_TA_PREPARE = _ROOT / "eval" / "ta" / "eval_kit" / "prepare_eval_set.py"


def _load_ta_prepare() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ta_prepare_eval_set", _TA_PREPARE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load TA prepare script: {_TA_PREPARE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(_ROOT / "data" / "locomo_official" / "locomo10.json"),
        help="Local official LoCoMo locomo10.json",
    )
    parser.add_argument(
        "--output",
        default=str(_ROOT / "data" / "eval_set.json"),
        help="TA-format output JSON",
    )
    parser.add_argument("--per_category", type=int, default=40)
    parser.add_argument("--categories", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Download the official LoCoMo locomo10.json first."
        )
    raw_data = json.loads(input_path.read_text(encoding="utf-8"))
    ta_prepare = _load_ta_prepare()
    eval_set = ta_prepare.build_eval_set(
        raw_data,
        per_category=args.per_category,
        include_categories=args.categories,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(eval_set, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    counts = Counter(
        qa["category_name"]
        for sample in eval_set
        for qa in sample["qa_list"]
    )
    print(f"[saved] {output_path}")
    print(json.dumps(dict(sorted(counts.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
