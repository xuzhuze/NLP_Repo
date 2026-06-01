"""Per-question trace logger. One JSONL line per question."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.out_dir / "traces.jsonl"
        self.metrics_path = self.out_dir / "metrics.json"
        self._fh = self.trace_path.open("w", encoding="utf-8")
        self._t0 = time.time()
        self._totals: dict[str, Any] = {"questions": 0, "elapsed_s": 0.0}

    def log(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._totals["questions"] += 1

    def close(self, extra: dict[str, Any] | None = None) -> None:
        self._totals["elapsed_s"] = round(time.time() - self._t0, 2)
        if extra:
            self._totals.update(extra)
        self.metrics_path.write_text(
            json.dumps(self._totals, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._fh.close()
