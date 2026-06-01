from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).parent


def test_ta_generation_wrapper_runs_end_to_end():
    output = TESTS / "_output" / "ta_cli" / "nested" / "predictions.json"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "memory_agent.eval.run_ta_generation",
            "--eval_set",
            str(TESTS / "fixtures" / "ta_eval_set.json"),
            "--agent",
            "memory_agent.tests.fake_ta_agent:FakeTAAgent",
            "--output",
            str(output),
        ],
        check=True,
        env=env,
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )

    predictions = json.loads(output.read_text(encoding="utf-8"))
    assert predictions[0]["qa_id"] == "conv-test_q0"
    assert predictions[0]["prediction"] == "a cat"
