from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from memory_agent.agent import controller
from memory_agent.baselines import full_context, no_memory, vanilla_rag

FIXTURES = Path(__file__).parent / "fixtures"
OUTPUTS = Path(__file__).parent / "_output"


class ConstantEmbed:
    def __init__(self):
        self.call_count = 0

    def embed(self, texts):
        self.call_count += 1
        rows = np.zeros((len(texts), 1024), dtype=np.float32)
        rows[:, 0] = 1.0
        return rows


class ConstantLLM:
    def __init__(self, answer="a cat"):
        self.answer = answer
        self.call_count = 0

    def chat(self, *_args, **_kwargs):
        self.call_count += 1
        return self.answer


class ControllerLLM:
    def __init__(self):
        self.call_count = 0

    def chat(self, prompt, *_args, **_kwargs):
        self.call_count += 1
        if "You are a memory extractor" in prompt:
            if "no longer" in prompt:
                return '[{"text": "Alice no longer owns a cat.", "source_turn_ids": ["D2:1"]}]'
            return '[{"text": "Alice owns a cat.", "source_turn_ids": ["D1:1"]}]'
        if "You are a memory curator" in prompt:
            return '{"label": "UPDATE", "reason": "The ownership status changed."}'
        return "Alice no longer owns a cat."


def test_baseline_cli_entries_write_traces(monkeypatch):
    data_dir = FIXTURES / "simplified"
    baseline_cases = [
        (no_memory, "B0"),
        (full_context, "B1"),
        (vanilla_rag, "B2"),
    ]
    for module, name in baseline_cases:
        out_dir = OUTPUTS / name
        monkeypatch.setattr(module, "load_llm", lambda: ConstantLLM())
        if module is vanilla_rag:
            monkeypatch.setattr(module, "load_embed", lambda: ConstantEmbed())
        monkeypatch.setattr(
            sys,
            "argv",
            [module.__name__, "--data", str(data_dir), "--out", str(out_dir)],
        )

        module.main()

        assert (out_dir / "traces.jsonl").exists()
        assert json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))["questions"] == 1


def test_conflict_aware_controller_cli_writes_raw_dialogue_and_update(monkeypatch):
    data_dir = FIXTURES / "simplified"
    out_dir = OUTPUTS / "S2"
    monkeypatch.setattr(controller, "load_llm", lambda: ControllerLLM())
    monkeypatch.setattr(controller, "load_embed", lambda: ConstantEmbed())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            controller.__name__,
            "--mode",
            "conflict_aware",
            "--data",
            str(data_dir),
            "--out",
            str(out_dir),
        ],
    )

    controller.main()

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    dumped_store = json.loads((out_dir / "stores" / "d1.json").read_text(encoding="utf-8"))
    assert (out_dir / "raw_dialogues.json").exists()
    assert metrics["write_actions_summary"]["d1"] == {"ADD": 1, "UPDATE": 1}
    assert [item["status"] for item in dumped_store["memories"]] == ["superseded", "active"]
