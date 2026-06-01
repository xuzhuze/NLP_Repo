from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from memory_agent.agent import ta_adapter

OUTPUTS = Path(__file__).parent / "_output"
TA_RUN_GENERATION = (
    Path(__file__).parents[1] / "eval" / "ta" / "eval_kit" / "run_generation.py"
)


class ConstantEmbed:
    def __init__(self):
        self.call_count = 0

    def embed(self, texts):
        self.call_count += 1
        rows = np.zeros((len(texts), 1024), dtype=np.float32)
        rows[:, 0] = 1.0
        return rows


class AdapterLLM:
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


def _conversation():
    return {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "sessions": [
            {
                "session_id": 1,
                "date_time": "1 Jan 2026",
                "turns": [{"dia_id": "D1:1", "speaker": "Alice", "text": "I own a cat."}],
            },
            {
                "session_id": 2,
                "date_time": "2 Jan 2026",
                "turns": [
                    {"dia_id": "D2:1", "speaker": "Alice", "text": "I no longer own a cat."}
                ],
            },
        ],
    }


def _patch_clients(monkeypatch):
    monkeypatch.setenv("MEMORY_AGENT_TRACE_DIR", str(OUTPUTS / "ta_traces"))
    monkeypatch.setattr(ta_adapter, "load_llm", lambda: AdapterLLM())
    monkeypatch.setattr(ta_adapter, "load_embed", lambda: ConstantEmbed())


def test_ta_eval_kit_can_dynamically_load_adapter_class():
    spec = importlib.util.spec_from_file_location("ta_run_generation", TA_RUN_GENERATION)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    loaded = module.load_agent_class("memory_agent.agent.ta_adapter:ConflictAwareMemoryAgent")

    assert loaded is ta_adapter.ConflictAwareMemoryAgent


def test_ta_baseline_adapters_follow_ingest_answer_contract(monkeypatch):
    _patch_clients(monkeypatch)
    for agent_class in [
        ta_adapter.NoMemoryAgent,
        ta_adapter.FullContextAgent,
        ta_adapter.VanillaRAGAgent,
    ]:
        agent = agent_class()
        agent.ingest(_conversation())

        assert agent.answer("Does Alice still own a cat?")


def test_ta_structured_adapters_follow_ingest_answer_contract(monkeypatch):
    _patch_clients(monkeypatch)
    for agent_class in [
        ta_adapter.AppendOnlyMemoryAgent,
        ta_adapter.ConflictAwareMemoryAgent,
    ]:
        agent = agent_class()
        agent.ingest(_conversation())
        answer = agent.answer("Does Alice still own a cat?")
        ingest = json.loads((agent.trace.run_dir / "ingest.json").read_text(encoding="utf-8"))

        assert answer == "Alice no longer owns a cat."
        assert (agent.trace.run_dir / "store.json").exists()
        assert ingest["active_memories"] >= 1


def test_ta_conflict_aware_adapter_supersedes_old_memory(monkeypatch):
    _patch_clients(monkeypatch)
    agent = ta_adapter.ConflictAwareMemoryAgent()

    agent.ingest(_conversation())

    store = json.loads((agent.trace.run_dir / "store.json").read_text(encoding="utf-8"))
    assert [memory["status"] for memory in store["memories"]] == ["superseded", "active"]
