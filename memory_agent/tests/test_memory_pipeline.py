from __future__ import annotations

import numpy as np
import pytest

from memory_agent.memory.retriever import DenseRetriever
from memory_agent.memory.store import MemoryStore, MemoryUnit
from memory_agent.memory.updater import MemoryUpdater
from memory_agent.memory.writer import MemoryWriter


class FakeEmbed:
    def __init__(self, vectors):
        self.vectors = vectors
        self.call_count = 0

    def embed(self, texts):
        self.call_count += 1
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class QueueLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def chat(self, *_args, **_kwargs):
        self.call_count += 1
        return self.responses.pop(0)


def _unit(text: str) -> MemoryUnit:
    return MemoryUnit.new(text=text, source_session_id="s1")


def test_store_filters_all_stale_rows_and_checks_dimensions():
    store = MemoryStore(2)
    for idx in range(5):
        stale = _unit(f"stale-{idx}")
        store.add(stale, np.array([1.0, 0.0]))
        store.update_status(stale.id, "superseded")
    active = _unit("active")
    store.add(active, np.array([0.0, 1.0]))

    assert store.search(np.array([1.0, 0.0]), k=1)[0][0].id == active.id
    with pytest.raises(ValueError):
        store.add(_unit("wrong-dim"), np.array([1.0, 0.0, 0.0]))


def test_writer_preserves_source_turn_ids():
    llm = QueueLLM([
        '[{"text": "Alice adopted a cat.", "importance": 8, "source_turn_ids": ["D1:1"]}]'
    ])
    writer = MemoryWriter(llm, "Dialogue:\n{dialogue}")

    units = writer.extract(
        session_id="s1",
        turns=[{"id": "D1:1", "speaker": "Alice", "text": "I adopted a cat."}],
    )

    assert units[0].source_turn_ids == ["D1:1"]
    assert units[0].importance == 8


def test_conflict_update_supersedes_old_memory_and_retrieves_new_one():
    vectors = {
        "Alice owns a cat.": [1.0, 0.0],
        "Alice no longer owns a cat.": [1.0, 0.0],
        "Does Alice own a pet?": [1.0, 0.0],
    }
    embed = FakeEmbed(vectors)
    llm = QueueLLM(['{"label": "UPDATE", "reason": "The ownership status changed."}'])
    store = MemoryStore(2)
    old = _unit("Alice owns a cat.")
    store.add(old, embed.embed([old.text])[0])
    updater = MemoryUpdater(store, embed, llm, "{new_text}\n{old_text}", mode="conflict_aware")

    new = _unit("Alice no longer owns a cat.")
    actions = updater.write([new])
    hits = DenseRetriever(store, embed, top_k=1).retrieve("Does Alice own a pet?")

    assert actions[0]["action"] == "UPDATE"
    assert old.status == "superseded"
    assert new.supersedes == old.id
    assert new.version == 2
    assert hits[0][0].id == new.id


def test_conflict_updater_checks_more_than_the_top_neighbor():
    vectors = {
        "related": [1.0, 0.0],
        "duplicate": [0.99, 0.1],
        "new": [1.0, 0.0],
    }
    embed = FakeEmbed(vectors)
    llm = QueueLLM([
        '{"label": "ADD", "reason": "Related but distinct."}',
        '{"label": "NOOP", "reason": "Duplicate."}',
    ])
    store = MemoryStore(2)
    store.add(_unit("related"), embed.embed(["related"])[0])
    store.add(_unit("duplicate"), embed.embed(["duplicate"])[0])
    updater = MemoryUpdater(
        store,
        embed,
        llm,
        "{new_text}\n{old_text}",
        mode="conflict_aware",
        similarity_threshold=0.5,
        top_k_neighbors=2,
    )

    actions = updater.write([_unit("new")])

    assert actions[0]["action"] == "NOOP"
    assert len(store.all_active()) == 2
