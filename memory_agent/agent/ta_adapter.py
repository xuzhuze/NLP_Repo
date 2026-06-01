"""Agents compatible with the TA eval_kit ingest()/answer() interface."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .llm_client import load_embed, load_llm
from ..memory.retriever import DenseRetriever
from ..memory.store import MemoryStore, MemoryUnit
from ..memory.updater import MemoryUpdater, UpdateMode
from ..memory.writer import MemoryWriter

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROOT / "configs"


def _load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    prompts = yaml.safe_load((_CONFIG_DIR / "prompts.yaml").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((_CONFIG_DIR / "model.yaml").read_text(encoding="utf-8"))
    return prompts, model_cfg


def _history_text(conversation: dict[str, Any]) -> str:
    lines = []
    for session in conversation.get("sessions", []):
        lines.append(f"[Session {session.get('session_id', '')} @ {session.get('date_time', '')}]")
        for turn in session.get("turns", []):
            lines.append(f"{turn.get('speaker', '?')}: {turn.get('text', '')}")
    return "\n".join(lines)


class _TraceSink:
    """Persist raw input and per-answer traces without depending on eval_kit."""

    def __init__(self, system: str):
        default_root = _ROOT / "data" / "traces" / "ta_generation"
        root = Path(os.environ.get("MEMORY_AGENT_TRACE_DIR", default_root))
        self.run_dir = root / f"{system}_{uuid.uuid4().hex[:12]}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "traces.jsonl"

    def save_ingest(self, conversation: dict[str, Any], **extra: Any) -> None:
        record = {"conversation": conversation, **extra}
        (self.run_dir / "ingest.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log_answer(self, record: dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class _BaseAgent:
    def __init__(self, system: str):
        self.system = system
        self.prompts, self.model_cfg = _load_configs()
        self.llm = load_llm()
        self.trace = _TraceSink(system)

    def ingest(self, conversation: dict[str, Any]) -> None:
        self.trace.save_ingest(conversation)

    def _generate(
        self,
        question: str,
        prompt: str,
        retrieved: list[dict[str, Any]] | None = None,
    ) -> str:
        t0 = time.time()
        answer = self.llm.chat(
            prompt,
            temperature=self.model_cfg["generation"]["temperature"],
            max_tokens=self.model_cfg["generation"]["max_tokens"],
        ).strip()
        self.trace.log_answer({
            "system": self.system,
            "question": question,
            "prompt": prompt,
            "retrieved": retrieved or [],
            "prediction": answer,
            "latency_s": round(time.time() - t0, 3),
        })
        return answer


class NoMemoryAgent(_BaseAgent):
    """B0: answer without historical dialogue."""

    def __init__(self):
        super().__init__("B0_no_memory")

    def answer(self, question: str) -> str:
        prompt = self.prompts["qa_no_memory"].format(question=question)
        return self._generate(question, prompt)


class FullContextAgent(_BaseAgent):
    """B1: answer with the most recent part of the full dialogue."""

    def __init__(self):
        super().__init__("B1_full_context")
        self.history = ""

    def ingest(self, conversation: dict[str, Any]) -> None:
        self.history = _history_text(conversation)
        max_chars = self.model_cfg["full_context"]["max_context_chars"]
        if len(self.history) > max_chars:
            self.history = self.history[-max_chars:]
        self.trace.save_ingest(conversation, history_chars=len(self.history))

    def answer(self, question: str) -> str:
        prompt = self.prompts["qa_full_context"].format(
            history=self.history,
            question=question,
        )
        return self._generate(question, prompt)


class VanillaRAGAgent(_BaseAgent):
    """B2: retrieve raw dialogue turns instead of derived memory units."""

    def __init__(self):
        super().__init__("B2_vanilla_rag")
        self.embed = load_embed()
        self.store = MemoryStore(dim=self.model_cfg["retrieval"]["embed_dim"])
        self.retriever = DenseRetriever(
            self.store,
            self.embed,
            top_k=self.model_cfg["retrieval"]["top_k"],
        )

    def ingest(self, conversation: dict[str, Any]) -> None:
        chunks = []
        for session in conversation.get("sessions", []):
            session_id = str(session.get("session_id", ""))
            timestamp = str(session.get("date_time", ""))
            for idx, turn in enumerate(session.get("turns", [])):
                turn_id = str(turn.get("dia_id") or turn.get("id") or f"{session_id}_t{idx}")
                text = f"{turn.get('speaker', '?')}: {turn.get('text', '')}"
                chunks.append((turn_id, text, session_id, timestamp))
        if chunks:
            texts = [chunk[1] for chunk in chunks]
            batches = [self.embed.embed(texts[idx:idx + 32]) for idx in range(0, len(texts), 32)]
            vectors = np.vstack(batches)
            for (turn_id, text, session_id, timestamp), vector in zip(chunks, vectors):
                unit = MemoryUnit.new(
                    text=text,
                    source_session_id=session_id,
                    source_turn_ids=[turn_id],
                    timestamp=timestamp,
                )
                self.store.add(unit, vector)
        self.store.dump(self.trace.run_dir / "store.json")
        self.trace.save_ingest(
            conversation,
            vector_backend=self.store.backend,
            raw_turn_chunks=len(chunks),
        )

    def answer(self, question: str) -> str:
        hits = self.retriever.retrieve(question)
        memories = "\n".join(f"- {memory.text}" for memory, _ in hits) or "(no memories)"
        prompt = self.prompts["qa"].format(memories=memories, question=question)
        retrieved = [
            {"id": memory.id, "text": memory.text, "score": score}
            for memory, score in hits
        ]
        return self._generate(question, prompt, retrieved)


class _StructuredMemoryAgent(_BaseAgent):
    """Shared S1/S2 implementation using writer, updater, store, and retriever."""

    def __init__(self, mode: UpdateMode):
        super().__init__(f"S_{mode}")
        self.embed = load_embed()
        self.store = MemoryStore(dim=self.model_cfg["retrieval"]["embed_dim"])
        self.writer = MemoryWriter(
            self.llm,
            self.prompts["writer"],
            max_memories=self.model_cfg["writer"]["max_memories_per_session"],
        )
        self.updater = MemoryUpdater(
            store=self.store,
            embed_client=self.embed,
            llm=self.llm,
            prompt_template=self.prompts["conflict"],
            mode=mode,
            similarity_threshold=self.model_cfg["conflict"]["similarity_threshold"],
            top_k_neighbors=self.model_cfg["conflict"]["top_k_neighbors"],
        )
        self.retriever = DenseRetriever(
            self.store,
            self.embed,
            top_k=self.model_cfg["retrieval"]["top_k"],
        )

    def ingest(self, conversation: dict[str, Any]) -> None:
        actions = []
        for session in conversation.get("sessions", []):
            units = self.writer.extract(
                session_id=str(session.get("session_id", "")),
                turns=session.get("turns", []),
                session_timestamp=str(session.get("date_time", "")),
            )
            actions.extend(self.updater.write(units))
        self.store.dump(self.trace.run_dir / "store.json")
        self.trace.save_ingest(
            conversation,
            vector_backend=self.store.backend,
            write_actions=actions,
            active_memories=len(self.store.all_active()),
        )

    def answer(self, question: str) -> str:
        hits = self.retriever.retrieve(question)
        memories = "\n".join(f"- {memory.text}" for memory, _ in hits) or "(no memories)"
        prompt = self.prompts["qa"].format(memories=memories, question=question)
        retrieved = [
            {"id": memory.id, "text": memory.text, "score": score}
            for memory, score in hits
        ]
        return self._generate(question, prompt, retrieved)


class AppendOnlyMemoryAgent(_StructuredMemoryAgent):
    """S1: derived memories with append-only updates."""

    def __init__(self):
        super().__init__("append_only")


class ConflictAwareMemoryAgent(_StructuredMemoryAgent):
    """S2: derived memories with conflict-aware updates."""

    def __init__(self):
        super().__init__("conflict_aware")
