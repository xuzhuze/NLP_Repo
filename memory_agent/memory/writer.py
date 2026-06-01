"""Memory Writer: extract structured memory units from a finished session."""
from __future__ import annotations

import json
import re
from typing import Any

from ..agent.llm_client import LLMClient
from .store import MemoryUnit


def _format_dialogue(turns: list[dict[str, Any]]) -> str:
    lines = []
    for idx, t in enumerate(turns):
        turn_id = t.get("id") or t.get("turn_id") or t.get("dia_id") or f"turn_{idx}"
        speaker = t.get("speaker") or t.get("role", "?")
        text = t.get("text") or t.get("content", "")
        lines.append(f"[{turn_id}] {speaker}: {text}")
    return "\n".join(lines)


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    """Best-effort JSON array extraction (strips code fences, finds first [...])."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return [d for d in data if isinstance(d, dict) and "text" in d]
    except json.JSONDecodeError:
        return []


class MemoryWriter:
    def __init__(self, llm: LLMClient, prompt_template: str, max_memories: int = 30):
        self.llm = llm
        self.tpl = prompt_template
        self.max_memories = max_memories

    def extract(
        self,
        session_id: str,
        turns: list[dict[str, Any]],
        session_timestamp: str = "",
    ) -> list[MemoryUnit]:
        if not turns:
            return []
        prompt = self.tpl.format(dialogue=_format_dialogue(turns))
        raw = self.llm.chat(prompt, temperature=0.0, max_tokens=1024)
        items = _parse_json_array(raw)[: self.max_memories]
        units: list[MemoryUnit] = []
        for item in items:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            try:
                importance = int(item.get("importance", 5))
            except (TypeError, ValueError):
                importance = 5
            source_turn_ids = item.get("source_turn_ids") or item.get("evidence_turn_ids") or []
            if isinstance(source_turn_ids, str):
                source_turn_ids = [source_turn_ids]
            units.append(
                MemoryUnit.new(
                    text=text,
                    source_session_id=session_id,
                    source_turn_ids=[str(turn_id) for turn_id in source_turn_ids],
                    timestamp=session_timestamp,
                    importance=max(1, min(10, importance)),
                )
            )
        return units
