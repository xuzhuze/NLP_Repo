"""Memory Updater: append_only (config A) and conflict_aware (config B)."""
from __future__ import annotations

import json
import re
from typing import Literal

from ..agent.llm_client import LLMClient
from .store import MemoryStore, MemoryUnit

UpdateMode = Literal["append_only", "conflict_aware"]


def _parse_label(raw: str) -> tuple[str, str]:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return "ADD", "parse_fail"
    try:
        obj = json.loads(m.group(0))
        label = str(obj.get("label", "ADD")).upper()
        reason = str(obj.get("reason", ""))
        if label not in {"NOOP", "ADD", "UPDATE", "DELETE"}:
            label = "ADD"
        return label, reason
    except json.JSONDecodeError:
        return "ADD", "parse_fail"


class MemoryUpdater:
    def __init__(
        self,
        store: MemoryStore,
        embed_client: LLMClient,
        llm: LLMClient,
        prompt_template: str,
        mode: UpdateMode = "append_only",
        similarity_threshold: float = 0.75,
        top_k_neighbors: int = 3,
    ):
        self.store = store
        self.embed = embed_client
        self.llm = llm
        self.tpl = prompt_template
        self.mode = mode
        self.tau = similarity_threshold
        self.k_nb = top_k_neighbors

    def write(self, units: list[MemoryUnit]) -> list[dict]:
        if not units:
            return []
        texts = [u.text for u in units]
        embs = self.embed.embed(texts)
        actions: list[dict] = []
        for unit, emb in zip(units, embs):
            if self.mode == "append_only":
                self.store.add(unit, emb)
                actions.append({"action": "ADD", "id": unit.id, "text": unit.text})
                continue
            # conflict_aware
            neighbors = self.store.search(emb, k=self.k_nb, only_active=True)
            top = [(m, s) for m, s in neighbors if s >= self.tau]
            if not top:
                self.store.add(unit, emb)
                actions.append({"action": "ADD", "id": unit.id, "text": unit.text})
                continue
            comparisons = []
            for old, score in top:
                prompt = self.tpl.format(new_text=unit.text, old_text=old.text)
                raw = self.llm.chat(prompt, temperature=0.0, max_tokens=128)
                label, reason = _parse_label(raw)
                comparisons.append(
                    {"old": old.id, "score": score, "label": label, "reason": reason}
                )
                if label == "NOOP":
                    actions.append(
                        {"action": "NOOP", "old": old.id, "score": score, "reason": reason}
                    )
                    break
                if label == "UPDATE":
                    self.store.update_status(old.id, "superseded")
                    unit.supersedes = old.id
                    unit.version = old.version + 1
                    self.store.add(unit, emb)
                    actions.append(
                        {"action": "UPDATE", "id": unit.id, "old": old.id, "score": score,
                         "reason": reason}
                    )
                    break
                if label == "DELETE":
                    self.store.update_status(old.id, "deleted")
                    actions.append(
                        {"action": "DELETE", "old": old.id, "score": score, "reason": reason}
                    )
                    break
            else:
                self.store.add(unit, emb)
                actions.append(
                    {"action": "ADD", "id": unit.id, "comparisons": comparisons}
                )
        return actions
