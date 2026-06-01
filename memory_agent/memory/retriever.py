"""Dense retriever. Three-factor scoring is an optional stretch goal."""
from __future__ import annotations

from .store import MemoryStore, MemoryUnit
from ..agent.llm_client import LLMClient


class DenseRetriever:
    def __init__(self, store: MemoryStore, embed_client: LLMClient, top_k: int = 5):
        self.store = store
        self.embed = embed_client
        self.top_k = top_k

    def retrieve(self, query: str) -> list[tuple[MemoryUnit, float]]:
        q_emb = self.embed.embed([query])
        return self.store.search(q_emb, k=self.top_k, only_active=True)
