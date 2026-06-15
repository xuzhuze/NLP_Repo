"""OpenAI-compatible client with retry. Used for chat, embedding, and judge."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# README asks users to put credentials in memory_agent/.env while commands run
# from the repository root. Load that package-local file explicitly.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key: str
    model: str

    @staticmethod
    def from_env(prefix: str) -> "ModelConfig":
        base_url = os.environ.get(f"{prefix}_BASE_URL")
        api_key = os.environ.get(f"{prefix}_API_KEY")
        model = os.environ.get(f"{prefix}_MODEL")
        if not (base_url and api_key and model):
            raise RuntimeError(
                f"Missing env vars: {prefix}_BASE_URL / {prefix}_API_KEY / {prefix}_MODEL"
            )
        return ModelConfig(base_url=base_url, api_key=api_key, model=model)


class LLMClient:
    """Stateless wrapper. One instance per (base_url, api_key) pair."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        self.call_count = 0

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        self.call_count += 1
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    # Some providers cap inputs per embedding request (e.g. DashScope
    # text-embedding-v3 allows at most 10). Chunk to stay within that limit.
    EMBED_BATCH = 10

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        self.call_count += 1
        resp = self._client.embeddings.create(model=self.cfg.model, input=batch)
        return [d.embedding for d in resp.data]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        items = list(texts)
        raw: list[list[float]] = []
        for start in range(0, len(items), self.EMBED_BATCH):
            raw.extend(self._embed_batch(items[start : start + self.EMBED_BATCH]))
        vecs = np.asarray(raw, dtype=np.float32)
        # L2-normalize so dot product == cosine similarity (FAISS IndexFlatIP)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def load_llm() -> LLMClient:
    return LLMClient(ModelConfig.from_env("LLM"))


def load_embed() -> LLMClient:
    return LLMClient(ModelConfig.from_env("EMBED"))


def load_judge() -> LLMClient:
    return LLMClient(ModelConfig.from_env("JUDGE"))
