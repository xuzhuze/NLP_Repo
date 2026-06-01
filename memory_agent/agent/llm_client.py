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

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self.call_count += 1
        resp = self._client.embeddings.create(model=self.cfg.model, input=list(texts))
        vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
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
