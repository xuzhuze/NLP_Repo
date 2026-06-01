"""FAISS + JSON metadata store. One store per experiment run."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    faiss = None


def _normalized_rows(values: np.ndarray, dim: int) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != dim:
        raise ValueError(f"Expected embeddings with shape (*, {dim}), got {rows.shape}")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rows / norms


class _NumpyIndexFlatIP:
    """Small FAISS-compatible fallback used when faiss-cpu is unavailable."""

    def __init__(self, dim: int):
        self.dim = dim
        self._rows = np.zeros((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self._rows)

    def add(self, rows: np.ndarray) -> None:
        self._rows = np.vstack([self._rows, rows])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = query @ self._rows.T
        take = min(k, self.ntotal)
        order = np.argsort(-scores, axis=1)[:, :take]
        values = np.take_along_axis(scores, order, axis=1)
        return values, order


@dataclass
class MemoryUnit:
    id: str
    text: str
    source_session_id: str
    source_turn_ids: list[str] = field(default_factory=list)
    timestamp: str = ""           # ISO datetime from dialogue if available
    importance: int = 5
    version: int = 1
    status: str = "active"         # active | superseded | deleted
    supersedes: str | None = None  # id of memory this replaces

    @staticmethod
    def new(**kwargs: Any) -> "MemoryUnit":
        return MemoryUnit(id=str(uuid.uuid4()), **kwargs)


class MemoryStore:
    """In-memory FAISS index (IndexFlatIP on normalized vectors == cosine).

    Metadata kept in a parallel dict keyed by id; index row → id via _ids list.
    """

    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim) if faiss else _NumpyIndexFlatIP(dim)
        self.backend = "faiss" if faiss else "numpy"
        self._ids: list[str] = []
        self._meta: dict[str, MemoryUnit] = {}

    def add(self, unit: MemoryUnit, embedding: np.ndarray) -> str:
        embedding = _normalized_rows(embedding, self.dim)
        if len(embedding) != 1:
            raise ValueError(f"Expected one embedding for one memory, got {len(embedding)}")
        self.index.add(embedding)
        self._ids.append(unit.id)
        self._meta[unit.id] = unit
        return unit.id

    def get(self, mid: str) -> MemoryUnit | None:
        return self._meta.get(mid)

    def update_status(self, mid: str, status: str) -> None:
        if mid in self._meta:
            self._meta[mid].status = status

    def search(
        self,
        query_emb: np.ndarray,
        k: int = 5,
        only_active: bool = True,
    ) -> list[tuple[MemoryUnit, float]]:
        if self.index.ntotal == 0:
            return []
        if k <= 0:
            return []
        query_emb = _normalized_rows(query_emb, self.dim)
        if len(query_emb) != 1:
            raise ValueError(f"Expected one query embedding, got {len(query_emb)}")
        # Fetch all rows when filtering so stale rows never hide active memories.
        fetch = self.index.ntotal if only_active else min(self.index.ntotal, k)
        scores, idxs = self.index.search(query_emb, fetch)
        out: list[tuple[MemoryUnit, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            unit = self._meta[self._ids[idx]]
            if only_active and unit.status != "active":
                continue
            out.append((unit, float(score)))
            if len(out) >= k:
                break
        return out

    def all_active(self) -> list[MemoryUnit]:
        return [m for m in self._meta.values() if m.status == "active"]

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dim": self.dim,
            "backend": self.backend,
            "memories": [asdict(m) for m in self._meta.values()],
            "order": self._ids,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
