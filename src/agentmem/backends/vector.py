"""Plain vector-store backend — zero external frameworks.

A self-contained reference implementation: DeepSeek extracts atomic facts, a
pluggable embedder vectorizes them, and retrieval is numpy cosine similarity.
No mem0/Letta/Chroma required. This is the dependency-free fallback and the
clearest illustration of the contract every other adapter must also satisfy.

In-memory by design; persist `_store` to disk if you need durability.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..embeddings import Embedder, default_embedder
from ..llm import DeepSeekLLM
from ..types import MemoryItem, Message


class VectorStoreBackend:
    name = "vector"

    def __init__(
        self,
        llm: DeepSeekLLM,
        embedder: Embedder | None = None,
        persist_path: str = "",
    ):
        self.llm = llm
        self.embedder = embedder or default_embedder()
        self.persist_path = persist_path
        # user_id -> list of (MemoryItem, embedding)
        self._store: dict[str, list[tuple[MemoryItem, np.ndarray]]] = {}
        self._counter = 0
        if persist_path:
            self.load()

    def _next_id(self) -> str:
        self._counter += 1
        return f"vec-{self._counter}"

    def add(
        self, messages: list[Message], user_id: str, metadata: dict | None = None
    ) -> list[MemoryItem]:
        facts = self.llm.extract_facts(messages)
        if not facts:
            return []
        vecs = self.embedder.embed(facts)
        bucket = self._store.setdefault(user_id, [])
        stored: list[MemoryItem] = []
        for fact, vec in zip(facts, vecs):
            item = MemoryItem(
                id=self._next_id(),
                text=fact,
                user_id=user_id,
                metadata=dict(metadata or {}),
            )
            bucket.append((item, vec))
            stored.append(item)
        self._maybe_persist()
        return stored

    def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryItem]:
        bucket = self._store.get(user_id, [])
        if not bucket:
            return []
        q = self.embedder.embed([query])[0]
        scored = []
        for item, vec in bucket:
            score = float(np.dot(q, vec))  # both L2-normalized -> cosine
            scored.append(
                MemoryItem(
                    id=item.id,
                    text=item.text,
                    score=score,
                    user_id=user_id,
                    metadata=item.metadata,
                )
            )
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:limit]

    def get_all(self, user_id: str) -> list[MemoryItem]:
        return [item for item, _ in self._store.get(user_id, [])]

    def delete(self, memory_id: str, user_id: str) -> None:
        bucket = self._store.get(user_id, [])
        self._store[user_id] = [(i, v) for i, v in bucket if i.id != memory_id]
        self._maybe_persist()

    # -- durability -----------------------------------------------------------
    def _maybe_persist(self) -> None:
        if self.persist_path:
            self.save()

    def save(self) -> None:
        """Write the in-memory store to a single JSON file (vectors inlined)."""
        data = {
            "counter": self._counter,
            "users": {
                user_id: [
                    {
                        "id": item.id,
                        "text": item.text,
                        "metadata": item.metadata,
                        "vector": vec.tolist(),
                    }
                    for item, vec in bucket
                ]
                for user_id, bucket in self._store.items()
            },
        }
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        tmp = f"{self.persist_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.persist_path)  # atomic

    def load(self) -> None:
        if not os.path.exists(self.persist_path):
            return
        with open(self.persist_path) as f:
            data = json.load(f)
        self._counter = data.get("counter", 0)
        self._store = {
            user_id: [
                (
                    MemoryItem(
                        id=rec["id"],
                        text=rec["text"],
                        user_id=user_id,
                        metadata=rec.get("metadata") or {},
                    ),
                    np.asarray(rec["vector"], dtype=np.float32),
                )
                for rec in recs
            ]
            for user_id, recs in data.get("users", {}).items()
        }
