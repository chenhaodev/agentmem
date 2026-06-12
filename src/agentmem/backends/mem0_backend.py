"""mem0 adapter — the baseline long-term backend.

Wraps the mem0 library configured for a fully-local stack:
  - LLM       : DeepSeek (mem0's `deepseek` provider) — does the fact extraction
  - embedder  : HuggingFace sentence-transformers (CPU, no API key)
  - vectorDB  : embedded Qdrant (local path, no server)
mem0 does its own extraction + ADD/UPDATE/DELETE/NOOP reconciliation, so we hand
it raw messages and normalize its results into MemoryItem.

Verified live against mem0 2.0.5: `add` takes user_id; `search`/`get_all` take
`filters={"user_id": ...}` (top-level user_id was removed in mem0 2.x).

Requires: pip install "mem0ai>=2.0" qdrant-client sentence-transformers
"""

from __future__ import annotations

import os

from ..config import Config
from ..types import MemoryItem, Message


def _build_mem0_config(config: Config) -> dict:
    """A complete, runnable mem0 config — LLM + embedder + vector store.

    Defaults to a no-external-service setup (DeepSeek + local HF embeddings +
    embedded Qdrant) so the backend works with only a DeepSeek key. mem0's
    DeepSeek provider also reads DEEPSEEK_API_KEY from env; we pass it explicitly.
    """
    qdrant_path = config.mem0_vector_path
    os.makedirs(qdrant_path, exist_ok=True)
    return {
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": config.deepseek_model,
                "api_key": config.deepseek_api_key,
                "deepseek_base_url": config.deepseek_base_url,
            },
        },
        "embedder": {
            "provider": config.mem0_embedder_provider,
            "config": {"model": config.embedding_model},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": qdrant_path,
                "collection_name": "agentmem",
                "embedding_model_dims": config.mem0_embedding_dims,
            },
        },
    }


class Mem0Backend:
    name = "mem0"

    def __init__(self, config: Config):
        try:
            from mem0 import Memory
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "mem0 backend selected but not installed. "
                'Run: pip install "mem0ai>=2.0" qdrant-client sentence-transformers'
            ) from e
        self._mem = Memory.from_config(_build_mem0_config(config))

    @staticmethod
    def _to_item(raw: dict) -> MemoryItem:
        return MemoryItem(
            id=str(raw.get("id", "")),
            text=raw.get("memory", raw.get("text", "")),
            score=float(raw.get("score", 0.0) or 0.0),
            user_id=raw.get("user_id"),
            metadata=raw.get("metadata") or {},
        )

    @staticmethod
    def _unwrap(res) -> list[dict]:
        results = res.get("results", res) if isinstance(res, dict) else res
        return results or []

    def add(
        self, messages: list[Message], user_id: str, metadata: dict | None = None
    ) -> list[MemoryItem]:
        res = self._mem.add(
            [m.as_dict() for m in messages], user_id=user_id, metadata=metadata
        )
        return [self._to_item(r) for r in self._unwrap(res)]

    def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryItem]:
        # mem0 2.x search takes `top_k` (not `limit`); a wrong kwarg is silently
        # ignored and returns everything, so slice too as a hard guarantee.
        res = self._mem.search(query, filters={"user_id": user_id}, top_k=limit)
        return [self._to_item(r) for r in self._unwrap(res)][:limit]

    def get_all(self, user_id: str) -> list[MemoryItem]:
        res = self._mem.get_all(filters={"user_id": user_id})
        return [self._to_item(r) for r in self._unwrap(res)]

    def delete(self, memory_id: str, user_id: str) -> None:
        self._mem.delete(memory_id=memory_id)
