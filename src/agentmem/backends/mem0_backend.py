"""mem0 adapter — the baseline long-term backend.

Wraps the mem0 library configured to use DeepSeek as its extraction LLM. mem0
does its own fact extraction + ADD/UPDATE/DELETE/NOOP reconciliation, so we
hand it raw messages and let it manage consistency. We normalize its results
into MemoryItem so callers never see mem0-specific shapes.

Requires: pip install "mem0ai>=0.1.40"
"""

from __future__ import annotations

from ..config import Config
from ..types import MemoryItem, Message


def _build_mem0_config(config: Config) -> dict:
    """mem0 reads DEEPSEEK_API_KEY from env; we also pass it explicitly.

    mem0 still needs an embedder. We default mem0 to its OpenAI embedder only if
    an OpenAI key exists; otherwise the caller should configure a local embedder.
    Kept minimal here — extend `embedder` per your deployment.
    """
    return {
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": config.deepseek_model,
                "api_key": config.deepseek_api_key,
                "deepseek_base_url": config.deepseek_base_url,
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
                'Run: pip install "mem0ai>=0.1.40"'
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

    def add(
        self, messages: list[Message], user_id: str, metadata: dict | None = None
    ) -> list[MemoryItem]:
        res = self._mem.add(
            [m.as_dict() for m in messages], user_id=user_id, metadata=metadata
        )
        results = res.get("results", res) if isinstance(res, dict) else res
        return [self._to_item(r) for r in (results or [])]

    def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryItem]:
        res = self._mem.search(query, user_id=user_id, limit=limit)
        results = res.get("results", res) if isinstance(res, dict) else res
        return [self._to_item(r) for r in (results or [])]

    def get_all(self, user_id: str) -> list[MemoryItem]:
        res = self._mem.get_all(user_id=user_id)
        results = res.get("results", res) if isinstance(res, dict) else res
        return [self._to_item(r) for r in (results or [])]

    def delete(self, memory_id: str, user_id: str) -> None:
        self._mem.delete(memory_id=memory_id)
