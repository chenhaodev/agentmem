"""The common long-term backend contract.

Every framework (mem0, plain vector store, Letta, ...) is wrapped to satisfy
this Protocol. The interface is intentionally the lowest common denominator so
switching backends is a one-line config change. Backend-specific superpowers
(temporal queries, self-editing blocks) surface through MemoryItem.metadata
rather than widening this contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import MemoryItem, Message


@runtime_checkable
class LongTermBackend(Protocol):
    name: str

    def add(
        self,
        messages: list[Message],
        user_id: str,
        metadata: dict | None = None,
    ) -> list[MemoryItem]:
        """Persist memories derived from `messages`. Returns what was stored."""
        ...

    def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryItem]:
        """Relevance-ranked retrieval for a user."""
        ...

    def get_all(self, user_id: str) -> list[MemoryItem]:
        """All memories for a user."""
        ...

    def delete(self, memory_id: str, user_id: str) -> None:
        ...
