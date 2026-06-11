"""Letta adapter (optional).

Letta is a full agent framework whose memory is exposed through archival memory
(`archival_memory_insert` / `archival_memory_search`). We map our contract onto
a Letta agent's archival store so it fits the same switchable interface. This is
heavier than mem0/vector — use it when you actually want Letta's self-editing,
long-running agent semantics.

Requires: pip install "letta-client>=0.1" and a running Letta server
(set LETTA_BASE_URL, default http://localhost:8283).
"""

from __future__ import annotations

import os

from ..config import Config
from ..types import MemoryItem, Message


class LettaBackend:
    name = "letta"

    def __init__(self, config: Config, agent_id: str | None = None):
        try:
            from letta_client import Letta
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "letta backend selected but not installed. "
                'Run: pip install "letta-client>=0.1" and start a Letta server.'
            ) from e
        base_url = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
        self._client = Letta(base_url=base_url)
        self._agent_id = agent_id or os.getenv("LETTA_AGENT_ID")
        if not self._agent_id:
            raise ValueError(
                "Letta backend needs an agent. Set LETTA_AGENT_ID to an existing "
                "Letta agent id (one archival store per agent)."
            )

    def add(
        self, messages: list[Message], user_id: str, metadata: dict | None = None
    ) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        for m in messages:
            if m.role != "user":
                continue
            passage = self._client.agents.passages.create(
                agent_id=self._agent_id, text=m.content
            )
            items.append(
                MemoryItem(id=str(passage.id), text=m.content, user_id=user_id)
            )
        return items

    def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryItem]:
        results = self._client.agents.passages.search(
            agent_id=self._agent_id, query=query, limit=limit
        )
        return [
            MemoryItem(
                id=str(r.id), text=r.text, score=float(getattr(r, "score", 0.0) or 0.0),
                user_id=user_id,
            )
            for r in results
        ]

    def get_all(self, user_id: str) -> list[MemoryItem]:
        passages = self._client.agents.passages.list(agent_id=self._agent_id)
        return [MemoryItem(id=str(p.id), text=p.text, user_id=user_id) for p in passages]

    def delete(self, memory_id: str, user_id: str) -> None:
        self._client.agents.passages.delete(agent_id=self._agent_id, memory_id=memory_id)
