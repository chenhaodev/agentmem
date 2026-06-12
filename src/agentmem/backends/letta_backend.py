"""Letta adapter — archival memory in a stateful Letta agent.

Letta is a full agent server; we use its **archival memory** (semantically
searchable passages) as a long-term store. One agent per user_id gives tenant
isolation (mirrors the lightrag per-user dir). The agent is created with the
server's DeepSeek LLM handle plus an embedding config — Letta computes passage
embeddings server-side.

Verified live against a self-hosted Letta server (v0.16.8, letta-client 1.12.1)
with model `openai-proxy/deepseek-v4-flash` and Ollama `nomic-embed-text`
embeddings. Note: Letta's Ollama embedding endpoint must be OpenAI-compatible
(`.../v1`), so point LETTA_EMBEDDING_ENDPOINT at the Ollama `/v1` url.

Requires: pip install "letta-client>=1.0" + a running Letta server.
"""

from __future__ import annotations

from ..config import Config
from ..types import MemoryItem, Message


class LettaBackend:
    name = "letta"

    def __init__(self, config: Config):
        try:
            from letta_client import Letta
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "letta backend selected but not installed. "
                'Run: pip install "letta-client>=1.0" and start a Letta server.'
            ) from e
        self.config = config
        self._client = Letta(
            base_url=config.letta_base_url, timeout=config.deepseek_timeout * 3
        )
        self._agents: dict[str, str] = {}  # user_id -> agent_id

    # -- agent lifecycle ------------------------------------------------------
    def _embedding_kwargs(self) -> dict:
        cfg = self.config
        if cfg.letta_embedding_endpoint:  # explicit OpenAI-compatible endpoint
            return {
                "embedding_config": {
                    "embedding_endpoint_type": "openai",
                    "embedding_endpoint": cfg.letta_embedding_endpoint,
                    "embedding_model": cfg.letta_embedding_model,
                    "embedding_dim": cfg.letta_embedding_dim,
                    "embedding_chunk_size": 300,
                }
            }
        return {"embedding": cfg.letta_embedding_handle}

    def _agent_id(self, user_id: str) -> str:
        if user_id in self._agents:
            return self._agents[user_id]
        name = f"agentmem-{user_id}"
        # reuse an existing agent across runs (one archival store per agent)
        existing = [a for a in self._client.agents.list(name=name) if a.name == name]
        if existing:
            agent_id = existing[0].id
        else:
            agent = self._client.agents.create(
                name=name,
                model=self.config.letta_model,
                memory_blocks=[
                    {"label": "human", "value": ""},
                    {"label": "persona", "value": "I am a long-term memory store."},
                ],
                **self._embedding_kwargs(),
            )
            agent_id = agent.id
        self._agents[user_id] = agent_id
        return agent_id

    # -- LongTermBackend contract --------------------------------------------
    def add(
        self, messages: list[Message], user_id: str, metadata: dict | None = None
    ) -> list[MemoryItem]:
        agent_id = self._agent_id(user_id)
        items: list[MemoryItem] = []
        for m in messages:
            if m.role != "user":  # store user turns; Letta extracts/searches them
                continue
            created = self._client.agents.passages.create(agent_id=agent_id, text=m.content)
            for p in created if isinstance(created, list) else [created]:
                items.append(
                    MemoryItem(
                        id=str(p.id),
                        text=getattr(p, "text", m.content),
                        user_id=user_id,
                        metadata={"backend": "letta"},
                    )
                )
        return items

    def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryItem]:
        agent_id = self._agent_id(user_id)
        resp = self._client.agents.passages.search(
            agent_id=agent_id, query=query, top_k=limit
        )
        results = getattr(resp, "results", resp) or []
        out: list[MemoryItem] = []
        for r in results:
            text = r.get("content") if isinstance(r, dict) else getattr(r, "content", None)
            rid = r.get("id") if isinstance(r, dict) else getattr(r, "id", "")
            out.append(
                MemoryItem(
                    id=str(rid),
                    text=text or "",
                    score=1.0,  # Letta search returns no comparable score
                    user_id=user_id,
                    metadata={"backend": "letta"},
                )
            )
        return out

    def get_all(self, user_id: str) -> list[MemoryItem]:
        agent_id = self._agent_id(user_id)
        passages = self._client.agents.passages.list(agent_id=agent_id)
        return [
            MemoryItem(
                id=str(p.id),
                text=getattr(p, "text", ""),
                user_id=user_id,
                metadata={"backend": "letta"},
            )
            for p in passages
        ]

    def delete(self, memory_id: str, user_id: str) -> None:
        agent_id = self._agent_id(user_id)
        self._client.agents.passages.delete(memory_id=memory_id, agent_id=agent_id)
