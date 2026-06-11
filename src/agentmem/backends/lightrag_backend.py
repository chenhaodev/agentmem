"""LightRAG adapter — graph-based long-term memory.

Unlike the vector/mem0 adapters, LightRAG builds a **knowledge graph**: it runs
its own entity + relation extraction over inserted text (using DeepSeek as its
LLM), so we feed it the raw conversation and let it do the extraction. Retrieval
returns graph-aware context (entities/relations/chunks), which is LightRAG's
sweet spot for multi-hop / relational recall.

Design notes:
- LightRAG is async-first. We drive it on one persistent event loop so this
  adapter still satisfies the synchronous LongTermBackend Protocol.
- One LightRAG instance per user_id (separate working_dir) for tenant isolation.
- `search` uses QueryParam(only_need_context=True) so we get retrieved context
  back as memory rather than a generated answer.
- `get_all` / `delete` don't map onto a graph store; they raise clearly instead
  of returning misleading data. (add + search is the supported surface.)

Requires: pip install "lightrag-hku>=1.0"
"""

from __future__ import annotations

import asyncio
import os

from ..config import Config
from ..embeddings import Embedder, build_embedder
from ..types import MemoryItem, Message


class LightRAGBackend:
    name = "lightrag"

    def __init__(self, config: Config, embedder: Embedder | None = None):
        try:
            from lightrag import LightRAG  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "lightrag backend selected but not installed. "
                'Run: pip install "lightrag-hku>=1.0"'
            ) from e
        self.config = config
        self.embedder = embedder or build_embedder(config)
        self.base_dir = config.lightrag_working_dir
        # one persistent loop drives all async LightRAG calls
        self._loop = asyncio.new_event_loop()
        self._instances: dict[str, object] = {}
        self._counter = 0

    # -- async plumbing -------------------------------------------------------
    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def _embedding_dim(self) -> int:
        dim = getattr(self.embedder, "dim", -1)
        if dim and dim > 0:
            return dim
        # OpenAI-compatible embedders learn their dim on first call — probe once.
        return int(self.embedder.embed(["dimension probe"]).shape[1])

    def _make_llm_func(self):
        from lightrag.llm.openai import openai_complete_if_cache

        cfg = self.config

        async def llm_model_func(
            prompt, system_prompt=None, history_messages=None, **kwargs
        ):
            return await openai_complete_if_cache(
                cfg.deepseek_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=cfg.deepseek_base_url,
                api_key=cfg.deepseek_api_key,
                **kwargs,
            )

        return llm_model_func

    def _make_embedding_func(self):
        from lightrag.utils import EmbeddingFunc

        embedder = self.embedder

        async def _embed(texts):
            return embedder.embed(list(texts))

        return EmbeddingFunc(
            embedding_dim=self._embedding_dim(),
            max_token_size=8192,
            func=_embed,
        )

    def _get_instance(self, user_id: str):
        """Lazily create + initialize a per-user LightRAG knowledge graph."""
        if user_id in self._instances:
            return self._instances[user_id]
        from lightrag import LightRAG

        working_dir = os.path.join(self.base_dir, user_id)
        os.makedirs(working_dir, exist_ok=True)
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=self._make_llm_func(),
            embedding_func=self._make_embedding_func(),
        )
        self._run(rag.initialize_storages())
        # pipeline status is required by newer LightRAG versions; optional on old.
        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status

            self._run(initialize_pipeline_status())
        except Exception:
            pass
        self._instances[user_id] = rag
        return rag

    # -- LongTermBackend contract --------------------------------------------
    def add(
        self, messages: list[Message], user_id: str, metadata: dict | None = None
    ) -> list[MemoryItem]:
        text = "\n".join(f"{m.role}: {m.content}" for m in messages).strip()
        if not text:
            return []
        rag = self._get_instance(user_id)
        self._run(rag.ainsert(text))  # LightRAG extracts entities/relations itself
        self._counter += 1
        return [
            MemoryItem(
                id=f"lightrag-{user_id}-{self._counter}",
                text=text,
                user_id=user_id,
                metadata={**(metadata or {}), "backend": "lightrag"},
            )
        ]

    def search(self, query: str, user_id: str, limit: int = 5) -> list[MemoryItem]:
        from lightrag import QueryParam

        rag = self._get_instance(user_id)
        context = self._run(
            rag.aquery(
                query,
                param=QueryParam(mode="hybrid", top_k=limit, only_need_context=True),
            )
        )
        context = (context or "").strip() if isinstance(context, str) else str(context)
        if not context:
            return []
        # LightRAG returns a synthesized graph context blob, not discrete rows.
        return [
            MemoryItem(
                id=f"lightrag-ctx-{user_id}",
                text=context,
                score=1.0,
                user_id=user_id,
                metadata={"backend": "lightrag", "mode": "hybrid"},
            )
        ]

    def get_all(self, user_id: str) -> list[MemoryItem]:
        raise NotImplementedError(
            "LightRAG stores a knowledge graph, not an enumerable list of "
            "memories. Use search(query, user_id). To reset, delete the user's "
            f"working_dir under {self.base_dir!r}."
        )

    def delete(self, memory_id: str, user_id: str) -> None:
        raise NotImplementedError(
            "LightRAG adapter does not support granular delete (graph store). "
            "Delete the user's working_dir to reset their memory."
        )
