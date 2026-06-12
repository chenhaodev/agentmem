"""Long-term backend adapters + the factory that selects one from config."""

from __future__ import annotations

import re

from ..config import Config
from ..llm import DeepSeekLLM
from .base import LongTermBackend


def _build_single(name: str, config: Config, llm: DeepSeekLLM) -> LongTermBackend:
    """Build one backend by name. Adapters import lazily so you only need the
    deps for the one(s) you use."""
    if name == "vector":
        from ..embeddings import build_embedder
        from .vector import VectorStoreBackend

        return VectorStoreBackend(
            llm,
            embedder=build_embedder(config),
            persist_path=config.vector_persist_path,
        )
    if name == "mem0":
        from .mem0_backend import Mem0Backend

        return Mem0Backend(config)
    if name == "letta":
        from .letta_backend import LettaBackend

        return LettaBackend(config)
    if name == "lightrag":
        from .lightrag_backend import LightRAGBackend

        return LightRAGBackend(config)
    raise ValueError(
        f"Unknown long-term backend {name!r}. "
        "Expected one of: vector, mem0, letta, lightrag."
    )


def build_backend(config: Config, llm: DeepSeekLLM) -> LongTermBackend:
    """Select the long-term backend(s) from LONG_TERM_BACKEND.

    A single name (e.g. "mem0") builds that backend directly. A "+"/"," list
    (e.g. "vector+mem0") builds each and wraps them in a RouterBackend that fans
    out writes and merges reads — "long memory in different frameworks at once".
    """
    spec = config.long_term_backend.lower()
    names = [p.strip() for p in re.split(r"[+,]", spec) if p.strip()]
    if len(names) == 1:
        return _build_single(names[0], config, llm)

    from .router import RouterBackend

    children = {name: _build_single(name, config, llm) for name in names}
    return RouterBackend(
        children, merge=config.router_merge, write=config.router_write
    )


__all__ = ["LongTermBackend", "build_backend"]
