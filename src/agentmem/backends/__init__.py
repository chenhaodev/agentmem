"""Long-term backend adapters + the factory that selects one from config."""

from __future__ import annotations

from ..config import Config
from ..llm import DeepSeekLLM
from .base import LongTermBackend


def build_backend(config: Config, llm: DeepSeekLLM) -> LongTermBackend:
    """Switch backends with a single config value (LONG_TERM_BACKEND).

    Adapters are imported lazily so you only need the deps for the one you use.
    """
    name = config.long_term_backend.lower()
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
    raise ValueError(
        f"Unknown LONG_TERM_BACKEND={name!r}. Expected one of: vector, mem0, letta."
    )


__all__ = ["LongTermBackend", "build_backend"]
