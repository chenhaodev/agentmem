"""Configuration loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()  # populate os.environ from .env if present
except Exception:  # python-dotenv optional at import time
    pass


@dataclass
class Config:
    # --- DeepSeek (OpenAI-compatible) ---
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_timeout: int = 60
    deepseek_max_retries: int = 3

    # --- backend selection ---
    long_term_backend: str = "vector"  # "vector" | "mem0" | "letta" | "lightrag"

    # --- lightrag backend ---
    lightrag_working_dir: str = ".data/lightrag"  # per-user subdir created under this

    # --- mem0 backend (local stack: deepseek + HF embeddings + embedded qdrant) ---
    mem0_embedder_provider: str = "huggingface"
    mem0_embedding_dims: int = 384  # all-MiniLM-L6-v2 -> 384; match your embedder
    mem0_vector_path: str = ".data/mem0_qdrant"

    # --- short-term ---
    short_term_store: str = "memory"  # "memory" | "redis"
    short_term_max_turns: int = 12  # raw turns kept per session buffer
    redis_url: str = "redis://localhost:6379/0"
    redis_namespace: str = "agentmem:st"  # key prefix for shared short-term

    # --- embeddings (vector backend) ---
    embedding_provider: str = "auto"  # "auto" | "hash" | "sentence_transformers" | "openai"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_base_url: str = ""  # OpenAI-compatible embeddings endpoint
    embedding_api_key: str = ""

    # --- persistence (vector backend) ---
    vector_persist_path: str = ""  # empty = in-memory only

    # --- consolidation policy ---
    consolidate_every: int = 4  # promote short->long every N user turns

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            deepseek_base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            deepseek_timeout=int(os.getenv("DEEPSEEK_TIMEOUT", "60")),
            deepseek_max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
            long_term_backend=os.getenv("LONG_TERM_BACKEND", "vector"),
            lightrag_working_dir=os.getenv("LIGHTRAG_WORKING_DIR", ".data/lightrag"),
            mem0_embedder_provider=os.getenv("MEM0_EMBEDDER_PROVIDER", "huggingface"),
            mem0_embedding_dims=int(os.getenv("MEM0_EMBEDDING_DIMS", "384")),
            mem0_vector_path=os.getenv("MEM0_VECTOR_PATH", ".data/mem0_qdrant"),
            short_term_store=os.getenv("SHORT_TERM_STORE", "memory"),
            short_term_max_turns=int(os.getenv("SHORT_TERM_MAX_TURNS", "12")),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            redis_namespace=os.getenv("REDIS_NAMESPACE", "agentmem:st"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "auto"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", ""),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
            vector_persist_path=os.getenv("VECTOR_PERSIST_PATH", ""),
            consolidate_every=int(os.getenv("CONSOLIDATE_EVERY", "4")),
        )
