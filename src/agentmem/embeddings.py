"""Pluggable embedders for the plain vector-store backend.

Default is a zero-dependency deterministic hashing embedder so the demo runs
fully offline. Swap in SentenceTransformerEmbedder for real semantic quality.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:  # (n, dim), L2-normalized
        ...


class HashingEmbedder:
    """Hashed bag-of-words embedding. No model download, deterministic, offline.

    Quality is modest (lexical, not semantic) but enough to demonstrate
    retrieval end to end. Good for tests/demos; use a real model in production.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class SentenceTransformerEmbedder:
    """Real semantic embeddings, run locally. Requires sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True), dtype=np.float32
        )


class OpenAICompatibleEmbedder:
    """Embeddings via any OpenAI-compatible /embeddings endpoint.

    Lets you plug a hosted embeddings provider (set EMBEDDING_BASE_URL /
    EMBEDDING_API_KEY / EMBEDDING_MODEL) without changing code. Vectors are
    L2-normalized so the vector backend's dot product is cosine similarity.
    """

    def __init__(self, model: str, base_url: str = "", api_key: str = ""):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key or None, base_url=base_url or None)
        self.dim = -1  # learned from the first response

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        self.dim = vecs.shape[1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def build_embedder(config) -> Embedder:
    """Select an embedder from config (EMBEDDING_PROVIDER).

    'auto' prefers a local sentence-transformers model, then falls back to the
    zero-dependency offline hashing embedder so things always work.
    """
    provider = (getattr(config, "embedding_provider", "auto") or "auto").lower()
    if provider == "hash":
        return HashingEmbedder()
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(config.embedding_model)
    if provider == "openai":
        return OpenAICompatibleEmbedder(
            config.embedding_model, config.embedding_base_url, config.embedding_api_key
        )
    # auto
    try:
        return SentenceTransformerEmbedder(config.embedding_model)
    except Exception:
        return HashingEmbedder()


def default_embedder() -> Embedder:
    """Prefer a real local model if installed, else the offline hashing fallback."""
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return HashingEmbedder()
