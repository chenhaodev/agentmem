"""Opt-in LIVE integration tests — real DeepSeek calls + installed backends.

These are SKIPPED unless RUN_LIVE=1, because they need network, a DeepSeek key,
and the optional backend deps installed. They encode the setups verified by hand
so the mem0 and lightrag adapters don't silently rot.

Run:
    set -a && . ./.env && set +a
    export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
    export TOKENIZERS_PARALLELISM=false RUN_LIVE=1
    python tests/test_live.py

Verified live (2026-06-11): mem0 2.0.5, lightrag-hku 1.5.2, DeepSeek-V4-Flash.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentmem import Config, MemoryManager, Message  # noqa: E402

LIVE = os.getenv("RUN_LIVE") == "1"


def _skip(name: str) -> bool:
    if not LIVE:
        print(f"skip {name} (set RUN_LIVE=1 to run)")
        return True
    if not os.getenv("DEEPSEEK_API_KEY"):
        print(f"skip {name} (no DEEPSEEK_API_KEY)")
        return True
    return False


def test_mem0_live():
    """DeepSeek extraction + HF embeddings + embedded Qdrant, via MemoryManager."""
    if _skip("mem0"):
        return
    cfg = Config.from_env()
    cfg.long_term_backend = "mem0"
    cfg.mem0_vector_path = tempfile.mkdtemp(prefix="mem0q_")
    mm = MemoryManager(cfg)
    mm.add_turn("s", "bob", Message("user", "I'm Bob and I'm allergic to peanuts"))
    mm.end_session("s", "bob")
    hits = mm.recall("bob", "what is the user allergic to", k=3)
    assert hits, "no memories returned"
    assert any("peanut" in h.text.lower() for h in hits), [h.text for h in hits]
    print("ok  mem0 live: extracted + recalled peanut allergy")


def test_lightrag_live():
    """LightRAG builds a knowledge graph and retrieves relational context."""
    if _skip("lightrag"):
        return
    cfg = Config.from_env()
    cfg.long_term_backend = "lightrag"
    cfg.embedding_provider = "sentence_transformers"
    cfg.lightrag_working_dir = tempfile.mkdtemp(prefix="lrag_")
    mm = MemoryManager(cfg)
    mm.remember("u", "Carol lives in Tokyo and her teammate is Dave.")
    hits = mm.recall("u", "where does Carol live", k=2)
    assert hits and "Tokyo" in hits[0].text, hits
    # graph store: get_all/delete are intentionally unsupported
    for fn in (lambda: mm.long_term.get_all("u"),
               lambda: mm.long_term.delete("x", "u")):
        try:
            fn()
            assert False, "expected NotImplementedError"
        except NotImplementedError:
            pass
    print("ok  lightrag live: graph built + relational recall + contract")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} live test(s) processed (RUN_LIVE={LIVE})")


if __name__ == "__main__":
    main()
