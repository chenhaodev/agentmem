"""Offline smoke tests for agentmem. Run: python tests/test_smoke.py

No network, no API key, no external services required — exercises the in-process
short-term store, the plain vector backend (hashing embedder), the manager's
consolidation/dedup policy, and vector persistence round-trip. Plain asserts so
it runs without pytest, but `pytest tests/` also discovers the test_* functions.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentmem import Config, MemoryManager, Message  # noqa: E402
from agentmem.backends.base import LongTermBackend  # noqa: E402
from agentmem.short_term import (  # noqa: E402
    InProcessShortTermMemory,
    ShortTermMemory,
)


def _cfg(**kw) -> Config:
    # Force the fully-offline path regardless of any .env present.
    base = dict(
        long_term_backend="vector",
        short_term_store="memory",
        embedding_provider="hash",
        consolidate_every=4,
        deepseek_api_key="",  # no key -> extract_facts falls back to raw turns
    )
    base.update(kw)
    return Config(**base)


def test_short_term_ring_buffer():
    st = InProcessShortTermMemory(max_turns=3)
    for i in range(5):
        st.append("s", Message("user", f"m{i}"))
    recent = st.recent("s")
    assert [m.content for m in recent] == ["m2", "m3", "m4"], recent
    assert isinstance(st, ShortTermMemory)  # satisfies the Protocol
    print("ok  short_term ring buffer + Protocol")


def test_vector_backend_contract():
    mm = MemoryManager(_cfg())
    assert isinstance(mm.long_term, LongTermBackend)
    mm.remember("u", "I am vegetarian")
    mm.remember("u", "I live in Berlin")
    hits = mm.recall("u", "what does the user eat", k=2)
    assert hits, "expected at least one recall hit"
    assert any("vegetarian" in h.text for h in hits), hits
    assert len(mm.long_term.get_all("u")) == 2
    print("ok  vector backend add/search/get_all contract")


def test_consolidation_dedup():
    """The reported bug: a mid-session consolidate + end_session must not
    double-store the same turns."""
    mm = MemoryManager(_cfg(consolidate_every=4))
    for i in range(4):  # hits consolidate_every -> one consolidation
        mm.add_turn("s", "u", Message("user", f"fact number {i}"))
    end = mm.end_session("s", "u")  # nothing pending -> no re-store
    assert end == [], f"end_session re-stored turns: {end}"
    all_mem = mm.long_term.get_all("u")
    texts = [m.text for m in all_mem]
    assert len(texts) == len(set(texts)), f"duplicates found: {texts}"
    assert len(texts) == 4, texts
    print("ok  consolidation does not duplicate memories")


def test_build_context_shape():
    mm = MemoryManager(_cfg())
    mm.add_turn("s", "u", Message("user", "I love hiking"))
    ctx = mm.build_context("s", "u", query="outdoor", k=3)
    assert ctx.recent and ctx.recent[-1].content == "I love hiking"
    block = ctx.as_prompt_block()
    assert "Recent conversation" in block
    print("ok  build_context returns recent + memories block")


def test_async_consolidation_drains_and_dedups():
    """Background consolidation: add_turn returns without persisting inline; a
    flush makes the memories searchable; no duplicates; errors are surfaced."""
    import threading

    mm = MemoryManager(_cfg(consolidation_async=True, consolidate_every=4))
    try:
        assert mm._worker is not None
        for i in range(4):  # 4th user turn submits a background job
            mm.add_turn("s", "u", Message("user", f"durable fact {i}"))
        # the worker is a separate thread, not the caller's
        assert mm._worker._thread is not threading.current_thread()
        mm.flush()  # block until the background job is done
        mem = mm.long_term.get_all("u")
        texts = [m.text for m in mem]
        assert len(texts) == 4, texts
        assert len(texts) == len(set(texts)), f"duplicates: {texts}"
        assert mm.consolidation_errors == [], mm.consolidation_errors
        # end_session on an async manager must also drain cleanly
        mm.add_turn("s", "u", Message("user", "one more fact"))
        end = mm.end_session("s", "u")
        assert any("one more" in m.text for m in end), end
        print("ok  async consolidation drains, dedups, surfaces errors")
    finally:
        mm.close()


def test_unknown_backend_errors_clearly():
    from agentmem.backends import build_backend
    from agentmem.llm import DeepSeekLLM

    cfg = _cfg(long_term_backend="nope")
    try:
        build_backend(cfg, DeepSeekLLM(cfg))
        assert False, "expected ValueError for unknown backend"
    except ValueError as e:
        assert "lightrag" in str(e)  # error lists all known backends
    print("ok  unknown backend raises a helpful ValueError")


def test_lightrag_optional_dep_message():
    """LightRAG selected but not installed must fail with our guidance, not a
    raw ModuleNotFoundError (skips cleanly if lightrag IS installed)."""
    from agentmem.backends import build_backend
    from agentmem.llm import DeepSeekLLM

    try:
        import lightrag  # noqa: F401

        print("skip lightrag installed; dep-message path not exercised")
        return
    except ImportError:
        pass
    cfg = _cfg(long_term_backend="lightrag")
    try:
        build_backend(cfg, DeepSeekLLM(cfg))
        assert False, "expected ImportError"
    except ImportError as e:
        assert "lightrag-hku" in str(e)
    print("ok  lightrag missing-dep gives install guidance")


def test_vector_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.json")
        mm1 = MemoryManager(_cfg(vector_persist_path=path))
        mm1.remember("u", "persistent fact about cats")
        assert os.path.exists(path), "persist file not written"
        # New manager, same path -> memories survive the restart.
        mm2 = MemoryManager(_cfg(vector_persist_path=path))
        hits = mm2.recall("u", "cats", k=1)
        assert hits and "cats" in hits[0].text, hits
    print("ok  vector persistence survives restart")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
