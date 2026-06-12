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


def test_mem0_async_consolidation_live():
    """add_turn must return immediately while DeepSeek extraction runs in the
    background; flush() then makes the memory searchable."""
    if _skip("mem0-async"):
        return
    import time

    cfg = Config.from_env()
    cfg.long_term_backend = "mem0"
    cfg.consolidation_async = True
    cfg.consolidate_every = 1
    cfg.mem0_vector_path = tempfile.mkdtemp(prefix="mem0q_")
    with MemoryManager(cfg) as mm:
        t0 = time.time()
        mm.add_turn("s", "bob", Message("user", "I'm Bob, allergic to peanuts"))
        elapsed_ms = (time.time() - t0) * 1000
        assert elapsed_ms < 250, f"add_turn blocked for {elapsed_ms:.0f}ms"
        mm.flush()
        assert mm.consolidation_errors == [], mm.consolidation_errors
        hits = mm.recall("bob", "allergy", k=3)
        assert any("peanut" in h.text.lower() for h in hits), [h.text for h in hits]
    print("ok  mem0 async: hot path non-blocking, flush drains, recall works")


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


def test_redis_short_term_cross_process():
    """The point of Redis short-term is cross-process sharing: a CHILD process
    writes turns, this parent reads the same session, and the ring buffer cap
    holds across the boundary. Needs a reachable Redis (skips otherwise)."""
    if not LIVE:
        print("skip redis (set RUN_LIVE=1 to run)")
        return
    import subprocess

    import redis as _redis

    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        _redis.Redis.from_url(url).ping()
    except Exception as e:
        print(f"skip redis (not reachable at {url}: {e})")
        return

    from agentmem import Config
    from agentmem.short_term import build_short_term

    cfg = Config.from_env()
    cfg.short_term_store = "redis"
    cfg.short_term_max_turns = 3
    session = "redis-xproc"
    parent = build_short_term(cfg)
    parent.clear(session)

    # child process writes 4 turns into the same Redis session
    child = (
        "import sys; sys.path.insert(0,'src')\n"
        "from agentmem import Config\n"
        "from agentmem.short_term import build_short_term\n"
        "from agentmem.types import Message\n"
        "c=Config.from_env(); c.short_term_store='redis'; c.short_term_max_turns=3\n"
        "st=build_short_term(c)\n"
        "for x in ['a','b','c','d']: st.append('redis-xproc', Message('user', x))\n"
    )
    subprocess.run([sys.executable, "-c", child], env=os.environ, check=True)

    # parent (separate process) sees the child's writes — cross-process sharing
    got = [m.content for m in parent.recent(session)]
    assert got == ["b", "c", "d"], f"cross-process/ring-buffer wrong: {got}"
    assert session in parent.sessions()
    parent.clear(session)
    print("ok  redis short-term: cross-process sharing + ring buffer cap")


def test_router_vector_mem0_live():
    """Cross-backend routing: fan out a write to vector + mem0, merge reads."""
    if _skip("router"):
        return
    cfg = Config.from_env()
    cfg.long_term_backend = "vector+mem0"
    cfg.embedding_provider = "sentence_transformers"
    cfg.mem0_vector_path = tempfile.mkdtemp(prefix="mem0q_")
    mm = MemoryManager(cfg)
    mm.add_turn("s", "dana", Message("user", "I am Dana and I am lactose intolerant."))
    stored = mm.end_session("s", "dana")
    backends = {m.metadata.get("backend") for m in stored}
    assert backends == {"vector", "mem0"}, f"fan-out incomplete: {backends}"
    hits = mm.recall("dana", "what food issues does the user have", k=5)
    assert hits and {h.metadata.get("backend") for h in hits} & {"vector", "mem0"}
    assert mm.long_term.errors == [], mm.long_term.errors
    print("ok  router live: fan-out to vector+mem0, merged provenance-tagged read")


def main():
    """Run each live test in its OWN subprocess.

    mem0 and LightRAG are heavy ML stacks; running them in one process — and
    especially running mem0 in a background thread after LightRAG has used the
    process — interferes (shared torch/threading state). Real deployments use one
    long-term backend per process, so we isolate each test the same way.
    """
    import subprocess

    names = [k for k in sorted(globals()) if k.startswith("test_")]
    if not LIVE:  # nothing to isolate; just report the skips inline
        for name in names:
            globals()[name]()
        print(f"\n{len(names)} live test(s) skipped (set RUN_LIVE=1)")
        return
    failed = []
    for name in names:
        print(f"\n=== {name} (isolated subprocess) ===")
        rc = subprocess.run([sys.executable, __file__, name], env=os.environ).returncode
        if rc != 0:
            failed.append(name)
    print(f"\n{len(names) - len(failed)}/{len(names)} live tests passed")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:  # child: run one named test in this fresh process
        globals()[sys.argv[1]]()
    else:
        main()
