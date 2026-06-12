"""Benchmark the long-term backends on one shared dataset.

Measures, per backend: write latency (store all facts), average query latency,
and recall@3 (did the top-3 results contain the answer). Each backend runs in
its OWN subprocess — mem0 and LightRAG must not share a process (heavy ML stacks
interfere), and isolation keeps timings clean.

    set -a && . ./.env && set +a
    export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
    export TOKENIZERS_PARALLELISM=false
    export LETTA_EMBEDDING_ENDPOINT=http://host.docker.internal:11434/v1   # for letta
    python benchmark.py                 # vector mem0 lightrag letta
    python benchmark.py vector mem0     # subset

Note: "stored" differs by design — vector/mem0 store *extracted* facts, letta
stores raw passages, lightrag stores one graph doc. Compare write/query/recall.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, "src")

FACTS = [
    "I'm Maya, a pediatric nurse living in Lisbon.",
    "I am allergic to penicillin.",
    "My partner Tom is a chef at an Italian restaurant.",
    "I have a golden retriever named Biscuit.",
    "I'm training for the Lisbon marathon in October.",
    "I prefer vegetarian food and avoid dairy.",
    "My favourite programming language is Python.",
    "I drive a blue Tesla Model 3.",
]

# (query, acceptable answer substrings — case-insensitive)
QUERIES = [
    ("what medication is the user allergic to?", ["penicillin"]),
    ("what pet does the user have?", ["biscuit", "retriever", "dog"]),
    ("where does the user live?", ["lisbon"]),
    ("what does the user's partner do for work?", ["chef", "cook"]),
    ("what are the user's dietary preferences?", ["vegetarian", "dairy"]),
    ("what car does the user drive?", ["tesla", "model 3"]),
]

ALL_BACKENDS = ["vector", "mem0", "lightrag", "letta"]


def run_one(backend: str) -> dict:
    from agentmem import Config, MemoryManager, Message

    cfg = Config.from_env()
    cfg.long_term_backend = backend
    cfg.consolidation_async = False
    if backend in ("vector", "lightrag"):
        cfg.embedding_provider = "sentence_transformers"
    if backend == "lightrag":
        cfg.lightrag_working_dir = tempfile.mkdtemp(prefix="bench_lrag_")
    if backend == "mem0":
        cfg.mem0_vector_path = tempfile.mkdtemp(prefix="bench_mem0_")

    uid = f"bench-{backend}"
    mm = MemoryManager(cfg)
    agent_id = None
    try:
        msgs = [Message("user", f) for f in FACTS]
        t0 = time.perf_counter()
        stored = mm.long_term.add(msgs, user_id=uid)
        write_s = time.perf_counter() - t0

        qtimes, hits = [], 0
        for query, expected in QUERIES:
            t = time.perf_counter()
            res = mm.recall(uid, query, k=3)
            qtimes.append(time.perf_counter() - t)
            blob = " ".join((r.text or "").lower() for r in res)
            if any(e in blob for e in expected):
                hits += 1

        if backend == "letta":
            agent_id = mm.long_term._agent_id(uid)
        return {
            "backend": backend,
            "ok": True,
            "facts": len(FACTS),
            "stored": len(stored),
            "write_s": round(write_s, 2),
            "query_ms": round(1000 * sum(qtimes) / len(qtimes), 1),
            "recall_at3": round(hits / len(QUERIES), 2),
        }
    except Exception as e:
        return {"backend": backend, "ok": False, "error": repr(e)[:300]}
    finally:
        if agent_id:
            try:
                mm.long_term._client.agents.delete(agent_id=agent_id)
            except Exception:
                pass
        mm.close()


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 64)
    print(f"{'backend':<10} {'write(s)':>9} {'query(ms)':>10} {'recall@3':>9} {'stored':>7}")
    print("-" * 64)
    for r in rows:
        if r.get("ok"):
            print(
                f"{r['backend']:<10} {r['write_s']:>9} {r['query_ms']:>10} "
                f"{r['recall_at3']:>9} {r['stored']:>7}"
            )
        else:
            print(f"{r['backend']:<10} {'ERROR':>9}  {r.get('error', '')[:38]}")
    print("=" * 64)
    print(f"dataset: {len(FACTS)} facts, {len(QUERIES)} queries  |  recall@3 = top-3 contains answer")


def main(backends: list[str]) -> None:
    import subprocess

    rows = []
    for b in backends:
        print(f"running {b} (isolated subprocess) ...", flush=True)
        p = subprocess.run(
            [sys.executable, __file__, "--one", b],
            env=os.environ, capture_output=True, text=True,
        )
        lines = [l for l in p.stdout.splitlines() if l.strip().startswith("{")]
        if lines:
            rows.append(json.loads(lines[-1]))
        else:
            rows.append({"backend": b, "ok": False, "error": (p.stderr or p.stdout)[-120:]})
    _print_table(rows)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--one":
        print(json.dumps(run_one(sys.argv[2])))
    else:
        chosen = [a for a in sys.argv[1:] if not a.startswith("-")] or ALL_BACKENDS
        main(chosen)
