"""Benchmark the long-term backends on a shared dataset.

Two datasets:
  (default) EASY  — isolated facts, simple queries: mostly a LATENCY comparison.
  --hard          — distractors, temporal updates, multi-hop relations, and
                    negations, scored for RETRIEVAL QUALITY (hit@1 / hit@3 /
                    clean / MRR), which actually separates the backends.

Each backend runs in its OWN subprocess (mem0 and LightRAG interfere when they
share a process; isolation also keeps timings clean).

    set -a && . ./.env && set +a
    export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
    export TOKENIZERS_PARALLELISM=false
    export LETTA_EMBEDDING_ENDPOINT=http://host.docker.internal:11434/v1   # for letta
    python benchmark.py --hard            # quality, all backends
    python benchmark.py --hard vector mem0

Metrics (top-5 retrieved per query):
  hit@1  answer in the #1 result        hit@3  answer in the top 3
  clean  answer retrieved AND no stale/wrong fact ranked above it (precision)
  MRR    mean reciprocal rank of the first correct result
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, "src")

# (query, expected-answer substrings, stale/wrong substrings that must NOT outrank)
EASY_FACTS = [
    "I'm Maya, a pediatric nurse living in Lisbon.",
    "I am allergic to penicillin.",
    "My partner Tom is a chef at an Italian restaurant.",
    "I have a golden retriever named Biscuit.",
    "I'm training for the Lisbon marathon in October.",
    "I prefer vegetarian food and avoid dairy.",
    "My favourite programming language is Python.",
    "I drive a blue Tesla Model 3.",
]
EASY_QUERIES = [
    ("what medication is the user allergic to?", ["penicillin"], []),
    ("what pet does the user have?", ["biscuit", "retriever", "dog"], []),
    ("where does the user live?", ["lisbon"], []),
    ("what does the user's partner do for work?", ["chef", "cook"], []),
    ("what are the user's dietary preferences?", ["vegetarian", "dairy"], []),
    ("what car does the user drive?", ["tesla", "model 3"], []),
]

# Harder: each query has a competing wrong/stale fact in the store.
HARD_FACTS = [
    "I'm Maya Okonkwo, a pediatric cardiologist.",
    "I trained in Lisbon but relocated to Porto in early 2025.",        # current = Porto
    "Before that move, I had lived in Lisbon for six years.",           # stale: Lisbon
    "My younger sister Ada is severely allergic to peanuts.",           # distractor: sister
    "I am allergic to penicillin; I have no food allergies myself.",    # user allergy
    "My research mentor is Professor Adebayo.",                         # relation
    "Professor Adebayo runs the neuroimaging lab at the university.",   # multi-hop target
    "I used to commute by car, but since the move I bike everywhere.",  # update: bikes now
    "My partner Tomas manages a vineyard in the Douro valley.",         # relation
    "We adopted two cats last spring, Mochi and Pixel.",               # aggregation: 2
    "After years of Python, I'm now learning Rust.",                    # update: Rust now
    "I gave up coffee in 2024 and only drink rooibos tea these days.",  # negation: tea now
    "Ada, my sister, works as a marine biologist in Oslo.",            # distractor entity
    "I no longer own a car; I sold my old blue Tesla after moving.",   # stale: Tesla
    "My emergency contact is my partner Tomas.",
]
HARD_QUERIES = [
    ("which city does Maya live in now?", ["porto"], ["lisbon"]),                  # temporal
    ("what is Maya herself allergic to?", ["penicillin"], ["peanut"]),            # distractor
    ("what does Maya's research mentor's lab focus on?", ["neuroimaging"], []),    # multi-hop
    ("how does Maya get around since relocating?", ["bike", "bicycl", "cycl"], ["car", "drive", "tesla"]),  # update
    ("what does Maya's partner do for work?", ["vineyard", "wine"], []),           # relation
    ("how many cats does Maya have?", ["two", "mochi", "pixel"], []),             # aggregation
    ("what language is Maya learning now?", ["rust"], ["python"]),                 # update
    ("what does Maya drink these days?", ["rooibos", "tea"], ["coffee"]),          # negation
]


def run_one(backend: str, hard: bool) -> dict:
    from agentmem import Config, MemoryManager, Message

    facts = HARD_FACTS if hard else EASY_FACTS
    queries = HARD_QUERIES if hard else EASY_QUERIES

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
        t0 = time.perf_counter()
        stored = mm.long_term.add([Message("user", f) for f in facts], user_id=uid)
        write_s = time.perf_counter() - t0

        qtimes, rr, hit1, hit3, clean = [], [], 0, 0, 0
        for query, expected, avoid in queries:
            t = time.perf_counter()
            res = mm.recall(uid, query, k=5)
            qtimes.append(time.perf_counter() - t)
            texts = [(r.text or "").lower() for r in res]
            rank_e = next((i + 1 for i, tx in enumerate(texts) if any(e in tx for e in expected)), None)
            rank_a = next((i + 1 for i, tx in enumerate(texts) if any(a in tx for a in avoid)), None) if avoid else None
            if rank_e == 1:
                hit1 += 1
            if rank_e and rank_e <= 3:
                hit3 += 1
            if rank_e is not None and (rank_a is None or rank_e < rank_a):
                clean += 1
            rr.append(1.0 / rank_e if rank_e else 0.0)

        if backend == "letta":
            agent_id = mm.long_term._agent_id(uid)
        n = len(queries)
        return {
            "backend": backend, "ok": True, "stored": len(stored),
            "write_s": round(write_s, 1),
            "query_ms": round(1000 * sum(qtimes) / n, 1),
            "hit1": round(hit1 / n, 2), "hit3": round(hit3 / n, 2),
            "clean": round(clean / n, 2), "mrr": round(sum(rr) / n, 3),
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


def _print_table(rows: list[dict], hard: bool) -> None:
    print("\n" + "=" * 78)
    if hard:
        print(f"{'backend':<10}{'write(s)':>9}{'query(ms)':>10}{'hit@1':>7}{'hit@3':>7}{'clean':>7}{'MRR':>7}")
        print("-" * 78)
        for r in rows:
            if r.get("ok"):
                print(f"{r['backend']:<10}{r['write_s']:>9}{r['query_ms']:>10}"
                      f"{r['hit1']:>7}{r['hit3']:>7}{r['clean']:>7}{r['mrr']:>7}")
            else:
                print(f"{r['backend']:<10}{'ERROR':>9}  {r.get('error','')[:46]}")
        print("=" * 78)
        print(f"{len(HARD_FACTS)} facts / {len(HARD_QUERIES)} queries (distractors, updates, "
              "multi-hop, negation).")
        print("clean = answer retrieved with no stale/wrong fact ranked above it.")
    else:
        print(f"{'backend':<10}{'write(s)':>9}{'query(ms)':>10}{'hit@3':>7}{'stored':>8}")
        print("-" * 78)
        for r in rows:
            if r.get("ok"):
                print(f"{r['backend']:<10}{r['write_s']:>9}{r['query_ms']:>10}{r['hit3']:>7}{r['stored']:>8}")
            else:
                print(f"{r['backend']:<10}{'ERROR':>9}  {r.get('error','')[:46]}")
        print("=" * 78)
        print(f"{len(EASY_FACTS)} facts / {len(EASY_QUERIES)} queries — easy set saturates; use --hard.")


def main(backends: list[str], hard: bool) -> None:
    import subprocess

    rows = []
    for b in backends:
        print(f"running {b} (isolated subprocess){' [hard]' if hard else ''} ...", flush=True)
        argv = [sys.executable, __file__, "--one", b] + (["--hard"] if hard else [])
        p = subprocess.run(argv, env=os.environ, capture_output=True, text=True)
        lines = [l for l in p.stdout.splitlines() if l.strip().startswith("{")]
        rows.append(json.loads(lines[-1]) if lines else
                    {"backend": b, "ok": False, "error": (p.stderr or p.stdout)[-120:]})
    _print_table(rows, hard)


if __name__ == "__main__":
    hard = "--hard" in sys.argv
    rest = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--one" in sys.argv:
        main_backend = sys.argv[sys.argv.index("--one") + 1]
        print(json.dumps(run_one(main_backend, hard)))
    else:
        main(rest or ["vector", "mem0", "lightrag", "letta"], hard)
