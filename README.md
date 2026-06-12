# agentmem — a switchable agent-memory middleware

One interface over multiple agent-memory frameworks, with **short-term** and
**long-term** memory cleanly separated. Powered by **DeepSeek-V4-Flash**
(OpenAI-compatible). Switch the long-term backend with a single env var.

## Why this shape

- **Short-term ≠ long-term.** Short-term is the recent raw-turn buffer — kept
  lightweight and framework-agnostic (no LLM on the hot path), living in a shared
  namespace so multiple agents/handlers in a process see the same recent context.
- **Long-term is pluggable.** The heavy frameworks (mem0, Letta, …) only matter
  here. They sit behind one `LongTermBackend` contract, so swapping them is a
  config change, not a code change.
- **Common interface, not common features.** The contract is the lowest common
  denominator (`add / search / get_all / delete` over `{text, metadata, score}`).
  Backend-specific superpowers ride along in `MemoryItem.metadata`.

```
MemoryManager (facade)
 ├─ ShortTermMemory   pluggable, shared, no LLM                 (short_term.py)
 │    ├─ memory   in-process ring buffer (zero deps)  ← default
 │    └─ redis    Redis list per session — shared across processes/machines
 └─ LongTermBackend   pluggable, LLM-backed extraction          (backends/)
      ├─ vector    plain numpy cosine + pluggable embedder + disk persistence ← default
      ├─ mem0      mem0 library, DeepSeek as its extraction LLM
      ├─ lightrag  knowledge-graph memory; builds its own entity/relation graph
      ├─ letta     Letta archival memory (needs a Letta server)
      └─ router    fan-out writes + merged reads across several of the above (a+b)
 └─ DeepSeekLLM     one OpenAI-compatible client, injected where extraction is needed
```

Embedders for the vector backend (`EMBEDDING_PROVIDER`): `auto` (local
sentence-transformers → hashing fallback), `hash` (offline, zero-dep),
`sentence_transformers`, or `openai` (any OpenAI-compatible `/embeddings`
endpoint via `EMBEDDING_BASE_URL`). `sentence_transformers` runs on CPU — no
GPU needed. On older macOS see `requirements-local-cpu.txt` for the pinned,
verified-working dependency set.

Write path: `add_turn` → short-term (always) → consolidate to long-term every
`CONSOLIDATE_EVERY` user turns. Read path: `build_context` → recent raw turns
**+** relevance-ranked memories.

### Async consolidation

Consolidation runs the LLM (fact extraction / graph building), which is slow. Set
`CONSOLIDATION_ASYNC=1` to run it on a background worker so `add_turn` returns
immediately (verified: ~0–1 ms vs seconds inline):

```python
cfg = Config.from_env(); cfg.consolidation_async = True
with MemoryManager(cfg) as mm:                 # context manager stops the worker
    mm.add_turn("s", "u", Message("user", "..."))   # returns instantly
    mm.flush()                                  # wait for background extraction
    mm.recall("u", "...")                       # now searchable
    # end_session() also drains automatically; mm.consolidation_errors surfaces failures
```

A single worker thread serializes all long-term access (the backends aren't
thread-safe), so reads and the background write never race. **Run one long-term
backend per process** — mem0 and LightRAG are heavy ML stacks that interfere when
sharing a process (the opt-in live tests isolate each in its own subprocess).

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # add your DEEPSEEK_API_KEY
python demo.py              # default: vector backend, zero extra deps
```

```python
from agentmem import MemoryManager, Message

mm = MemoryManager()                                   # backend from env
mm.add_turn("sess1", "alice", Message("user", "I'm vegetarian and love the Alps"))
mm.end_session("sess1", "alice")                       # flush short->long
print(mm.recall("alice", "what food does she eat?"))
ctx = mm.build_context("sess1", "alice", query="trip ideas")
prompt = ctx.as_prompt_block()                         # feed into your next LLM call
```

## Switching backends

```bash
LONG_TERM_BACKEND=vector python demo.py   # default, no extra install
LONG_TERM_BACKEND=mem0     python demo.py   # pip install "mem0ai>=2.0" qdrant-client sentence-transformers
LONG_TERM_BACKEND=lightrag python demo.py   # pip install "lightrag-hku>=1.0"  (graph memory)
LONG_TERM_BACKEND=letta    python demo.py   # pip install "letta-client>=0.1" + Letta server + LETTA_AGENT_ID
```

Both `mem0` and `lightrag` are verified live against DeepSeek-V4-Flash
(mem0 2.0.5 = DeepSeek + HuggingFace MiniLM embeddings + embedded Qdrant;
lightrag-hku 1.5.2 = knowledge graph). Reproduce with the opt-in live tests:

```bash
set -a && . ./.env && set +a
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
export TOKENIZERS_PARALLELISM=false RUN_LIVE=1
python tests/test_live.py

# Share short-term memory across processes (verified live):
docker run -d --name agentmem-redis -p 6379:6379 redis:7-alpine   # pip install redis
SHORT_TERM_STORE=redis REDIS_URL=redis://localhost:6379/0 python demo.py

# Store long memory in several frameworks at once (cross-backend routing):
LONG_TERM_BACKEND=vector+mem0 python demo.py
```

### Cross-backend routing

Set `LONG_TERM_BACKEND` to a `+`-list to wrap several backends in a router that
**fans out writes** and **merges reads** — "long memory in different frameworks
at once". A router *is* a `LongTermBackend`, so nothing else changes.

- **Write:** fans out to all children (`ROUTER_WRITE=first` to write only the
  first). Route a single write to one child with `add(..., metadata={"backend":
  "mem0"})`. A child failing doesn't lose the write to the others.
- **Read:** queries every child and merges. Scores aren't comparable across
  heterogeneous backends, so the default `ROUTER_MERGE=interleave` (round-robin)
  gives balanced results; `ROUTER_MERGE=score` sorts by raw score. Results are
  deduped by text and tagged with `metadata["backend"]` for provenance.
- Verified live: `vector+mem0` fans out and returns merged, provenance-tagged
  memories. (Don't combine mem0 + lightrag in one process — see below.)

## Tests

```bash
python tests/test_smoke.py     # fully offline; also discoverable via `pytest tests/`
```

## Configuration (env / `.env`)

| Var | Default | Meaning |
|-----|---------|---------|
| `DEEPSEEK_API_KEY` | — | DeepSeek key (OpenAI-compatible endpoint) |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | model id |
| `DEEPSEEK_API_BASE` | `https://api.deepseek.com` | base url |
| `LONG_TERM_BACKEND` | `vector` | `vector` \| `mem0` \| `lightrag` \| `letta` \| a `+`-list e.g. `vector+mem0` |
| `ROUTER_MERGE` | `interleave` | cross-backend read merge: `interleave` \| `score` |
| `ROUTER_WRITE` | `all` | cross-backend write fan-out: `all` \| `first` |
| `LIGHTRAG_WORKING_DIR` | `.data/lightrag` | per-user graph dir (lightrag backend) |
| `SHORT_TERM_MAX_TURNS` | `12` | raw turns kept per session |
| `CONSOLIDATE_EVERY` | `4` | promote short→long every N user turns |
| `CONSOLIDATION_ASYNC` | `0` | run consolidation on a background worker (`1` to enable) |

## Notes & current limits

- **Offline-friendly:** with no network/key, fact extraction falls back to raw
  user turns and the vector backend uses a deterministic **hashing embedder**
  (lexical, not semantic — so some recall scores are ~0). Install
  `sentence-transformers` or supply a live `DEEPSEEK_API_KEY` for real quality.
- The `vector` backend is in-memory by default; set `VECTOR_PERSIST_PATH` to
  persist to disk (atomic JSON write, auto-loaded on startup).
- Short-term sharing across processes: set `SHORT_TERM_STORE=redis` (verified
  live — a write in one process is seen by another via a Redis list per session,
  ring-buffer capped by `SHORT_TERM_MAX_TURNS`). Within a single process the
  default `memory` store already shares across agents.
- The `lightrag` backend builds a knowledge graph and does its own extraction,
  so it supports **add + search** (search returns graph-aware context). It does
  **not** support `get_all`/`delete` (no flat memory list) — reset by removing a
  user's dir under `LIGHTRAG_WORKING_DIR`.

## Adding a backend

Implement the `LongTermBackend` Protocol (`backends/base.py`) — `add`, `search`,
`get_all`, `delete` returning `MemoryItem`s — and register it in
`backends/__init__.py::build_backend`. That's the whole extension surface.

## Layout

```
src/agentmem/
  manager.py        facade: short+long orchestration, consolidation policy
  short_term.py     in-process shared recency buffer
  llm.py            DeepSeek-V4-Flash client + fact extraction
  embeddings.py     hashing (default) / sentence-transformers embedders
  config.py         env-driven config
  types.py          Message, MemoryItem
  backends/
    base.py         LongTermBackend Protocol
    vector.py       plain numpy backend (default)
    mem0_backend.py mem0 adapter
    letta_backend.py Letta adapter
demo.py             runnable end-to-end demo
```
