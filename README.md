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
      ├─ vector   plain numpy cosine + pluggable embedder + disk persistence ← default
      ├─ mem0     mem0 library, DeepSeek as its extraction LLM
      └─ letta    Letta archival memory (needs a Letta server)
 └─ DeepSeekLLM     one OpenAI-compatible client, injected where extraction is needed
```

Embedders for the vector backend (`EMBEDDING_PROVIDER`): `auto` (local
sentence-transformers → hashing fallback), `hash` (offline, zero-dep),
`sentence_transformers`, or `openai` (any OpenAI-compatible `/embeddings`
endpoint via `EMBEDDING_BASE_URL`).

Write path: `add_turn` → short-term (always) → consolidate to long-term every
`CONSOLIDATE_EVERY` user turns (extraction happens here, off the hot path).
Read path: `build_context` → recent raw turns **+** relevance-ranked memories.

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
LONG_TERM_BACKEND=mem0   python demo.py   # pip install "mem0ai>=0.1.40"
LONG_TERM_BACKEND=letta  python demo.py   # pip install "letta-client>=0.1" + Letta server + LETTA_AGENT_ID

# Share short-term memory across processes:
SHORT_TERM_STORE=redis REDIS_URL=redis://localhost:6379/0 python demo.py  # pip install redis
```

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
| `LONG_TERM_BACKEND` | `vector` | `vector` \| `mem0` \| `letta` |
| `SHORT_TERM_MAX_TURNS` | `12` | raw turns kept per session |
| `CONSOLIDATE_EVERY` | `4` | promote short→long every N user turns |

## Notes & current limits

- **Offline-friendly:** with no network/key, fact extraction falls back to raw
  user turns and the vector backend uses a deterministic **hashing embedder**
  (lexical, not semantic — so some recall scores are ~0). Install
  `sentence-transformers` or supply a live `DEEPSEEK_API_KEY` for real quality.
- The `vector` backend is in-memory by default; set `VECTOR_PERSIST_PATH` to
  persist to disk (atomic JSON write, auto-loaded on startup).
- Short-term sharing across processes: set `SHORT_TERM_STORE=redis`. Within a
  single process the default `memory` store already shares across agents.

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
